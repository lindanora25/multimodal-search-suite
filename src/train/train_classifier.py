"""
train_classifier.py

Trains a product category classifier on top of image embeddings.

Pipeline:
1) Load OpenCLIP backbone (optionally retrieval-tuned checkpoint)
2) Freeze backbone
3) Compute image embeddings on the fly
4) Train a small classifier head: embedding -> category logits

Metrics:
- Top-K accuracy (Top-1/3/5)
- Macro-F1 (important because categories are imbalanced)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List

import yaml
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from torch.amp import autocast

from sklearn.metrics import f1_score

from src.models.openclip_backbone import load_openclip
from src.models.classifier_head import ClassifierHead, save_classifier_checkpoint


# -------------------------
# Utilities
# -------------------------


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# -------------------------
# Dataset
# -------------------------


class ParquetImageLabelDataset(Dataset):
    """
    Expects parquet with:
      - image_path
      - label_col (string label)
    We map string labels -> integer ids for training.
    """

    def __init__(
        self,
        parquet_path: str,
        image_col: str,
        label_col: str,
        preprocess,
        label_to_id: Dict[str, int],
    ):
        self.df = pd.read_parquet(parquet_path)
        self.image_col = image_col
        self.label_col = label_col
        self.preprocess = preprocess
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row[self.image_col]

        label_str = " ".join(str(row[self.label_col]).split())
        y = self.label_to_id[label_str]

        img = Image.open(img_path).convert("RGB")
        x = self.preprocess(img)
        return x, y


def collate(batch):
    xs, ys = [], []
    for x, y in batch:
        if x is None:
            continue
        xs.append(x)  # CPU tensor from preprocess
        ys.append(y)  # int label
    if len(xs) == 0:
        return None
    xs = torch.stack(xs, dim=0)  # CPU
    ys = torch.tensor(ys, dtype=torch.long)  # CPU
    return xs, ys


# -------------------------
# Metrics
# -------------------------


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """
    logits: (B, C)
    targets: (B,)
    """
    topk = logits.topk(k, dim=-1).indices  # (B, k)
    correct = (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    return float(correct)


@torch.no_grad()
def eval_classifier(
    clip_model,
    head: ClassifierHead,
    loader: DataLoader,
    device: str,
    topk_list: List[int],
) -> Dict[str, float]:
    clip_model.eval()
    head.eval()

    all_preds = []
    all_targets = []

    topk_sums = {k: 0.0 for k in topk_list}
    n = 0

    for batch in tqdm(loader, desc="eval", leave=False):
        if batch is None:
            continue
        images, targets = batch

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        amp = True
        use_amp = str(device).startswith("cuda")
        with autocast("cuda", enabled=use_amp):
            # embeddings from CLIP backbone
            img_emb = clip_model.encode_image(images)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            logits = head(img_emb)

        n_batch = targets.size(0)
        n += n_batch

        for k in topk_list:
            topk_sums[k] += topk_accuracy(logits, targets, k) * n_batch

        preds = torch.argmax(logits, dim=-1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    metrics = {f"top{k}_acc": float(topk_sums[k] / n) for k in topk_list}
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    return metrics


# -------------------------
# Main training loop
# -------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    device = cfg["run"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["run"].get("amp", True)) and str(device).startswith("cuda")
    set_seed(int(cfg["run"].get("seed", 42)))

    data_cfg = cfg["data"]
    train_path = data_cfg["train_path"]
    val_path = data_cfg["val_path"]
    test_path = data_cfg["test_path"]
    image_col = data_cfg["image_col"]
    label_col = data_cfg["label_col"]

    # Build label mapping from train set (stable & reproducible)
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    def norm_label(x):
        # Strip an collapse whitespace: avoids hidden space label mismatches
        return " ".join(str(x).split())

    all_labels = pd.concat(
        [train_df[label_col], val_df[label_col], test_df[label_col]], axis=0
    )

    labels = sorted({norm_label(x) for x in all_labels.dropna().tolist()})

    label_to_id = {lab: i for i, lab in enumerate(labels)}
    num_classes = len(labels)

    # Load CLIP backbone with retrieval-tuned weights
    model_cfg = cfg["model"]
    bundle = load_openclip(
        arch=model_cfg["arch"],
        pretrained=model_cfg["pretrained"],
        device=device,
        ckpt_path=model_cfg.get("base_ckpt"),
    )
    clip_model = bundle.model
    preprocess = bundle.preprocess

    # Freeze backbone
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model.eval()

    # Determine embedding dimension by running a tiny forward pass
    with torch.no_grad():
        dummy = torch.zeros(
            1,
            3,
            data_cfg.get("image_size", 224),
            data_cfg.get("image_size", 224),
            device=device,
        )
        emb = clip_model.encode_image(dummy)
        embed_dim = int(emb.shape[-1])

    # Build head
    head_cfg = cfg.get("head", {})
    dropout = float(head_cfg.get("dropout", 0.1))
    head = ClassifierHead(
        embed_dim=embed_dim, num_classes=num_classes, dropout=dropout
    ).to(device)

    # DataLoaders
    batch_size = int(cfg["train"].get("batch_size", 128))
    num_workers = int(data_cfg.get("num_workers", 4))

    train_ds = ParquetImageLabelDataset(
        train_path, image_col, label_col, preprocess, label_to_id
    )
    val_ds = ParquetImageLabelDataset(
        val_path, image_col, label_col, preprocess, label_to_id
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )

    # Optimizer
    lr = float(cfg["train"].get("lr", 1e-3))
    wd = float(cfg["train"].get("weight_decay", 0.01))
    optim = AdamW(head.parameters(), lr=lr, weight_decay=wd)

    epochs = int(cfg["train"].get("epochs", 5))
    log_every = int(cfg["train"].get("log_every", 50))
    eval_every = int(cfg["train"].get("eval_every", 500))

    topk_list = cfg.get("metrics", {}).get("topk", [1, 3, 5])

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Output checkpoint
    out_cfg = cfg["outputs"]
    ckpt_dir = Path(out_cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / out_cfg["ckpt_name"]

    # Train
    global_step = 0
    head.train()
    best_top1 = -1.0
    best_metrics = None

    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_loader, desc=f"classifier epoch {epoch}/{epochs}")
        running = 0.0

        for step, batch in enumerate(pbar, start=1):
            if batch is None:
                continue
            images, targets = batch

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.no_grad():
                img_emb = clip_model.encode_image(images)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)

            with autocast("cuda", enabled=amp):
                logits = head(img_emb)
                loss = F.cross_entropy(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            running += float(loss.item())
            global_step += 1

            if global_step % log_every == 0:
                pbar.set_postfix({"loss": running / log_every})
                running = 0.0

            if global_step % eval_every == 0:
                head.eval()
                metrics = eval_classifier(
                    clip_model, head, val_loader, device=device, topk_list=topk_list
                )
                print(f"\n[eval step={global_step}] {metrics}\n")
                head.train()

        head.eval()
        metrics = eval_classifier(
            clip_model, head, val_loader, device=device, topk_list=topk_list
        )
        print(f"\n[eval epoch={epoch}] {metrics}\n")
        head.train()

        # --- save per-epoch checkpoint ---
        epoch_ckpt = ckpt_dir / f"openclip_abo_classifier_epoch{epoch}.pt"
        save_classifier_checkpoint(
            path=str(epoch_ckpt),
            head=head,
            labels=labels,
            extra={"config": cfg, "epoch": epoch, "metrics": metrics},
        )
        print(f"[checkpoint] saved to {epoch_ckpt}")

        # --- save 'last' (always overwrites) ---
        # last_ckpt = ckpt_dir / "openclip_abo_classifier_last.pt"
        last_ckpt = ckpt_dir / f"{Path(out_cfg['ckpt_name']).stem}_last.pt"

        save_classifier_checkpoint(
            path=str(last_ckpt),
            head=head,
            labels=labels,
            extra={"config": cfg, "epoch": epoch, "metrics": metrics},
        )
        print(f"[checkpoint] saved to {last_ckpt}")

        # --- save 'best' only if improved ---
        if metrics["top1_acc"] > best_top1:
            best_top1 = metrics["top1_acc"]
            best_metrics = metrics
            best_ckpt = ckpt_dir / f"{Path(out_cfg['ckpt_name']).stem}_best.pt"

            save_classifier_checkpoint(
                path=str(best_ckpt),
                head=head,
                labels=labels,
                extra={
                    "config": cfg,
                    "epoch": epoch,
                    "best_top1": best_top1,
                    "metrics": metrics,
                },
            )
            print(f"[checkpoint] BEST saved to {best_ckpt} (top1={best_top1:.4f})")

    print(f"Done. Best top1={best_top1:.4f}, best_metrics={best_metrics}")


if __name__ == "__main__":
    main()
