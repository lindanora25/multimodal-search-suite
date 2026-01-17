"""
train_retrieval.py

Trains the multimodal encoder for Text↔Image retrieval using CLIP-style contrastive learning.

Core idea (the "retrieval" objective):
- For a batch of N matched pairs (image_i, text_i),
  we want image_i to be most similar to text_i among all texts in the batch,
  and text_i to be most similar to image_i among all images in the batch.

Loss:
- symmetric cross entropy on similarity matrix:
  loss = (CE(image->text) + CE(text->image)) / 2

Metrics during eval:
- Recall@K, MRR computed on a validation set by similarity ranking.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.models.openclip_backbone import (
    load_openclip,
    set_trainable_flags,
    get_trainable_params,
)


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
    torch.backends.cudnn.deterministic = (
        False  # faster; set True for strict determinism
    )
    torch.backends.cudnn.benchmark = True


def cosine_warmup_schedule(warmup_steps: int, total_steps: int):
    """
    Returns a lambda(step) -> lr_multiplier
    Warm up linearly then cosine decay to 0.
    """

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # cosine after warmup
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


# -------------------------
# Dataset / Collate
# -------------------------


class ParquetPairDataset(Dataset):
    """
    Expects a parquet with at least:
      - image_path: path to image
      - text: paired text
    """

    def __init__(
        self,
        parquet_path: str,
        image_col: str,
        text_col: str,
        preprocess,
        root_dir: Optional[str] = None,
    ):
        self.df = pd.read_parquet(parquet_path)
        self.image_col = image_col
        self.text_col = text_col
        self.preprocess = preprocess
        self.root_dir = Path(root_dir) if root_dir else None

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, p: str) -> str:
        path = Path(p)
        if not path.is_absolute() and self.root_dir is not None:
            path = self.root_dir / path
        return str(path)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(row[self.image_col])
        text = str(row[self.text_col])

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.preprocess(img)  # (3, H, W)
        return img_tensor, text


def make_collate_fn(tokenizer):
    def collate(batch):
        imgs, texts = [], []

        for sample in batch:
            if sample is None:
                continue

            img, text = sample
            if img is None or text is None:
                continue
            imgs.append(img)  # keep cpu
            texts.append(text)

        if len(imgs) == 0:
            return None  # Let

        imgs = torch.stack(imgs, dim=0)  # CPU (B,3,H,W)
        try:
            # OpenCLIP simpleTokenizer style (no kwargs)
            tokens = tokenizer(texts)  # torch.Longtensor [B, 77]
        except TypeError:
            # HF tokenizer style
            tok = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            # HF returns dict; convert to tensor if needed
            tokens = (
                tok["input_ids"]
                if isinstance(tok, dict) and "input_ids" in tok
                else tok
            )

        return imgs, tokens

    return collate


# -------------------------
# Loss + Eval
# -------------------------


def clip_contrastive_loss(model, images, text_tokens) -> torch.Tensor:
    """
    Computes CLIP loss for a batch.

    Steps:
    1) encode image/text -> embeddings
    2) normalize
    3) similarity logits = logit_scale * img @ txt.T
    4) CE both directions
    """
    img = model.encode_image(images)
    txt = model.encode_text(text_tokens)

    img = img / img.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    txt = txt / txt.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    logit_scale = (
        model.logit_scale.exp()
        if hasattr(model, "logit_scale")
        else torch.tensor(1.0, device=images.device)
    )
    logits = logit_scale * img @ txt.t()  # (B, B)

    targets = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.t(), targets)

    return 0.5 * (loss_i2t + loss_t2i)


@torch.no_grad()
def compute_retrieval_metrics(
    model,
    tokenizer,
    preprocess,
    parquet_path: str,
    image_col: str,
    text_col: str,
    device: str,
    k_list: List[int],
    compute_mrr: bool = True,
    max_eval: Optional[int] = 1000,
) -> Dict[str, float]:
    """
    Computes retrieval metrics on paired eval set.

    For N pairs:
      - Compute image embeddings I (N, D)
      - Compute text embeddings  T (N, D)
      - Similarity S = I @ T^T
    For each query text_i, rank images by S[:, i] (or S[i, :] depending direction).

    This is the standard "paired retrieval" evaluation.
    """
    df = pd.read_parquet(parquet_path)
    if max_eval is not None:
        df = df.sample(n=min(max_eval, len(df)), random_state=42).reset_index(drop=True)

    # Build embeddings
    img_embs = []
    txt_embs = []

    model.eval()
    for i in tqdm(range(len(df)), desc="eval-embeddings", leave=False):
        img_path = df.iloc[i][image_col]
        text = str(df.iloc[i][text_col])

        img = Image.open(img_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        tokens = tokenizer([text]).to(device)

        img_e = model.encode_image(img_tensor)
        txt_e = model.encode_text(tokens)

        img_e = img_e / img_e.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        txt_e = txt_e / txt_e.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        img_embs.append(img_e.squeeze(0).detach().cpu())
        txt_embs.append(txt_e.squeeze(0).detach().cpu())

    I = torch.stack(img_embs, dim=0)  # (N, D)
    T = torch.stack(txt_embs, dim=0)  # (N, D)

    S = (I @ T.t()).numpy()  # (N, N) cosine similarities

    # Text -> Image retrieval
    metrics = {}
    ranks = []
    for i in range(S.shape[0]):
        # For query text_i, score all images: similarities are S[:, i] if using column,
        # but we built S = I @ T^T, so row i is image_i vs all texts.
        # For text->image, we want image ranking given text_i:
        scores = S[:, i]  # all images compared to text_i
        order = np.argsort(-scores)  # descending
        rank = int(np.where(order == i)[0][0]) + 1  # 1-indexed
        ranks.append(rank)

    for k in k_list:
        metrics[f"text2img_R@{k}"] = float(
            np.mean([1.0 if r <= k else 0.0 for r in ranks])
        )

    if compute_mrr:
        metrics["text2img_MRR"] = float(np.mean([1.0 / r for r in ranks]))

    # Image -> Text retrieval (symmetry)
    ranks2 = []
    for i in range(S.shape[0]):
        scores = S[i, :]  # image_i compared to all texts
        order = np.argsort(-scores)
        rank = int(np.where(order == i)[0][0]) + 1
        ranks2.append(rank)

    for k in k_list:
        metrics[f"img2text_R@{k}"] = float(
            np.mean([1.0 if r <= k else 0.0 for r in ranks2])
        )

    if compute_mrr:
        metrics["img2text_MRR"] = float(np.mean([1.0 / r for r in ranks2]))

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
    amp = bool(cfg["run"].get("amp", True)) and device.startswith("cuda")
    set_seed(int(cfg["run"].get("seed", 42)))

    # Load model
    bundle = load_openclip(
        arch=cfg["model"]["arch"],
        pretrained=cfg["model"]["pretrained"],
        device=device,
        ckpt_path=None,
    )
    model = bundle.model
    preprocess = bundle.preprocess
    tokenizer = bundle.tokenizer

    # Freeze/unfreeze strategy
    frz = cfg.get("freeze", {})
    set_trainable_flags(
        model,
        freeze_vision=bool(frz.get("freeze_vision", True)),
        freeze_text=bool(frz.get("freeze_text", True)),
        train_projections=bool(frz.get("train_projections", True)),
        train_logit_scale=bool(frz.get("train_logit_scale", True)),
    )

    trainable = get_trainable_params(model)
    if len(trainable) == 0:
        raise RuntimeError("No trainable parameters! Check freeze flags in config.")

    # Data
    data_cfg = cfg["data"]
    train_ds = ParquetPairDataset(
        parquet_path=data_cfg["train_path"],
        image_col=data_cfg["image_col"],
        text_col=data_cfg["text_col"],
        preprocess=preprocess,
        root_dir=data_cfg.get("root_dir"),
    )
    val_path = data_cfg["val_path"]

    train_cfg = cfg["train"]
    batch_size = int(train_cfg.get("batch_size", 128))
    grad_accum = int(train_cfg.get("grad_accum_steps", 1))
    epochs = int(train_cfg.get("epochs", 3))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=True,
        collate_fn=make_collate_fn(tokenizer),
        drop_last=True,
    )

    # Optimizer + scheduler
    lr = float(train_cfg.get("lr", 1e-5))
    wd = float(train_cfg.get("weight_decay", 0.01))
    optim = AdamW(trainable, lr=lr, weight_decay=wd)

    total_steps = epochs * (len(train_loader) // max(1, grad_accum))
    warmup_steps = int(train_cfg.get("warmup_steps", 500))

    sched = LambdaLR(optim, lr_lambda=cosine_warmup_schedule(warmup_steps, total_steps))

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # Outputs
    out_cfg = cfg["outputs"]
    ckpt_dir = Path(out_cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / out_cfg["ckpt_name"]

    # Eval settings
    k_list = cfg.get("metrics", {}).get("retrieval_k", [1, 5, 10])
    compute_mrr = bool(cfg.get("metrics", {}).get("compute_mrr", True))
    eval_every = int(train_cfg.get("eval_every", 500))
    log_every = int(train_cfg.get("log_every", 50))

    global_step = 0
    model.train()

    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_loader, desc=f"train epoch {epoch}/{epochs}")
        optim.zero_grad(set_to_none=True)

        running = 0.0
        for step, (images, tokens) in enumerate(pbar, start=1):
            if images is None or tokens is None:
                continue

            images = images.to(device, non_blocking=True)

            if isinstance(tokens, dict):
                tokens = {k: v.to(device, non_blocking=True) for k, v in tokens.items()}
            else:
                tokens = tokens.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp):
                loss = clip_contrastive_loss(model, images, tokens)
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            running += float(loss.item()) * grad_accum

            if step % grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1

                if global_step % log_every == 0:
                    pbar.set_postfix(
                        {"loss": running / log_every, "lr": sched.get_last_lr()[0]}
                    )
                    running = 0.0

                if global_step % eval_every == 0:
                    model.eval()
                    metrics = compute_retrieval_metrics(
                        model=model,
                        tokenizer=tokenizer,
                        preprocess=preprocess,
                        parquet_path=val_path,
                        image_col=data_cfg["image_col"],
                        text_col=data_cfg["text_col"],
                        device=device,
                        k_list=k_list,
                        compute_mrr=compute_mrr,
                        max_eval=1000,
                    )
                    print(f"\n[eval step={global_step}] {metrics}\n")
                    model.train()

        # Save after each epoch (simple + safe)
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "global_step": global_step,
            },
            str(ckpt_path),
        )
        print(f"[checkpoint] saved to {ckpt_path}")

    print("Done.")


if __name__ == "__main__":
    main()
