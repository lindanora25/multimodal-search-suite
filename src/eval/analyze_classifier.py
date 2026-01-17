# src/eval/analyze_classifier.py
import argparse
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.models.openclip_backbone import load_openclip
from src.models.classifier_head import ClassifierHead


def norm_label(x) -> str:
    return " ".join(str(x).split())


def load_classifier_ckpt(path: str, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    # supports your format: {"head_state":..., "labels":..., "extra":...}
    state = ckpt.get("head_state", ckpt.get("head_state_dict", ckpt))
    labels = ckpt.get("labels", [])
    return state, labels, ckpt.get("extra", {})


class ParquetImageLabelDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        image_col: str,
        label_col: str,
        preprocess,
        label_to_id: dict,
    ):
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.image_col = image_col
        self.label_col = label_col
        self.preprocess = preprocess
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row[self.image_col]
        lab = norm_label(row[self.label_col])
        if lab not in self.label_to_id:
            # skip unknown labels (shouldn’t happen if ckpt labels cover split)
            return None
        y = self.label_to_id[lab]
        x = self.preprocess(Image.open(img_path).convert("RGB"))  # CPU tensor
        return x, y


def collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long)


@torch.no_grad()
def predict_all(clip_model, head, loader, device: str):
    clip_model.eval()
    head.eval()

    y_true, y_pred = [], []

    for batch in loader:
        if batch is None:
            continue
        images, targets = batch
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # CLIP image embedding
        img_emb = clip_model.encode_image(images)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        logits = head(img_emb)
        preds = torch.argmax(logits, dim=-1)

        y_true.append(targets.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

    return np.concatenate(y_true), np.concatenate(y_pred)


def plot_label_distribution(
    train_parquet: str, label_col: str, out_png: Path, topn: int = 30
):
    df = pd.read_parquet(train_parquet)
    counts = df[label_col].astype(str).map(norm_label).value_counts()

    top = counts.head(topn)[::-1]
    plt.figure(figsize=(10, 10))
    plt.barh(top.index, top.values)
    plt.title(f"Top-{topn} label frequencies (train)")
    plt.xlabel("count")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_top_confusions(y_true, y_pred, labels: list, out_png: Path, topn: int = 25):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    np.fill_diagonal(cm, 0)  # only mistakes

    # get top off-diagonal pairs
    flat = cm.flatten()
    if flat.max() == 0:
        # no errors
        plt.figure(figsize=(10, 4))
        plt.title("Top confusions (none)")
        plt.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=200)
        plt.close()
        return

    idxs = np.argsort(flat)[::-1]
    pairs = []
    for idx in idxs:
        v = flat[idx]
        if v <= 0:
            break
        i = idx // cm.shape[1]
        j = idx % cm.shape[1]
        pairs.append((i, j, int(v)))
        if len(pairs) >= topn:
            break

    names = [f"{labels[i]} → {labels[j]}" for i, j, _ in pairs][::-1]
    vals = [v for _, _, v in pairs][::-1]

    plt.figure(figsize=(12, 10))
    plt.barh(names, vals)
    plt.title(f"Top-{len(vals)} confusion pairs (true → predicted)")
    plt.xlabel("count")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/abo_classifier.yaml")
    ap.add_argument(
        "--clf_ckpt", default=None, help="artifacts/.../openclip_abo_classifier_best.pt"
    )
    ap.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="which split to analyze for confusions/F1",
    )
    ap.add_argument("--device", default=None, help="cuda or cpu (auto if empty)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--topn_labels", type=int, default=30)
    ap.add_argument("--topn_confusions", type=int, default=25)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg["data"]
    train_path = data_cfg["train_path"]
    val_path = data_cfg["val_path"]
    test_path = data_cfg["test_path"]
    image_col = data_cfg.get("image_col", "image_path")
    label_col = data_cfg.get("label_col", "category_l1")

    # --- 1) label distribution plot (train) ---
    plot_label_distribution(
        train_parquet=train_path,
        label_col=label_col,
        out_png=Path("docs/label_distribution.png"),
        topn=args.topn_labels,
    )
    print("[OK] docs/label_distribution.png")

    # --- 2) load retrieval-tuned CLIP backbone (same as classifier training) ---
    model_cfg = cfg["model"]
    bundle = load_openclip(
        arch=model_cfg["arch"],
        pretrained=model_cfg["pretrained"],
        device=device,
        ckpt_path=model_cfg.get("base_ckpt"),
    )
    clip_model = bundle.model
    preprocess = bundle.preprocess
    clip_model.eval()

    # --- 3) load classifier head checkpoint ---
    clf_ckpt = args.clf_ckpt or str(
        Path(cfg["outputs"]["ckpt_dir"]) / "openclip_abo_classifier_best.pt"
    )
    head_state, labels, _extra = load_classifier_ckpt(clf_ckpt, map_location="cpu")

    if not labels:
        raise ValueError(
            "Classifier checkpoint has no 'labels' list. Re-save checkpoint with labels."
        )

    label_to_id = {lab: i for i, lab in enumerate(labels)}
    embed_dim = int(head_state["fc.weight"].shape[1])
    num_classes = int(head_state["fc.weight"].shape[0])

    head = ClassifierHead(embed_dim=embed_dim, num_classes=num_classes, dropout=0.0).to(
        device
    )
    head.load_state_dict(head_state, strict=True)
    head.eval()

    # --- 4) pick split and run predictions ---
    split_path = val_path if args.split == "val" else test_path
    ds = ParquetImageLabelDataset(
        split_path, image_col, label_col, preprocess, label_to_id
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate,
        drop_last=False,
    )

    y_true, y_pred = predict_all(clip_model, head, loader, device=device)
    print(f"[OK] predicted split={args.split} N={len(y_true)}")

    # --- 5) per-class F1 CSV ---
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(labels))), zero_division=0
    )
    out_df = pd.DataFrame(
        {
            "label": labels,
            "support": support.astype(int),
            "precision": prec.astype(float),
            "recall": rec.astype(float),
            "f1": f1.astype(float),
        }
    ).sort_values("f1")

    out_dir = Path("artifacts/abo/classifier")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "per_class_f1.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"[OK] {out_csv}")

    # --- 6) top confusions plot ---
    plot_top_confusions(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        out_png=Path("docs/top_confusions.png"),
        topn=args.topn_confusions,
    )
    print("[OK] docs/top_confusions.png")


if __name__ == "__main__":
    main()
