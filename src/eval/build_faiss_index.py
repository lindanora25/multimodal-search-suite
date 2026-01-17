# src/eval/build_faiss_index.py
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import faiss  # pip install faiss-cpu

from src.models.openclip_backbone import load_openclip


class ImageOnlyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocess, image_col: str):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.image_col = image_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        img_path = self.df.loc[idx, self.image_col]
        img = Image.open(img_path).convert("RGB")
        x = self.preprocess(img)  # CPU tensor (3,H,W)
        return x


def collate_images(batch):
    # batch is list of CPU image tensors
    return torch.stack(batch, dim=0)  # CPU (B,3,H,W)


@torch.no_grad()
def embed_images(
    model,
    loader: DataLoader,
    device: str,
    normalize: bool = True,
) -> np.ndarray:
    embs: List[np.ndarray] = []

    for imgs in tqdm(loader, desc="Embedding images"):
        imgs = imgs.to(device, non_blocking=True)
        z = model.encode_image(imgs)  # (B,D)
        if normalize:
            z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        embs.append(z.detach().cpu().numpy().astype("float32"))

    return np.concatenate(embs, axis=0)  # (N,D) float32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config", type=str, default=None, help="configs/abo_retrieval.yaml (optional)"
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint .pt (optional if config has outputs)",
    )
    ap.add_argument(
        "--parquet",
        type=str,
        required=True,
        help="Parquet to index (usually test.parquet)",
    )
    ap.add_argument("--out_dir", type=str, default="artifacts/faiss")
    ap.add_argument("--image_col", type=str, default="image_path")
    ap.add_argument(
        "--max_items",
        type=int,
        default=20000,
        help="Cap items for demo. Use -1 for all.",
    )
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument(
        "--device", type=str, default=None, help="cuda or cpu (default auto)"
    )
    ap.add_argument(
        "--save_embeddings",
        action="store_true",
        help="Also save embeddings .npy (optional)",
    )
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Defaults if config not provided
    arch = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"
    ckpt_path = args.ckpt

    if args.config:
        import yaml

        cfg = yaml.safe_load(Path(args.config).read_text())
        arch = cfg.get("model", {}).get("arch", arch)
        pretrained = cfg.get("model", {}).get("pretrained", pretrained)

        if ckpt_path is None:
            out = cfg.get("outputs", {})
            ckpt_dir = out.get("ckpt_dir")
            ckpt_name = out.get("ckpt_name")
            if ckpt_dir and ckpt_name:
                candidate = Path(ckpt_dir) / ckpt_name
                if candidate.exists():
                    ckpt_path = str(candidate)

    if ckpt_path is None:
        raise ValueError(
            "No checkpoint provided. Pass --ckpt or --config with outputs.ckpt_dir/name."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_parquet(args.parquet).reset_index(drop=True)

    # Filter missing image paths (robustness)
    exists_mask = df[args.image_col].map(lambda p: Path(p).exists())
    if (~exists_mask).any():
        n_missing = int((~exists_mask).sum())
        print(f"[WARN] {n_missing} missing image files; dropping them.")
        df = df[exists_mask].reset_index(drop=True)

    if args.max_items != -1 and len(df) > args.max_items:
        df = df.sample(n=args.max_items, random_state=42).reset_index(drop=True)

    n = len(df)
    if n < 2:
        raise RuntimeError("Not enough rows to build an index.")

    print(f"[INFO] Building index for N={n} items on device={device}")
    print(f"[INFO] ckpt={ckpt_path}")

    # Load model + preprocess + tokenizer (tokenizer not needed for image index)
    bundle = load_openclip(
        arch=arch, pretrained=pretrained, device=device, ckpt_path=ckpt_path
    )
    model = bundle.model
    preprocess = bundle.preprocess
    model.eval()

    # Build loader
    ds = ImageOnlyDataset(df, preprocess, args.image_col)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_images,
        drop_last=False,
    )

    # Embed
    E = embed_images(model, loader, device=device, normalize=True)  # (N,D) float32
    dim = E.shape[1]
    print(f"[INFO] Embeddings shape: {E.shape}")

    # FAISS Index: cosine similarity = inner product on normalized vectors
    index = faiss.IndexFlatIP(dim)
    index.add(E)  # ids are 0..N-1

    # Save artifacts
    index_path = out_dir / "abo_images.index"
    meta_path = out_dir / "abo_meta.parquet"
    faiss.write_index(index, str(index_path))
    df.to_parquet(meta_path, index=False)

    print(f"[OK] Saved index -> {index_path}")
    print(f"[OK] Saved meta  -> {meta_path}")

    if args.save_embeddings:
        emb_path = out_dir / "abo_image_embs.npy"
        np.save(emb_path, E)
        print(f"[OK] Saved embs  -> {emb_path}")


if __name__ == "__main__":
    main()
