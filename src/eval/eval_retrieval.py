# src/eval/eval_retrieval.py
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Prefer your repo loader (handles preprocess/tokenizer and ckpt loading consistently)
from src.models.openclip_backbone import load_openclip


@dataclass
class RetrievalMetrics:
    direction: str  # "text_to_image" or "image_to_text"
    n: int
    recall_at: Dict[int, float]
    mrr: float


class ParquetPairs(Dataset):
    def __init__(self, df: pd.DataFrame, preprocess, image_col: str, text_col: str):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess
        self.image_col = image_col
        self.text_col = text_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row[self.image_col]
        text = row[self.text_col]

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_t = self.preprocess(img)  # CPU tensor
        return img_t, text


def collate_cpu(batch):
    # batch: List[(img_tensor, text)]
    imgs, texts = zip(*batch)
    imgs = torch.stack(list(imgs), dim=0)  # CPU
    texts = list(texts)
    return imgs, texts


@torch.no_grad()
def encode_texts(
    model, tokenizer, texts: List[str], device: str, batch_size: int
) -> torch.Tensor:
    embs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        # OpenCLIP tokenizer expects no HF kwargs
        tokens = tokenizer(chunk)
        if isinstance(tokens, dict):
            tokens = {k: v.to(device, non_blocking=True) for k, v in tokens.items()}
        else:
            tokens = tokens.to(device, non_blocking=True)

        z = model.encode_text(tokens)
        z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        embs.append(z.detach().cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def encode_images(model, loader: DataLoader, device: str) -> torch.Tensor:
    embs = []
    for imgs, _texts in loader:
        imgs = imgs.to(device, non_blocking=True)
        z = model.encode_image(imgs)
        z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        embs.append(z.detach().cpu())
    return torch.cat(embs, dim=0)


@torch.no_grad()
def compute_metrics_chunked(
    q: torch.Tensor, d: torch.Tensor, ks: List[int], device: str, query_chunk: int = 512
) -> Tuple[Dict[int, float], float]:
    """
    q: (N, D) query embeddings (normalized)
    d: (N, D) candidate embeddings (normalized)
    Computes:
      - Recall@K: correct index is in top-K
      - MRR: mean reciprocal rank

    Implementation uses chunked similarity to avoid NxN memory blow-ups.
    Rank is computed via: rank = 1 + count(sim > sim_true) (tie-safe-ish, exact if no ties).
    """
    N = q.shape[0]
    max_k = max(ks)

    q = q.to(device)
    d = d.to(device)

    hits = {k: 0 for k in ks}
    rr_sum = 0.0

    for start in range(0, N, query_chunk):
        end = min(start + query_chunk, N)
        q_chunk = q[start:end]  # (B, D)

        # (B, N)
        sims = q_chunk @ d.T

        # Top-K hits
        topk = torch.topk(sims, k=max_k, dim=1, largest=True).indices  # (B, max_k)
        # True indices are start..end-1
        true = torch.arange(start, end, device=device).unsqueeze(1)  # (B, 1)

        for k in ks:
            hit_k = (topk[:, :k] == true).any(dim=1).sum().item()
            hits[k] += int(hit_k)

        # MRR
        # true similarity per row (B,)
        row_idx = torch.arange(end - start, device=device)
        true_sims = sims[row_idx, torch.arange(start, end, device=device)]
        # rank = 1 + number of candidates with strictly greater similarity
        ranks = (sims > true_sims.unsqueeze(1)).sum(dim=1) + 1
        rr_sum += torch.sum(1.0 / ranks.float()).item()

    recall_at = {k: hits[k] / N for k in ks}
    mrr = rr_sum / N
    return recall_at, mrr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config (abo_retrieval.yaml).",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint path (.pt). If omitted, tries config outputs.",
    )
    ap.add_argument(
        "--parquet", type=str, required=True, help="Parquet path (val/test)."
    )
    ap.add_argument("--image_col", type=str, default="image_path")
    ap.add_argument("--text_col", type=str, default="text")
    ap.add_argument(
        "--batch_size", type=int, default=256, help="Eval batch size for encoding."
    )
    ap.add_argument(
        "--max_eval", type=int, default=2000, help="Limit examples for quick eval."
    )
    ap.add_argument("--k", type=str, default="1,5,10", help="Comma-separated K values.")
    ap.add_argument(
        "--device", type=str, default=None, help="cuda or cpu. Default: auto."
    )
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Optional path to save metrics JSON.",
    )
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load config if provided (for model arch/pretrained + default ckpt)
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

    # if ckpt_path is None:
    #     raise ValueError(
    #         "No checkpoint provided. Pass --ckpt or provide --config with outputs.ckpt_dir/name."
    #     )

    ###
    # Resolve optional checkpoint path:
    ckpt_path = args.ckpt  # may be None

    # If not explicitly provided, try config outputs ONLY if the file exists
    if ckpt_path is None and args.config:
        out = cfg.get("outputs", {})
        ckpt_dir = out.get("ckpt_dir")
        ckpt_name = out.get("ckpt_name")
        if ckpt_dir and ckpt_name:
            candidate = Path(ckpt_dir) / ckpt_name
            if candidate.exists():
                ckpt_path = str(candidate)

    if ckpt_path:
        raise ValueError(
            "No checkpoint provided. Pass --ckpt or provide --config with outputs.ckpt_dir/name."
        )
    else:
        print(f"[INFO] Using checkpoint: {ckpt_path}")
    ##
    # Load model + preprocess + tokenizer
    bundle = load_openclip(
        arch=arch, pretrained=pretrained, device=device, ckpt_path=ckpt_path
    )
    model = bundle.model
    preprocess = bundle.preprocess
    tokenizer = bundle.tokenizer
    model.eval()

    # Load parquet and sample
    df = pd.read_parquet(args.parquet)
    # max_eval: -1 means "use all", 0 means "use none" (rare), >0 means cap

    if args.max_eval is not None and args.max_eval > 0 and len(df) > args.max_eval:
        df = df.sample(n=args.max_eval, random_state=42).reset_index(drop=True)

    # Ensure images exist
    missing = (~df[args.image_col].map(lambda p: Path(p).exists())).sum()
    if missing > 0:
        print(f"[WARN] {missing} image paths missing. Dropping missing rows.")
        df = df[df[args.image_col].map(lambda p: Path(p).exists())].reset_index(
            drop=True
        )

    n = len(df)
    if n < 2:
        raise RuntimeError(
            "Not enough rows to evaluate after filtering missing images."
        )

    ks = [int(x.strip()) for x in args.k.split(",") if x.strip()]

    # Build loader for images (CPU workers, move to GPU in main)
    ds = ParquetPairs(df, preprocess, args.image_col, args.text_col)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_cpu,
        drop_last=False,
    )

    # Encode
    print(f"[INFO] Encoding {n} pairs on device={device} ...")
    texts = df[args.text_col].astype(str).tolist()

    text_emb = encode_texts(
        model, tokenizer, texts, device=device, batch_size=args.batch_size
    )  # (N, D) CPU
    img_emb = encode_images(model, loader, device=device)  # (N, D) CPU

    # Metrics: text->image and image->text
    print("[INFO] Computing metrics ...")
    t2i_recall, t2i_mrr = compute_metrics_chunked(text_emb, img_emb, ks, device=device)
    i2t_recall, i2t_mrr = compute_metrics_chunked(img_emb, text_emb, ks, device=device)

    m1 = RetrievalMetrics(
        direction="text_to_image", n=n, recall_at=t2i_recall, mrr=t2i_mrr
    )
    m2 = RetrievalMetrics(
        direction="image_to_text", n=n, recall_at=i2t_recall, mrr=i2t_mrr
    )

    def pretty(m: RetrievalMetrics):
        ks_sorted = sorted(m.recall_at.keys())
        parts = [f"Recall@{k}={m.recall_at[k]:.4f}" for k in ks_sorted]
        return f"{m.direction}: " + ", ".join(parts) + f", MRR={m.mrr:.4f}"

    print(pretty(m1))
    print(pretty(m2))

    if args.save_json:
        out = {
            "parquet": args.parquet,
            "ckpt": ckpt_path,
            "arch": arch,
            "pretrained": pretrained,
            "device": device,
            "metrics": [asdict(m1), asdict(m2)],
        }
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(json.dumps(out, indent=2))
        print(f"[OK] Saved metrics JSON -> {args.save_json}")


if __name__ == "__main__":
    main()
