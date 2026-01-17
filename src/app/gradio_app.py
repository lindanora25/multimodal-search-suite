# src/app/gradio_app.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import gradio as gr
import faiss

from src.models.openclip_backbone import load_openclip
from src.models.classifier_head import ClassifierHead


def _strip_prefix(sd: dict, prefix: str = "module.") -> dict:
    # If you ever trained with DataParallel, keys might be "module.fc.weight"
    if any(k.startswith(prefix) for k in sd.keys()):
        return {k[len(prefix) :]: v for k, v in sd.items()}
    return sd


def _load_classifier_head(clf_ckpt_path: str, device: str):
    from src.models.classifier_head import load_classifier_checkpoint

    ckpt = load_classifier_checkpoint(clf_ckpt_path, map_location="cpu")

    # Support both formats:
    # (A) our wrapped format: {"head_state_dict": ..., "labels": ...}
    # (B) raw state_dict: {"fc.weight": ..., "fc.bias": ...}
    state = (
        ckpt.get("head_state_dict")
        or ckpt.get("head_state")  # <-- your checkpoint uses this
        or ckpt.get("state_dict")
        or ckpt
    )

    state = _strip_prefix(state)

    if "fc.weight" not in state:
        raise KeyError(
            f"Classifier checkpoint doesn't contain 'fc.weight'. "
            f"Keys sample: {list(state.keys())[:10]}"
        )

    labels = ckpt.get("labels", [])
    embed_dim = int(state["fc.weight"].shape[1])
    num_classes = len(labels) if len(labels) > 0 else int(state["fc.weight"].shape[0])

    head = ClassifierHead(embed_dim=embed_dim, num_classes=num_classes, dropout=0.0).to(
        device
    )
    head.load_state_dict(state, strict=True)
    head.eval()
    return head, labels


def _load_faiss_and_meta(faiss_dir: str):
    faiss_dir = Path(faiss_dir)
    index_path = faiss_dir / "abo_images.index"
    meta_path = faiss_dir / "abo_meta.parquet"
    emb_path = faiss_dir / "abo_image_embs.npy"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta parquet: {meta_path}")

    meta = pd.read_parquet(str(meta_path)).reset_index(drop=True)

    # 1) Try normal FAISS read
    if index_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            return index, meta
        except Exception as e:
            print(f"[WARN] faiss.read_index failed: {e}")

    # 2) Try deserialize from bytes (more robust than read_index on some FS)
    if index_path.exists():
        try:
            b = index_path.read_bytes()
            try:
                index = faiss.deserialize_index(b)
            except TypeError:
                # some faiss builds expect uint8 array
                index = faiss.deserialize_index(np.frombuffer(b, dtype="uint8"))
            return index, meta
        except Exception as e:
            print(f"[WARN] faiss.deserialize_index failed: {e}")

    # 3) Fallback: load embeddings and build index in memory
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Cannot load FAISS index and embeddings not found. Expected {emb_path}. "
            f"Re-run build_faiss_index.py with --save_embeddings."
        )

    E = np.load(str(emb_path)).astype("float32")  # (N,D)
    # ensure L2-normalized (cosine == inner product)
    faiss.normalize_L2(E)
    index = faiss.IndexFlatIP(E.shape[1])
    index.add(E)
    print(f"[OK] Built FAISS index in-memory from embeddings: {E.shape}")
    return index, meta


def _normalize_np(x: np.ndarray) -> np.ndarray:
    # x: (B,D)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    return (x / n).astype("float32")


@torch.no_grad()
def _embed_text(model, tokenizer, text: str, device: str) -> np.ndarray:
    # Prefer open_clip.tokenize if tokenizer is SimpleTokenizer weirdness
    try:
        import open_clip

        tokens = open_clip.tokenize([text]).to(device)
    except Exception:
        tokens = tokenizer([text]).to(device)
    z = model.encode_text(tokens)
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return z.detach().cpu().numpy().astype("float32")


@torch.no_grad()
def _embed_image(model, preprocess, pil: Image.Image, device: str) -> np.ndarray:
    x = preprocess(pil.convert("RGB")).unsqueeze(0)  # (1,3,H,W) CPU
    x = x.to(device, non_blocking=True)
    z = model.encode_image(x)
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return z.detach().cpu().numpy().astype("float32")


def _search(index, q: np.ndarray, k: int):
    # FAISS expects float32 numpy on CPU
    q = _normalize_np(q)
    scores, ids = index.search(q, k)
    return scores[0], ids[0]


def _rows_to_gallery(meta: pd.DataFrame, scores: np.ndarray, ids: np.ndarray, k: int):
    items = []
    table_rows = []
    for rank in range(min(k, len(ids))):
        i = int(ids[rank])
        s = float(scores[rank])
        row = meta.iloc[i].to_dict()

        img_path = row.get("image_path") or row.get("img_path") or row.get("image")
        title = row.get("title", "")
        brand = row.get("brand", "")
        cat = row.get("category_l1", "")

        caption = f"#{rank + 1}  score={s:.3f} | {cat} | {brand} | {title}"
        items.append((img_path, caption))
        table_rows.append(
            {
                "rank": rank + 1,
                "score": round(s, 4),
                "category_l1": cat,
                "brand": brand,
                "title": title,
                "item_id": row.get("item_id", ""),
                "image_path": img_path,
            }
        )

    table_df = pd.DataFrame(table_rows)
    return items, table_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval_cfg", required=True, help="configs/abo_retrieval.yaml")
    ap.add_argument("--faiss_dir", required=True, help="artifacts/abo/faiss")
    ap.add_argument("--device", default=None, help="cuda or cpu (auto if empty)")
    ap.add_argument(
        "--clf_ckpt", default=None, help="Optional: classifier best checkpoint"
    )
    ap.add_argument(
        "--share", action="store_true", help="Gradio share link (useful in Colab)"
    )
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load retrieval model (must match the one used to build FAISS index)
    import yaml

    cfg = yaml.safe_load(Path(args.retrieval_cfg).read_text())
    arch = cfg["model"]["arch"]
    pretrained = cfg["model"]["pretrained"]
    ckpt_path = str(Path(cfg["outputs"]["ckpt_dir"]) / cfg["outputs"]["ckpt_name"])
    bundle = load_openclip(
        arch=arch, pretrained=pretrained, device=device, ckpt_path=ckpt_path
    )
    model = bundle.model
    preprocess = bundle.preprocess
    tokenizer = getattr(bundle, "tokenizer", None)
    model.eval()

    # Load FAISS + metadata
    index, meta = _load_faiss_and_meta(args.faiss_dir)

    # Optional classifier head
    # Optional classifier head

    clf = None
    clf_labels = None
    if args.clf_ckpt:
        clf, clf_labels = _load_classifier_head(args.clf_ckpt, device=device)
        print(f"[INFO] Loaded classifier head: classes={len(clf_labels)}")

    @torch.no_grad()
    def predict_category_from_pil(pil: Image.Image) -> str:
        if clf is None:
            return "Classifier not loaded."
        z = _embed_image(model, preprocess, pil, device=device)  # (1,D) np
        zt = torch.from_numpy(z).to(device)
        logits = clf(zt)
        pred = int(torch.argmax(logits, dim=-1).item())
        return f"{clf_labels[pred]}"

    def text_to_image_search(query: str, top_k: int):
        if not query or not query.strip():
            return [], pd.DataFrame()
        q = _embed_text(model, tokenizer, query.strip(), device=device)  # (1,D)
        scores, ids = _search(index, q, k=top_k)
        gallery, table = _rows_to_gallery(meta, scores, ids, k=top_k)
        return gallery, table

    def image_to_similar_search(img: Image.Image, top_k: int):
        if img is None:
            return "Upload an image.", [], pd.DataFrame()
        q = _embed_image(model, preprocess, img, device=device)  # (1,D)
        scores, ids = _search(index, q, k=top_k)
        gallery, table = _rows_to_gallery(meta, scores, ids, k=top_k)
        pred = (
            predict_category_from_pil(img)
            if clf is not None
            else "Classifier not loaded."
        )
        return pred, gallery, table

    with gr.Blocks(title="Multimodal Product Search Suite (OpenCLIP + FAISS)") as demo:
        gr.Markdown(
            "# Multimodal Product Search Suite\n"
            "OpenCLIP embeddings + FAISS index for fast retrieval.\n"
        )

        with gr.Tab("Text → Image Search"):
            q = gr.Textbox(
                label="Text query", placeholder="e.g., 'black leather office chair'"
            )
            k = gr.Slider(1, 25, value=10, step=1, label="Top-K")
            btn = gr.Button("Search")
            gallery = gr.Gallery(label="Results", columns=5, height=500)
            table = gr.Dataframe(label="Metadata", interactive=False)

            btn.click(text_to_image_search, inputs=[q, k], outputs=[gallery, table])

        with gr.Tab("Image → Similar Items"):
            img = gr.Image(type="pil", label="Upload a product image")
            k2 = gr.Slider(1, 25, value=10, step=1, label="Top-K")
            btn2 = gr.Button("Find similar")
            pred = gr.Textbox(
                label="Predicted category (optional classifier)", interactive=False
            )
            gallery2 = gr.Gallery(label="Results", columns=5, height=500)
            table2 = gr.Dataframe(label="Metadata", interactive=False)

            btn2.click(
                image_to_similar_search,
                inputs=[img, k2],
                outputs=[pred, gallery2, table2],
            )

        gr.Markdown(
            "### Notes\n"
            "- Cosine similarity is used via normalized embeddings + FAISS inner product.\n"
            "- Retrieval model: CLIP-style contrastive tuning.\n"
            "- Optional classifier: linear head on frozen embeddings.\n"
        )

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
