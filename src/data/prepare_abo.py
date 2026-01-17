import argparse
import glob
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

LANG_DEFAULT = "en_US"


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_jsonl_any(path: str):
    # Supports .json and .json.gz
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def pick_lang(arr, lang=LANG_DEFAULT):
    """
    arr is often like: [{"language_tag":"en_US","value":"..."}]
    Sometimes standardized_values exists; we prefer value.
    """
    if not arr:
        return None
    # array of dicts
    if isinstance(arr, list):
        for d in arr:
            if isinstance(d, dict) and d.get("language_tag") == lang:
                return d.get("value") or (
                    d.get("standardized_values")[0]
                    if d.get("standardized_values")
                    else None
                )
        # fallback: first element with value
        for d in arr:
            if isinstance(d, dict):
                v = d.get("value")
                if v:
                    return v
    return None


def pick_simple(arr):
    """
    product_type often like [{"value":"SOFA"}]
    """
    if not arr:
        return None
    if isinstance(arr, list) and len(arr) > 0:
        d = arr[0]
        if isinstance(d, dict):
            return d.get("value") or d.get("path") or d.get("node_name")
    return None


def pick_node_path(nodes):
    """
    nodes often like [{"node_id":..., "path":"/Categories/..."}]
    Example paths shown on ABO site include category paths like /Categories/Furniture/... :contentReference[oaicite:3]{index=3}
    """
    if not nodes:
        return None
    if isinstance(nodes, list):
        for d in nodes:
            if isinstance(d, dict):
                p = d.get("path") or d.get("node_name")
                if p:
                    return p
    return None


def category_l1_from_path(node_path: str):
    if not node_path:
        return None
    parts = [p for p in node_path.split("/") if p]
    if not parts:
        return None
    # drop leading "Categories" if present
    if parts[0].lower() == "categories":
        parts = parts[1:]
    return parts[0] if parts else None


def resolve_image_file(images_dir: Path, rel_path: str):
    """
    images.csv has rel paths like '14/14fe8812.jpg' 
    In abo-images-small.tar, images commonly live under images/small/<rel_path>
    but I try a few candidates to be robust.
    """
    candidates = [
        images_dir / "small" / rel_path,
        images_dir / rel_path,
        images_dir / "images" / "small" / rel_path,
        images_dir / "images" / rel_path,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def build_text(fields: dict, order: list[str]) -> str:
    parts = []
    for k in order:
        v = fields.get(k)
        if v:
            parts.append(f"{k}: {v}")
    return " | ".join(parts)


#####
def infer_rel_stem_from_id(image_id: str) -> str:
    # shard folder is first 2 chars
    shard = image_id[:2]
    return f"{shard}/{image_id}"


def resolve_image_file_from_stem(images_dir: Path, rel_stem: str):
    """
    rel_stem like "fe/<image_id>" (no extension)
    tries common extensions
    """
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        cand = images_dir / "small" / f"{rel_stem}{ext}"
        if cand.exists():
            return str(cand)
    return None


#####
def resolve_image_by_id_fuzzy(images_dir: Path, image_id: str):
    """
    images_dir points to data/raw/images (which contains small/<shard>/files...)

    Since ABO listing main_image_id is not hex.
    Instead, search for files that start with the image_id across shard folders.

    This is O(#files) if done naively, so we keep it efficient by:
      - searching only within small/* (one level deep)
      - using glob patterns that avoid scanning everything too broadly
    """
    base = images_dir / "small"

    # Try common extensions by scanning shard folders with a wildcard:
    # small/*/<image_id>.jpg
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        matches = list(base.glob(f"*/{image_id}{ext}"))
        if matches:
            return str(matches[0])

    # Sometimes filenames have suffixes: <image_id>_1.jpg or <image_id>-something.jpg
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        matches = list(base.glob(f"*/{image_id}*{ext}"))
        if matches:
            return str(matches[0])

    return None


def build_image_index(images_dir: Path):
    """
    Build a dict: stem -> full_path
    where stem is filename without extension.
    This makes lookup O(1) per listing.
    """
    base = images_dir / "small"
    idx = {}
    for p in base.rglob("*"):
        if p.is_file():
            stem = p.stem  # filename without extension
            # keep first occurrence
            idx.setdefault(stem, str(p))
    return idx


##
def build_image_index(images_dir: Path):
    """
    Index all images under images_dir/small into a dict:
      filename_stem (no extension) -> full path
    """
    base = images_dir / "small"
    idx = {}
    for p in base.rglob("*"):
        if p.is_file():
            idx.setdefault(p.stem, str(p))
    return idx


##
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    ####
    print("CONFIG PATH:", args.config)
    print("CFG KEYS:", cfg.keys())
    print("CFG['data']:", cfg.get("data"))

    ####
    data_cfg = cfg["data"]
    listings_glob = data_cfg["listings_glob"]

    ####
    images_dir = Path(data_cfg["images_dir"])

    ###
    image_index = None

    # if img_map is None:
    #     print("[INFO] Building image index (one-time)...")
    #     image_index = build_image_index(images_dir)
    #     print(f"[INFO] Indexed {len(image_index)} image files.")

    ###

    processed_dir = Path(data_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # --- optional images.csv mapping (may not exist in your extraction) ---
    images_csv = data_cfg.get("images_csv")  # may be missing
    img_map = None
    if images_csv and Path(images_csv).exists():
        img_df = pd.read_csv(images_csv)
        # expects columns: image_id, ..., path
        img_map = dict(zip(img_df["image_id"].astype(str), img_df["path"].astype(str)))
        print(f"[OK] loaded images.csv mapping: {images_csv}")
    else:
        print(
            "[WARN] images.csv not found; will infer image path as small/<id[:2]>/<id>.(jpg|jpeg|png|webp)"
        )

    #############
    image_index = None
    if img_map is None:
        print("[INFO] Building image index (one-time)...")
        image_index = build_image_index(images_dir)
        print(f"[INFO] Indexed {len(image_index)} image files.")

    #############
    subset_cfg = cfg["subset"]
    max_items = int(subset_cfg.get("max_items", 20000))
    seed = int(subset_cfg.get("seed", 42))
    lang = subset_cfg.get("language", LANG_DEFAULT)
    require_existing = bool(subset_cfg.get("require_existing_image_file", True))

    text_fields = cfg["fields"]["text_fields"]
    label_name = cfg["labels"]["label_name"]

    files = sorted(glob.glob(listings_glob))
    if not files:
        raise FileNotFoundError(f"No listing files found with glob: {listings_glob}")

    rng = np.random.default_rng(seed)

    rows = []
    # Stream through listing files and collect candidates
    for fp in files:
        for rec in read_jsonl_any(fp):
            main_image_id = str(rec.get("main_image_id") or "")
            if not main_image_id:
                continue
            ###
            # Determine relative stem/path for image
            if img_map is not None:
                rel = img_map.get(main_image_id)
                if not rel:
                    continue
                # images.csv paths often include extension already; keep as-is
                image_path = resolve_image_file(images_dir, rel)
                if require_existing and not image_path:
                    continue
            else:
                rel_stem = infer_rel_stem_from_id(main_image_id)  # "fe/<image_id>"
                ###
                image_path = (
                    image_index.get(main_image_id) if image_index is not None else None
                )
                if require_existing and not image_path:
                    continue

                ###
            title = pick_lang(rec.get("item_name"), lang=lang) or pick_lang(
                rec.get("item_name"), lang=LANG_DEFAULT
            )
            if not title:
                continue

            brand = pick_lang(rec.get("brand"), lang=lang) or pick_lang(
                rec.get("brand"), lang=LANG_DEFAULT
            )
            product_type = pick_simple(rec.get("product_type"))
            node_path = pick_node_path(rec.get("node"))
            color = pick_lang(rec.get("color"), lang=lang) or pick_lang(
                rec.get("color"), lang=LANG_DEFAULT
            )
            material = pick_lang(rec.get("material"), lang=lang) or pick_lang(
                rec.get("material"), lang=LANG_DEFAULT
            )

            cat_l1 = category_l1_from_path(node_path)

            fields = {
                "title": title,
                "brand": brand,
                "product_type": product_type,
                "node_path": node_path,
                "color": color,
                "material": material,
            }
            text = build_text(fields, text_fields)

            rows.append(
                {
                    "item_id": rec.get("item_id"),
                    "main_image_id": main_image_id,
                    "image_path": image_path
                    if image_path
                    else (str(images_dir / "small" / rel)),
                    "title": title,
                    "brand": brand,
                    "product_type": product_type,
                    "node_path": node_path,
                    "category_l1": cat_l1,
                    "text": text,
                }
            )

    if not rows:
        raise RuntimeError(
            "No rows built. Check images.csv path + listings_glob + extracted folder structure."
        )

    df = pd.DataFrame(rows)

    # Sample down to ABO-mini
    if len(df) > max_items:
        df = df.sample(n=max_items, random_state=seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Drop missing labels if needed
    df = df.dropna(subset=["image_path", "text"]).reset_index(drop=True)
    if label_name in df.columns:
        df = df.dropna(subset=[label_name]).reset_index(drop=True)

    # Split
    tr = float(cfg["splits"]["train_ratio"])
    va = float(cfg["splits"]["val_ratio"])
    te = float(cfg["splits"]["test_ratio"])
    assert abs((tr + va + te) - 1.0) < 1e-6

    idx = np.arange(len(df))
    rng.shuffle(idx)

    n_train = int(tr * len(df))
    n_val = int(va * len(df))

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_df.to_parquet(cfg["data"]["output_train"], index=False)
    val_df.to_parquet(cfg["data"]["output_val"], index=False)
    test_df.to_parquet(cfg["data"]["output_test"], index=False)

    print("[OK] ABO-mini created:")
    print(f"  train: {len(train_df)} -> {cfg['data']['output_train']}")
    print(f"  val:   {len(val_df)} -> {cfg['data']['output_val']}")
    print(f"  test:  {len(test_df)} -> {cfg['data']['output_test']}")
    print(f"  columns: {list(train_df.columns)}")


if __name__ == "__main__":
    main()
