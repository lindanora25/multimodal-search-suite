"""
openclip_backbone.py

This module is the "shared backbone" of the project: a CLIP-style multimodal encoder.
It provides:

- load_openclip(...): load a pretrained OpenCLIP model + preprocessing transforms + tokenizer
- encode_text(...): convert raw text -> normalized embedding vector
- encode_image(...): convert preprocessed image tensor -> normalized embedding vector

Core idea:
Both text and images are mapped into the SAME embedding space, so similarity search is just
a dot product (cosine similarity when normalized).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import open_clip


@dataclass
class OpenCLIPBundle:
    """
    Convenience container so downstream code can use:
      bundle.model, bundle.preprocess, bundle.tokenizer
    """

    model: torch.nn.Module
    preprocess: object  # torchvision transform pipeline returned by open_clip
    tokenizer: object  # callable tokenizer returned by open_clip


def load_openclip(
    arch: str,
    pretrained: str,
    device: str = "cuda",
    ckpt_path: Optional[str] = None,
) -> OpenCLIPBundle:
    """
    Loads an OpenCLIP model and optionally loads a fine-tuned checkpoint.

    Parameters
    ----------
    arch:
        OpenCLIP architecture string, e.g. "ViT-B-32"
    pretrained:
        Pretrained weights name, e.g. "laion2b_s34b_b79k"
    device:
        "cuda" or "cpu"
    ckpt_path:
        Optional path to a .pt file saved by our training scripts.

    Returns
    -------
    OpenCLIPBundle
        model: OpenCLIP model
        preprocess: image preprocessing transform (PIL -> tensor normalized)
        tokenizer: text tokenizer compatible with the model
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=arch,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(arch)

    # Optionally load our fine-tuned checkpoint (same architecture).
    if ckpt_path is not None and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)

        # We support two formats:
        # 1) {"model_state": ..., ...}
        # 2) directly a state_dict
        state = ckpt.get("model_state", ckpt)

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            # Missing keys can happen if you saved only partial states in future.
            # For our full-model checkpoints, missing should typically be empty.
            print(f"[load_openclip] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[load_openclip] Unexpected keys: {len(unexpected)}")

        print(f"[load_openclip] Loaded checkpoint: {ckpt_path}")

    model.eval()  # default to eval; training scripts will switch to train()
    return OpenCLIPBundle(model=model, preprocess=preprocess, tokenizer=tokenizer)


@torch.no_grad()
def encode_text(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Encode a single text string into a normalized embedding.

    Returns tensor shape: (1, D) where D is embedding dimension (e.g., 512)
    """
    model.eval()
    tokens = tokenizer([text]).to(device)  # (1, seq_len) token ids
    txt = model.encode_text(tokens)  # (1, D) raw embedding
    txt = txt / txt.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return txt


@torch.no_grad()
def encode_image(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Encode a preprocessed image tensor into a normalized embedding.

    image_tensor:
        shape (1, 3, H, W) already preprocessed (normalized) for OpenCLIP

    Returns:
        (1, D) normalized embedding
    """
    model.eval()
    img = image_tensor.to(device)
    emb = model.encode_image(img)  # (1, D)
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return emb


def set_trainable_flags(
    model: torch.nn.Module,
    freeze_vision: bool,
    freeze_text: bool,
    train_projections: bool,
    train_logit_scale: bool,
) -> None:
    """
    Controls which parts of the CLIP model are trainable.

    Why this matters:
    - Full fine-tuning is expensive.
    - A strong baseline is to freeze the big encoders and only train:
        * projection layers (map encoder outputs -> shared embedding)
        * logit_scale (temperature) parameter
    That usually converges fast and fits Colab well.
    """
    # Default: freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze specific modules
    # OpenCLIP has "visual" for the vision encoder and "transformer" for text encoder
    if not freeze_vision:
        for p in model.visual.parameters():
            p.requires_grad = True

    if not freeze_text:
        for p in model.transformer.parameters():
            p.requires_grad = True

    # Projection layers:
    # - model.text_projection: (transformer width -> embed_dim)
    # - model.visual.proj OR model.visual_projection depending on arch
    if train_projections:
        if hasattr(model, "text_projection") and model.text_projection is not None:
            model.text_projection.requires_grad = True

        # Vision projection naming differs across OpenCLIP variants
        if (
            hasattr(model, "visual")
            and hasattr(model.visual, "proj")
            and model.visual.proj is not None
        ):
            model.visual.proj.requires_grad = True
        if hasattr(model, "visual_projection") and model.visual_projection is not None:
            model.visual_projection.requires_grad = True

    # logit_scale is a learned temperature parameter in CLIP
    # It scales similarity logits before softmax; tuning it helps training dynamics.
    if train_logit_scale and hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = True


def get_trainable_params(model: torch.nn.Module):
    """Return a list of parameters that require gradients (for optimizer)."""
    return [p for p in model.parameters() if p.requires_grad]
