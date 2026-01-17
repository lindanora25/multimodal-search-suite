"""
classifier_head.py

This module defines a small classifier head that sits ON TOP of the frozen
image embeddings produced by OpenCLIP.

Concept:
- OpenCLIP gives you a strong "image representation" vector (embedding).
- For product category prediction, we map that embedding -> logits over categories.
- We keep the backbone frozen (fast, stable) and train only this head.

I also handle:
- saving/loading checkpoints with label mappings (idx -> label string)
- predicting probabilities for the Gradio demo
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierHead(nn.Module):
    """
    Simple classifier head: embedding -> logits over classes.

     can extend this later to a small MLP, but linear is a great baseline.
    """

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        emb: (B, D) normalized embeddings
        returns logits: (B, C)
        """
        x = self.dropout(emb)
        return self.fc(x)


@dataclass
class ClassifierBundle:
    model: ClassifierHead
    labels: List[str]  # index -> label name


def save_classifier_checkpoint(
    path: str,
    head: ClassifierHead,
    labels: List[str],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "head_state": head.state_dict(),
        "labels": labels,
        "extra": extra or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_classifier(
    ckpt_path: str,
    device: str = "cuda",
) -> ClassifierBundle:
    """
    Loads the classifier head + label mapping.

    Note:
    - We don't store embed_dim in config here; we infer it from the checkpoint weights.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    labels = ckpt["labels"]
    head_state = ckpt["head_state"]

    # Infer shapes:
    # fc.weight is (C, D)
    w = head_state["fc.weight"]
    num_classes, embed_dim = w.shape

    head = ClassifierHead(embed_dim=embed_dim, num_classes=num_classes, dropout=0.0)
    head.load_state_dict(head_state)
    head.to(device).eval()
    return ClassifierBundle(model=head, labels=labels)


def save_classifier_checkpoint(
    path: str,
    head: ClassifierHead,
    labels: list[str],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ckpt = {
        "head_state_dict": head.state_dict(),
        "labels": labels,
        "extra": extra or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_classifier_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Loads a classifier checkpoint saved by save_classifier_checkpoint().
    Expected format:
      {
        "head_state_dict": ...,
        "labels": [...],
        "extra": {...}
      }
    """
    ckpt = torch.load(path, map_location=map_location)

    # Some people might accidentally save just the state_dict.
    # Support both patterns.
    if isinstance(ckpt, dict) and ("head_state_dict" in ckpt or "labels" in ckpt):
        return ckpt

    # If it's a raw state_dict (fc.weight keys)
    if isinstance(ckpt, dict) and any(
        k.endswith("fc.weight") or k == "fc.weight" for k in ckpt.keys()
    ):
        return {"head_state_dict": ckpt, "labels": [], "extra": {}}

    raise ValueError(
        f"Unrecognized checkpoint format at {path}. Keys: {list(getattr(ckpt, 'keys', lambda: [])())[:20]}"
    )


@torch.no_grad()
def predict_probs(
    bundle: ClassifierBundle,
    emb: torch.Tensor,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Predict class probabilities for one embedding (1, D).
    Returns:
      probs: (C,) torch tensor on CPU
      labels: list[str]
    """
    bundle.model.eval()
    logits = bundle.model(emb)  # (1, C)
    probs = F.softmax(logits, dim=-1)[0].cpu()  # (C,)
    return probs, bundle.labels
