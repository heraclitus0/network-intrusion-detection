"""
NIDS-CNN package with Cognize-embedded control-plane.

Exports:
    - load_cicids_dataset: data loader & preprocessor
    - build_cnn: CNN with embedded Cognize hooks (non-optional)
    - AdaptivePlateau: Cognize-driven training controller
"""

from .data_utils import load_cicids_dataset
from .model import build_cnn
from .callbacks import AdaptivePlateau

__all__ = ["load_cicids_dataset", "build_cnn", "AdaptivePlateau"]
