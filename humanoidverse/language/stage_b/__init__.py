"""Stage B utilities for BFM + TextOp prompt-sequence generation."""

from .controller import BFMTextOpGenerator, load_bfm_textop_checkpoint
from .dataset import (
    BFMTextOpPrimitiveSequenceIterableConfig,
    BFMTextOpPrimitiveSequenceIterableDataset,
    BFMTextOpSequenceDataset,
    BFMTextOpSequenceDatasetConfig,
    collate_bfm_textop_sequences,
)
from .model import BFMTextOpLossConfig, BFMTextOpModel, BFMTextOpModelConfig
from .primitives import BFMTextOpPrimitiveExportConfig, export_bfm_textop_primitives

__all__ = [
    "BFMTextOpGenerator",
    "BFMTextOpLossConfig",
    "BFMTextOpModel",
    "BFMTextOpModelConfig",
    "BFMTextOpPrimitiveExportConfig",
    "BFMTextOpPrimitiveSequenceIterableConfig",
    "BFMTextOpPrimitiveSequenceIterableDataset",
    "BFMTextOpSequenceDataset",
    "BFMTextOpSequenceDatasetConfig",
    "collate_bfm_textop_sequences",
    "export_bfm_textop_primitives",
    "load_bfm_textop_checkpoint",
]
