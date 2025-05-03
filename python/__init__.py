"""
TABL: Temporal Attention-Augmented Bilinear Network for Financial Time-Series

This module provides:
- TABLConfig: Model configuration
- TABLModel: Main TABL model
- MultiHeadTABL: Multi-head attention variant
- BilinearLayer: Bilinear projection layer
- TemporalAttention: Temporal attention mechanism
"""

from .model import (
    TABLConfig,
    TABLModel,
    MultiHeadTABL,
    BilinearLayer,
    TemporalAttention,
)
from .data import (
    BybitDataLoader,
    prepare_tabl_data,
    extract_ohlcv_features,
)
from .strategy import (
    TABLStrategy,
    backtest_tabl_strategy,
    calculate_metrics,
)

__all__ = [
    # Model
    "TABLConfig",
    "TABLModel",
    "MultiHeadTABL",
    "BilinearLayer",
    "TemporalAttention",
    # Data
    "BybitDataLoader",
    "prepare_tabl_data",
    "extract_ohlcv_features",
    # Strategy
    "TABLStrategy",
    "backtest_tabl_strategy",
    "calculate_metrics",
]

__version__ = "0.1.0"
