//! TABL model module
//!
//! This module provides the core TABL (Temporal Attention-Augmented Bilinear Network) implementation.

mod attention;
mod bilinear;
mod tabl;

pub use attention::{AttentionType, TemporalAttention};
pub use bilinear::BilinearLayer;
pub use tabl::{TABLConfig, TABLModel};
