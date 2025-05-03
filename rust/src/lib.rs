//! # TABL
//!
//! Temporal Attention-Augmented Bilinear Network for financial time-series prediction
//! using cryptocurrency data from Bybit.
//!
//! ## Features
//!
//! - Temporal attention mechanism for focusing on important time steps
//! - Bilinear projection for efficient dimensionality reduction
//! - Multi-head attention support
//! - Bybit API integration
//!
//! ## Modules
//!
//! - `api` - Bybit API client
//! - `data` - Data loading and feature engineering
//! - `model` - TABL model architecture
//! - `strategy` - Trading strategy and backtesting
//!
//! ## Example
//!
//! ```no_run
//! use tabl::{BybitClient, TABLConfig, TABLModel, DataLoader};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Fetch data
//!     let client = BybitClient::new();
//!     let klines = client.get_klines("BTCUSDT", "60", 200).await?;
//!
//!     // Prepare data
//!     let loader = DataLoader::new();
//!     let (x, y) = loader.prepare_tabl_data(&klines)?;
//!
//!     // Create and use model
//!     let config = TABLConfig::default();
//!     let model = TABLModel::new(config);
//!     let predictions = model.predict_batch(&x);
//!
//!     Ok(())
//! }
//! ```

pub mod api;
pub mod data;
pub mod model;
pub mod strategy;

// Re-exports for convenience
pub use api::{BybitClient, BybitError, Kline, OrderBook};
pub use data::{DataLoader, Features, prepare_features};
pub use model::{TABLConfig, TABLModel, BilinearLayer, TemporalAttention, AttentionType};
pub use strategy::{BacktestEngine, BacktestResult, Signal, SignalGenerator, TABLStrategy, ThresholdStrategy, calculate_buy_and_hold};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration values
pub mod defaults {
    /// Input sequence length
    pub const SEQ_LEN: usize = 100;

    /// Number of input features
    pub const INPUT_DIM: usize = 6;

    /// Compressed temporal dimension
    pub const HIDDEN_T: usize = 20;

    /// Compressed feature dimension
    pub const HIDDEN_D: usize = 32;

    /// Attention hidden dimension
    pub const ATTENTION_DIM: usize = 64;

    /// Number of attention heads
    pub const N_HEADS: usize = 4;

    /// Number of output classes
    pub const N_CLASSES: usize = 3;

    /// Dropout rate
    pub const DROPOUT: f64 = 0.2;

    /// Learning rate
    pub const LEARNING_RATE: f64 = 0.001;

    /// Batch size
    pub const BATCH_SIZE: usize = 32;

    /// Number of epochs
    pub const EPOCHS: usize = 100;

    /// Prediction horizon
    pub const HORIZON: usize = 10;

    /// Classification threshold
    pub const THRESHOLD: f64 = 0.0002;
}
