//! Trading strategy and backtesting module
//!
//! This module provides signal generation and backtesting capabilities.

mod backtest;
mod signals;

pub use backtest::{BacktestEngine, BacktestResult, calculate_buy_and_hold};
pub use signals::{Signal, SignalGenerator, TABLStrategy, ThresholdStrategy};
