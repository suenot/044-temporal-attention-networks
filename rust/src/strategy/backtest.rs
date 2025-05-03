//! Backtesting framework

use crate::api::Kline;
use crate::strategy::signals::{Signal, SignalGenerator};
use ndarray::{Array2, Array3};

/// Results from a backtest
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Total return
    pub total_return: f64,
    /// Annualized return
    pub annualized_return: f64,
    /// Sharpe ratio (annualized)
    pub sharpe_ratio: f64,
    /// Maximum drawdown
    pub max_drawdown: f64,
    /// Win rate (percentage of profitable trades)
    pub win_rate: f64,
    /// Total number of trades
    pub n_trades: usize,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,
    /// List of signals generated
    pub signals: Vec<Signal>,
    /// Equity curve
    pub equity_curve: Vec<f64>,
    /// Daily returns
    pub returns: Vec<f64>,
}

impl BacktestResult {
    /// Print a summary of the backtest results
    pub fn summary(&self) -> String {
        format!(
            "Backtest Results:\n\
             ─────────────────────────\n\
             Total Return:     {:>10.2}%\n\
             Annual Return:    {:>10.2}%\n\
             Sharpe Ratio:     {:>10.2}\n\
             Max Drawdown:     {:>10.2}%\n\
             Win Rate:         {:>10.2}%\n\
             Trades:           {:>10}\n\
             Profit Factor:    {:>10.2}",
            self.total_return * 100.0,
            self.annualized_return * 100.0,
            self.sharpe_ratio,
            self.max_drawdown * 100.0,
            self.win_rate * 100.0,
            self.n_trades,
            self.profit_factor,
        )
    }
}

/// Backtest engine
pub struct BacktestEngine {
    /// Initial capital
    initial_capital: f64,
    /// Transaction cost (as fraction)
    transaction_cost: f64,
    /// Slippage (as fraction)
    slippage: f64,
    /// Risk-free rate for Sharpe ratio calculation
    risk_free_rate: f64,
    /// Trading periods per year (for annualization)
    periods_per_year: f64,
}

impl Default for BacktestEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl BacktestEngine {
    /// Create a new backtest engine with default parameters
    pub fn new() -> Self {
        Self {
            initial_capital: 10000.0,
            transaction_cost: 0.001, // 0.1%
            slippage: 0.0005,        // 0.05%
            risk_free_rate: 0.02,    // 2% annual
            periods_per_year: 365.0 * 24.0, // Hourly data
        }
    }

    /// Set initial capital
    pub fn with_capital(mut self, capital: f64) -> Self {
        self.initial_capital = capital;
        self
    }

    /// Set transaction cost
    pub fn with_transaction_cost(mut self, cost: f64) -> Self {
        self.transaction_cost = cost;
        self
    }

    /// Set slippage
    pub fn with_slippage(mut self, slippage: f64) -> Self {
        self.slippage = slippage;
        self
    }

    /// Set periods per year
    pub fn with_periods_per_year(mut self, periods: f64) -> Self {
        self.periods_per_year = periods;
        self
    }

    /// Run backtest with a signal generator and price data
    pub fn run<S: SignalGenerator>(
        &self,
        strategy: &S,
        data: &Array3<f64>,
        prices: &[f64],
    ) -> BacktestResult {
        let n_samples = data.shape()[0];
        assert_eq!(n_samples, prices.len(), "Data and prices must have same length");

        // Generate signals
        let signals = strategy.generate_signals(data);

        // Calculate returns and equity
        let mut equity = vec![self.initial_capital];
        let mut returns = Vec::with_capacity(n_samples);
        let mut position = 0.0;
        let mut trades = Vec::new();
        let mut current_trade_entry = 0.0;

        for i in 1..n_samples {
            let price_return = (prices[i] - prices[i - 1]) / prices[i - 1];
            let signal = signals[i - 1];

            // Calculate position change
            let new_position = signal.to_position();
            let position_change = (new_position - position).abs();

            // Transaction costs
            let cost = if position_change > 0.0 {
                (self.transaction_cost + self.slippage) * position_change
            } else {
                0.0
            };

            // Calculate return
            let strategy_return = position * price_return - cost;
            returns.push(strategy_return);

            // Update equity
            let new_equity = equity.last().unwrap() * (1.0 + strategy_return);
            equity.push(new_equity);

            // Track trades
            if position != 0.0 && new_position != position {
                // Closing a position (including position flips from long to short or vice versa)
                let trade_return = (prices[i] - current_trade_entry) / current_trade_entry * position;
                trades.push(trade_return);
            }
            if new_position != 0.0 && (position == 0.0 || new_position != position) {
                // Opening a position (from flat or after a flip)
                current_trade_entry = prices[i];
            }

            position = new_position;
        }

        // Close final position if any
        if position != 0.0 {
            let trade_return = (prices[n_samples - 1] - current_trade_entry)
                / current_trade_entry * position;
            trades.push(trade_return);
        }

        // Calculate metrics
        let total_return = (equity.last().unwrap() / self.initial_capital) - 1.0;
        let n_periods = returns.len() as f64;

        // Guard against empty returns to avoid division-by-zero
        let (annualized_return, sharpe_ratio) = if n_periods > 0.0 {
            let annualized_return =
                (1.0 + total_return).powf(self.periods_per_year / n_periods) - 1.0;

            // Sharpe ratio
            let mean_return = returns.iter().sum::<f64>() / n_periods;
            let variance = returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / n_periods;
            let std = variance.sqrt();
            let risk_free_per_period = self.risk_free_rate / self.periods_per_year;
            let sharpe_ratio = if std > 0.0 {
                (mean_return - risk_free_per_period) / std * self.periods_per_year.sqrt()
            } else {
                0.0
            };
            (annualized_return, sharpe_ratio)
        } else {
            (0.0, 0.0)
        };

        // Maximum drawdown
        let mut max_equity = self.initial_capital;
        let mut max_drawdown: f64 = 0.0;
        for e in &equity {
            max_equity = max_equity.max(*e);
            let drawdown = (max_equity - e) / max_equity;
            max_drawdown = max_drawdown.max(drawdown);
        }

        // Win rate and profit factor
        let winning_trades: Vec<_> = trades.iter().filter(|&&r| r > 0.0).collect();
        let losing_trades: Vec<_> = trades.iter().filter(|&&r| r < 0.0).collect();

        let win_rate = if !trades.is_empty() {
            winning_trades.len() as f64 / trades.len() as f64
        } else {
            0.0
        };

        let gross_profit: f64 = winning_trades.iter().map(|&&r| r).sum();
        let gross_loss: f64 = losing_trades.iter().map(|&&r| r.abs()).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        BacktestResult {
            total_return,
            annualized_return,
            sharpe_ratio,
            max_drawdown,
            win_rate,
            n_trades: trades.len(),
            profit_factor,
            signals,
            equity_curve: equity,
            returns,
        }
    }

    /// Run backtest with kline data
    pub fn run_with_klines<S: SignalGenerator>(
        &self,
        strategy: &S,
        data: &Array3<f64>,
        klines: &[Kline],
    ) -> BacktestResult {
        let prices: Vec<f64> = klines.iter().map(|k| k.close).collect();
        self.run(strategy, data, &prices)
    }
}

/// Calculate buy-and-hold benchmark
pub fn calculate_buy_and_hold(prices: &[f64]) -> f64 {
    if prices.is_empty() || prices[0] == 0.0 {
        return 0.0;
    }
    (prices.last().unwrap() - prices[0]) / prices[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{TABLConfig, TABLModel};
    use crate::strategy::signals::TABLStrategy;

    #[test]
    fn test_backtest_engine() {
        let config = TABLConfig::new(20, 4, 5, 8, 3);
        let model = TABLModel::new(config.clone());
        let strategy = TABLStrategy::new(model);

        // Create synthetic data
        let n_samples = 100;
        let data = Array3::from_shape_fn(
            (n_samples, config.seq_len, config.input_dim),
            |_| rand::random::<f64>(),
        );
        let prices: Vec<f64> = (0..n_samples)
            .map(|i| 100.0 + (i as f64 * 0.01).sin() * 10.0)
            .collect();

        let engine = BacktestEngine::new();
        let result = engine.run(&strategy, &data, &prices);

        assert_eq!(result.equity_curve.len(), n_samples);
        assert_eq!(result.returns.len(), n_samples - 1);
        assert!(result.max_drawdown >= 0.0 && result.max_drawdown <= 1.0);
        assert!(result.win_rate >= 0.0 && result.win_rate <= 1.0);
    }

    #[test]
    fn test_buy_and_hold() {
        let prices = vec![100.0, 105.0, 103.0, 110.0, 115.0];
        let bnh = calculate_buy_and_hold(&prices);
        assert!((bnh - 0.15).abs() < 1e-6);
    }

    #[test]
    fn test_backtest_result_summary() {
        let result = BacktestResult {
            total_return: 0.25,
            annualized_return: 0.50,
            sharpe_ratio: 1.5,
            max_drawdown: 0.10,
            win_rate: 0.55,
            n_trades: 50,
            profit_factor: 1.8,
            signals: vec![],
            equity_curve: vec![],
            returns: vec![],
        };

        let summary = result.summary();
        assert!(summary.contains("25.00%")); // total return
        assert!(summary.contains("50")); // trades
    }
}
