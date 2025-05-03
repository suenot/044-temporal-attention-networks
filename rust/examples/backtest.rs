//! Example: Backtest a TABL trading strategy
//!
//! This example demonstrates:
//! - Creating a TABL-based trading strategy
//! - Running a backtest
//! - Analyzing results

use tabl::{DataLoader, TABLConfig, TABLModel, BybitClient, BacktestResult};
use tabl::strategy::{TABLStrategy, SignalGenerator};
use tabl::model::AttentionType;
use ndarray::Array3;

/// Create synthetic price data for demonstration
fn create_synthetic_prices(n: usize, initial_price: f64) -> Vec<f64> {
    let mut prices = Vec::with_capacity(n);
    let mut price = initial_price;

    for i in 0..n {
        // Add trend and noise
        let trend = (i as f64 * 0.001).sin() * 0.5;
        let noise = (rand::random::<f64>() - 0.5) * 0.02;
        price *= 1.0 + trend * 0.01 + noise;
        prices.push(price);
    }

    prices
}

/// Create synthetic feature data
fn create_synthetic_features(n: usize, seq_len: usize, n_features: usize) -> Array3<f64> {
    Array3::from_shape_fn((n, seq_len, n_features), |_| {
        (rand::random::<f64>() - 0.5) * 2.0
    })
}

/// Calculate buy-and-hold return
fn calculate_buy_and_hold(prices: &[f64]) -> f64 {
    if prices.is_empty() || prices[0] == 0.0 {
        return 0.0;
    }
    (prices.last().unwrap() - prices[0]) / prices[0]
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("TABL - Backtesting Example\n");
    println!("===========================\n");

    // Configuration
    let seq_len = 50;
    let n_features = 6;
    let n_classes = 3;
    let n_samples = 500;

    // Create model
    let config = TABLConfig::new(seq_len, n_features, 10, 16, n_classes)
        .with_attention_type(AttentionType::Additive)
        .with_dropout(0.1);

    let model = TABLModel::new(config);
    println!("Created TABL model with {} parameters\n", model.num_params());

    // Create strategy
    let strategy = TABLStrategy::new(model)
        .with_threshold(0.4)
        .with_min_confidence(0.35);

    // Create synthetic data
    println!("Creating synthetic market data...");
    let prices = create_synthetic_prices(n_samples, 50000.0);
    let features = create_synthetic_features(n_samples, seq_len, n_features);

    println!("  Samples: {}", n_samples);
    println!("  Initial Price: ${:.2}", prices[0]);
    println!("  Final Price: ${:.2}", prices.last().unwrap());
    println!();

    // Generate signals
    println!("Generating trading signals...\n");
    let signals = strategy.generate_signals(&features);

    // Signal distribution
    let buy_count = signals.iter()
        .filter(|s| matches!(s, tabl::strategy::Signal::Buy)).count();
    let sell_count = signals.iter()
        .filter(|s| matches!(s, tabl::strategy::Signal::Sell)).count();
    let hold_count = signals.iter()
        .filter(|s| matches!(s, tabl::strategy::Signal::Hold)).count();

    println!("Signal Distribution:");
    println!("  Buy signals:  {} ({:.1}%)",
        buy_count, 100.0 * buy_count as f64 / n_samples as f64);
    println!("  Sell signals: {} ({:.1}%)",
        sell_count, 100.0 * sell_count as f64 / n_samples as f64);
    println!("  Hold signals: {} ({:.1}%)",
        hold_count, 100.0 * hold_count as f64 / n_samples as f64);

    // Simple P&L calculation
    println!("\nCalculating P&L...");
    let mut equity = 10000.0;
    let mut position: f64 = 0.0;
    let transaction_cost = 0.001;

    for i in 1..n_samples {
        let price_return = (prices[i] - prices[i - 1]) / prices[i - 1];
        let signal = &signals[i - 1];

        let new_position = match signal {
            tabl::strategy::Signal::Buy => 1.0,
            tabl::strategy::Signal::Sell => -1.0,
            tabl::strategy::Signal::Hold => 0.0,
        };

        // Transaction cost on position change
        let position_change = (new_position - position).abs();
        if position_change > 0.0 {
            equity *= 1.0 - transaction_cost * position_change;
        }

        // P&L from position
        equity *= 1.0 + position * price_return;
        position = new_position;
    }

    let strategy_return = (equity / 10000.0) - 1.0;
    let bnh_return = calculate_buy_and_hold(&prices);

    println!("\nResults:");
    println!("────────────────────");
    println!("  Initial Capital:   ${:>10.2}", 10000.0);
    println!("  Final Equity:      ${:>10.2}", equity);
    println!("  Strategy Return:   {:>10.2}%", strategy_return * 100.0);
    println!("  Buy & Hold Return: {:>10.2}%", bnh_return * 100.0);
    println!("  Excess Return:     {:>10.2}%", (strategy_return - bnh_return) * 100.0);

    // Try with real data
    println!("\n\nAttempting backtest with real data from Bybit...");
    let client = BybitClient::new();

    match client.get_klines("BTCUSDT", "60", 500).await {
        Ok(klines) => {
            println!("Loaded {} candles from Bybit", klines.len());

            let loader = DataLoader::with_params(seq_len, 5, 0.001);
            match loader.prepare_tabl_data(&klines) {
                Ok((x, _)) => {
                    let real_prices: Vec<f64> = klines.iter()
                        .skip(seq_len + 5 - 1)
                        .take(x.shape()[0])
                        .map(|k| k.close)
                        .collect();

                    if real_prices.len() == x.shape()[0] {
                        // Generate signals for real data
                        let real_signals = strategy.generate_signals(&x);

                        // Calculate P&L
                        let mut real_equity = 10000.0;
                        let mut real_position: f64 = 0.0;

                        for i in 1..real_prices.len() {
                            let price_return = (real_prices[i] - real_prices[i - 1]) / real_prices[i - 1];
                            let signal = &real_signals[i - 1];

                            let new_position = match signal {
                                tabl::strategy::Signal::Buy => 1.0,
                                tabl::strategy::Signal::Sell => -1.0,
                                tabl::strategy::Signal::Hold => 0.0,
                            };

                            let position_change = (new_position - real_position).abs();
                            if position_change > 0.0 {
                                real_equity *= 1.0 - transaction_cost * position_change;
                            }

                            real_equity *= 1.0 + real_position * price_return;
                            real_position = new_position;
                        }

                        let real_strategy_return = (real_equity / 10000.0) - 1.0;
                        let real_bnh = calculate_buy_and_hold(&real_prices);

                        println!("\nReal Data Results:");
                        println!("────────────────────");
                        println!("  Strategy Return:   {:>10.2}%", real_strategy_return * 100.0);
                        println!("  Buy & Hold Return: {:>10.2}%", real_bnh * 100.0);
                    }
                }
                Err(e) => println!("Error preparing data: {}", e),
            }
        }
        Err(e) => {
            println!("Could not load data: {} (expected without internet)", e);
        }
    }

    println!("\nDone!");
    Ok(())
}
