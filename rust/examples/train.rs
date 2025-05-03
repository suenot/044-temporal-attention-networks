//! Example: Train a TABL model
//!
//! This example demonstrates how to:
//! - Load and prepare data
//! - Create a TABL model
//! - Make predictions
//! - Evaluate model performance

use tabl::{DataLoader, TABLConfig, TABLModel, BybitClient};
use tabl::model::AttentionType;
use ndarray::Array3;

/// Create synthetic training data for demonstration
fn create_synthetic_data(n_samples: usize, seq_len: usize, n_features: usize) -> (Array3<f64>, Vec<i32>) {
    let mut x = Array3::zeros((n_samples, seq_len, n_features));
    let mut y = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        // Generate synthetic patterns
        let trend = if i % 3 == 0 { 1.0 } else if i % 3 == 1 { -1.0 } else { 0.0 };

        for t in 0..seq_len {
            for f in 0..n_features {
                let noise = (rand::random::<f64>() - 0.5) * 0.1;
                let signal = (t as f64 / seq_len as f64) * trend * 0.5;
                x[[i, t, f]] = signal + noise;
            }
        }

        // Label based on trend
        y.push(if trend > 0.5 { 2 } else if trend < -0.5 { 0 } else { 1 });
    }

    (x, y)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("TABL - Training Example\n");
    println!("========================\n");

    // Configuration
    let seq_len = 50;
    let n_features = 6;
    let n_classes = 3;
    let hidden_t = 10;
    let hidden_d = 16;

    // Create model configuration
    let config = TABLConfig::new(seq_len, n_features, hidden_t, hidden_d, n_classes)
        .with_attention_type(AttentionType::Additive)
        .with_dropout(0.2);

    println!("Model Configuration:");
    println!("  Sequence Length: {}", config.seq_len);
    println!("  Input Features: {}", config.input_dim);
    println!("  Hidden T: {}", config.hidden_t);
    println!("  Hidden D: {}", config.hidden_d);
    println!("  Classes: {}", config.n_classes);
    println!();

    // Create model
    let model = TABLModel::new(config.clone());
    println!("Model created with {} parameters\n", model.num_params());

    // Create synthetic data
    println!("Creating synthetic training data...");
    let n_train = 500;
    let n_test = 100;

    let (x_train, y_train) = create_synthetic_data(n_train, seq_len, n_features);
    let (x_test, y_test) = create_synthetic_data(n_test, seq_len, n_features);

    println!("  Training samples: {}", n_train);
    println!("  Test samples: {}", n_test);
    println!();

    // Make predictions on training data
    println!("Making predictions...");
    let train_preds = model.predict_batch(&x_train);
    let test_preds = model.predict_batch(&x_test);

    // Calculate accuracy
    let train_acc = model.compute_accuracy(&train_preds, &y_train);
    let test_acc = model.compute_accuracy(&test_preds, &y_test);

    println!("\nResults (untrained model - random predictions):");
    println!("  Training Accuracy: {:.2}%", train_acc * 100.0);
    println!("  Test Accuracy: {:.2}%", test_acc * 100.0);

    // Class distribution in predictions
    let pred_distribution = |preds: &[usize]| {
        let down = preds.iter().filter(|&&p| p == 0).count();
        let hold = preds.iter().filter(|&&p| p == 1).count();
        let up = preds.iter().filter(|&&p| p == 2).count();
        (down, hold, up)
    };

    let (down, hold, up) = pred_distribution(&test_preds);
    println!("\nTest Predictions Distribution:");
    println!("  Down (0): {} ({:.1}%)", down, 100.0 * down as f64 / n_test as f64);
    println!("  Hold (1): {} ({:.1}%)", hold, 100.0 * hold as f64 / n_test as f64);
    println!("  Up (2): {} ({:.1}%)", up, 100.0 * up as f64 / n_test as f64);

    // Demonstrate attention weights
    println!("\nAttention Analysis:");
    let sample = x_test.slice(ndarray::s![0, .., ..]).to_owned();
    let attention_weights = model.get_attention_weights(&sample);

    println!("  Attention weights for first test sample:");
    let top_5: Vec<_> = {
        let mut indexed: Vec<_> = attention_weights.iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        indexed.into_iter().take(5).collect()
    };

    for (idx, weight) in top_5 {
        println!("    Time step {}: {:.4}", idx, weight);
    }

    // Try loading real data (if available)
    println!("\n\nAttempting to load real data from Bybit...");
    let client = BybitClient::new();

    match client.get_klines("BTCUSDT", "60", 200).await {
        Ok(klines) => {
            println!("Loaded {} candles from Bybit", klines.len());

            let loader = DataLoader::with_params(seq_len, 5, 0.001);
            match loader.prepare_tabl_data(&klines) {
                Ok((x, y)) => {
                    println!("Prepared {} samples for TABL", x.shape()[0]);

                    // Make predictions
                    let predictions = model.predict_batch(&x);
                    let accuracy = model.compute_accuracy(&predictions, &y);
                    println!("Accuracy on real data: {:.2}%", accuracy * 100.0);
                }
                Err(e) => println!("Error preparing data: {}", e),
            }
        }
        Err(e) => {
            println!("Could not load data: {} (this is expected without internet)", e);
        }
    }

    println!("\nDone!");
    Ok(())
}
