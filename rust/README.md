# TABL Rust Implementation

Rust implementation of the Temporal Attention-Augmented Bilinear Network (TABL) for financial time-series prediction using Bybit cryptocurrency data.

## Features

- High-performance TABL model implementation
- Bybit API client for real-time and historical data
- Feature engineering for OHLCV data
- Backtesting framework
- Multi-head attention support

## Project Structure

```text
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Main library exports
│   ├── api/                # Bybit API client
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP client
│   │   └── types.rs        # API types
│   ├── data/               # Data processing
│   │   ├── mod.rs
│   │   ├── loader.rs       # Data loading
│   │   ├── features.rs     # Feature engineering
│   │   └── dataset.rs      # Dataset utilities
│   ├── model/              # TABL architecture
│   │   ├── mod.rs
│   │   ├── bilinear.rs     # Bilinear layer
│   │   ├── attention.rs    # Temporal attention
│   │   └── tabl.rs         # Complete TABL model
│   └── strategy/           # Trading strategy
│       ├── mod.rs
│       ├── signals.rs      # Signal generation
│       └── backtest.rs     # Backtesting engine
└── examples/
    ├── fetch_data.rs       # Download Bybit data
    ├── train.rs            # Train model
    └── backtest.rs         # Run backtest
```

## Quick Start

```bash
# Build the project
cargo build --release

# Fetch data from Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT --interval 1h --days 30

# Train model
cargo run --example train -- --epochs 100 --batch-size 32

# Run backtest
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Usage

### As a Library

```rust
use tabl::{BybitClient, TABLConfig, TABLModel, DataLoader};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Fetch data
    let client = BybitClient::new();
    let klines = client.get_klines("BTCUSDT", "1h", 1000).await?;

    // Prepare features
    let loader = DataLoader::new();
    let (x, y) = loader.prepare_tabl_data(&klines, 100, 10)?;

    // Create model
    let config = TABLConfig {
        seq_len: 100,
        input_dim: 6,
        hidden_t: 20,
        hidden_d: 32,
        ..Default::default()
    };
    let model = TABLModel::new(config);

    // Make predictions
    let predictions = model.predict(&x)?;

    Ok(())
}
```

## Configuration

### TABLConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seq_len` | 100 | Input sequence length |
| `input_dim` | 6 | Number of input features |
| `hidden_t` | 20 | Compressed temporal dimension |
| `hidden_d` | 32 | Compressed feature dimension |
| `attention_dim` | 64 | Attention hidden dimension |
| `n_heads` | 4 | Number of attention heads |
| `n_classes` | 3 | Output classes (up/stationary/down) |
| `dropout` | 0.2 | Dropout rate |

## Performance

The Rust implementation offers:
- **Memory efficient**: Zero-copy operations where possible
- **Fast inference**: Optimized matrix operations via ndarray
- **Parallel processing**: Multi-threaded data preparation
- **Production ready**: Error handling and logging

## License

MIT
