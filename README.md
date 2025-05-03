# Chapter 46: Temporal Attention Networks for Financial Time-Series

This chapter explores **Temporal Attention Networks** — specialized attention mechanisms designed to capture temporal dependencies in financial data. We focus on the **TABL (Temporal Attention-Augmented Bilinear Layer)** architecture and its variants, which have proven highly effective for predicting market movements from Limit Order Book (LOB) data and other time-series sources.

<p align="center">
<img src="https://i.imgur.com/Zy8R4qF.png" width="70%">
</p>

## Contents

1. [Introduction to Temporal Attention](#introduction-to-temporal-attention)
    * [Why Temporal Attention?](#why-temporal-attention)
    * [Key Advantages](#key-advantages)
    * [Comparison with Other Models](#comparison-with-other-models)
2. [TABL Architecture](#tabl-architecture)
    * [Bilinear Projection](#bilinear-projection)
    * [Temporal Attention Mechanism](#temporal-attention-mechanism)
    * [BL (Bilinear Layer)](#bl-bilinear-layer)
    * [TABL (Temporal Attention Bilinear Layer)](#tabl-temporal-attention-bilinear-layer)
3. [Multi-Head Temporal Attention](#multi-head-temporal-attention)
    * [Multi-Head TABL](#multi-head-tabl)
    * [Parallel Attention Heads](#parallel-attention-heads)
4. [Data Processing](#data-processing)
    * [Limit Order Book Features](#limit-order-book-features)
    * [OHLCV Features](#ohlcv-features)
    * [Feature Engineering](#feature-engineering)
5. [Practical Examples](#practical-examples)
    * [01: Data Preparation](#01-data-preparation)
    * [02: TABL Architecture](#02-tabl-architecture)
    * [03: Model Training](#03-model-training)
    * [04: Attention Visualization](#04-attention-visualization)
    * [05: Trading Strategy](#05-trading-strategy)
6. [Rust Implementation](#rust-implementation)
7. [Python Implementation](#python-implementation)
8. [Best Practices](#best-practices)
9. [Resources](#resources)

## Introduction to Temporal Attention

Temporal Attention Networks are designed to solve a fundamental challenge in financial forecasting: **which past events matter most for predicting the future?**

Unlike standard recurrent models that process all time steps equally, temporal attention mechanisms learn to **focus on the most informative moments** in the input sequence.

### Why Temporal Attention?

Traditional models treat all time steps equally:

```
Time:    t-5  t-4  t-3  t-2  t-1  t
Weight:   1    1    1    1    1   1
         (All events equally important)
```

Temporal Attention learns adaptive weights:

```
Time:    t-5  t-4  t-3  t-2  t-1  t
Weight:  0.05 0.10 0.40 0.30 0.10 0.05
         (Attention focuses on t-3 and t-2)
```

**Key insight**: In financial markets, certain moments carry disproportionate importance — large trades, sudden volatility spikes, or specific patterns often precede price movements. Temporal attention automatically learns to identify these critical moments.

### Key Advantages

1. **Automatic Feature Selection in Time**
   - Learns which time steps are relevant for prediction
   - No manual feature engineering of "important moments"
   - Adapts to different market conditions

2. **Interpretability**
   - Attention weights reveal which events influenced predictions
   - Useful for understanding model decisions
   - Enables post-hoc analysis of trading signals

3. **Computational Efficiency**
   - TABL has O(T·D) complexity vs O(T²·D) for self-attention
   - Much faster than LSTM for long sequences
   - Lower memory requirements

4. **Strong Performance**
   - Outperforms LSTM and CNN on LOB prediction tasks
   - Achieves state-of-the-art on FI-2010 benchmark
   - Effective with just 1-2 layers

### Comparison with Other Models

| Feature | LSTM | CNN | Transformer | TABL |
|---------|------|-----|-------------|------|
| Temporal attention | ✗ | ✗ | ✓ (self) | ✓ (learned) |
| Complexity | O(T·D²) | O(T·K·D) | O(T²·D) | O(T·D) |
| Interpretability | Low | Low | Medium | High |
| LOB prediction | Good | Good | Good | Best |
| Memory efficient | ✗ | ✓ | ✗ | ✓ |
| Few layers needed | ✗ | ✗ | ✗ | ✓ |

## TABL Architecture

The TABL architecture combines **bilinear projections** with **temporal attention** to create an efficient and interpretable model.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    TEMPORAL ATTENTION NETWORK                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Input: X ∈ ℝ^(T×D)                                                  │
│  (T timesteps, D features)                                           │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │           Bilinear Projection (BL)                        │        │
│  │                                                           │        │
│  │   H = σ(W₁ · X · W₂ + b)                                 │        │
│  │                                                           │        │
│  │   W₁ ∈ ℝ^(T'×T)  - Temporal projection                   │        │
│  │   W₂ ∈ ℝ^(D×D')  - Feature projection                    │        │
│  │   H ∈ ℝ^(T'×D')  - Compressed representation             │        │
│  │                                                           │        │
│  └──────────────────────────────────────────────────────────┘        │
│                          │                                            │
│                          ▼                                            │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │           Temporal Attention (TA)                         │        │
│  │                                                           │        │
│  │   α = softmax(w · tanh(U · X^T))                         │        │
│  │   c = X^T · α                                             │        │
│  │                                                           │        │
│  │   α ∈ ℝ^T         - Attention weights                    │        │
│  │   c ∈ ℝ^D         - Context vector                       │        │
│  │                                                           │        │
│  └──────────────────────────────────────────────────────────┘        │
│                          │                                            │
│                          ▼                                            │
│  ┌──────────────────────────────────────────────────────────┐        │
│  │           Output Layer                                    │        │
│  │                                                           │        │
│  │   y = softmax(W_out · flatten(H, c) + b_out)             │        │
│  │                                                           │        │
│  │   3-class: Up / Stationary / Down                        │        │
│  │                                                           │        │
│  └──────────────────────────────────────────────────────────┘        │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Bilinear Projection

The bilinear layer performs two simultaneous linear transformations:

```python
class BilinearLayer(nn.Module):
    """
    Bilinear projection: H = σ(W₁ · X · W₂ + b)

    Transforms (T, D) → (T', D') by projecting both
    temporal and feature dimensions simultaneously.
    """

    def __init__(self, T_in, T_out, D_in, D_out, dropout=0.1):
        super().__init__()
        # Temporal projection: (T_out, T_in)
        self.W1 = nn.Parameter(torch.randn(T_out, T_in) * 0.01)
        # Feature projection: (D_in, D_out)
        self.W2 = nn.Parameter(torch.randn(D_in, D_out) * 0.01)
        # Bias: (T_out, D_out)
        self.bias = nn.Parameter(torch.zeros(T_out, D_out))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: (batch, T_in, D_in)
        # W1·X: (batch, T_out, D_in)
        out = torch.matmul(self.W1, x)
        # W1·X·W2: (batch, T_out, D_out)
        out = torch.matmul(out, self.W2)
        out = out + self.bias
        out = self.activation(out)
        return self.dropout(out)
```

**Why Bilinear?**
- Captures interactions between time and features
- More expressive than simple linear layers
- Reduces dimensionality in both axes efficiently

### Temporal Attention Mechanism

The temporal attention computes a weighted sum over time steps:

```python
class TemporalAttention(nn.Module):
    """
    Temporal attention: α = softmax(w · tanh(U · X^T))

    Learns to focus on important time steps.
    """

    def __init__(self, D, attention_dim=64):
        super().__init__()
        # Project features to attention space
        self.U = nn.Linear(D, attention_dim, bias=False)
        # Attention query vector
        self.w = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x):
        # x: (batch, T, D)
        # Compute attention scores
        h = torch.tanh(self.U(x))          # (batch, T, attention_dim)
        scores = self.w(h).squeeze(-1)      # (batch, T)

        # Softmax over time dimension
        alpha = F.softmax(scores, dim=-1)   # (batch, T)

        # Weighted sum: context vector
        context = torch.bmm(
            alpha.unsqueeze(1),             # (batch, 1, T)
            x                                # (batch, T, D)
        ).squeeze(1)                         # (batch, D)

        return context, alpha
```

**Interpretation:**
- `alpha[t]` indicates importance of time step t
- High attention on specific events reveals model's focus
- Context vector `c` summarizes the sequence

### BL (Bilinear Layer)

The BL layer is the attention-free version:

```python
class BL(nn.Module):
    """Bilinear Layer without attention"""

    def __init__(self, config):
        super().__init__()
        self.bilinear = BilinearLayer(
            config.seq_len, config.hidden_T,
            config.input_dim, config.hidden_D
        )

    def forward(self, x):
        h = self.bilinear(x)
        return h.flatten(1)  # (batch, hidden_T * hidden_D)
```

### TABL (Temporal Attention Bilinear Layer)

The full TABL combines both components:

```python
class TABL(nn.Module):
    """
    Temporal Attention-Augmented Bilinear Layer

    Combines bilinear projection with temporal attention.
    """

    def __init__(self, config):
        super().__init__()
        self.bilinear = BilinearLayer(
            config.seq_len, config.hidden_T,
            config.input_dim, config.hidden_D
        )
        self.attention = TemporalAttention(
            config.input_dim,
            config.attention_dim
        )

    def forward(self, x, return_attention=False):
        # Bilinear projection
        h = self.bilinear(x)  # (batch, hidden_T, hidden_D)
        h_flat = h.flatten(1)  # (batch, hidden_T * hidden_D)

        # Temporal attention
        context, alpha = self.attention(x)  # (batch, D), (batch, T)

        # Concatenate
        out = torch.cat([h_flat, context], dim=-1)

        if return_attention:
            return out, alpha
        return out
```

## Multi-Head Temporal Attention

Extending TABL with multiple attention heads allows the model to focus on different aspects simultaneously:

### Multi-Head TABL

```python
class MultiHeadTABL(nn.Module):
    """
    Multi-Head Temporal Attention Bilinear Layer

    Uses multiple attention heads to capture different
    temporal patterns in the data.
    """

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads

        # Shared bilinear projection
        self.bilinear = BilinearLayer(
            config.seq_len, config.hidden_T,
            config.input_dim, config.hidden_D
        )

        # Multiple attention heads
        self.attention_heads = nn.ModuleList([
            TemporalAttention(config.input_dim, config.attention_dim)
            for _ in range(config.n_heads)
        ])

        # Head combination
        self.head_combine = nn.Linear(
            config.n_heads * config.input_dim,
            config.input_dim
        )

    def forward(self, x, return_attention=False):
        # Bilinear projection
        h = self.bilinear(x)
        h_flat = h.flatten(1)

        # Multi-head attention
        contexts = []
        alphas = []
        for head in self.attention_heads:
            ctx, alpha = head(x)
            contexts.append(ctx)
            alphas.append(alpha)

        # Combine heads
        multi_context = torch.cat(contexts, dim=-1)
        combined = self.head_combine(multi_context)

        # Final output
        out = torch.cat([h_flat, combined], dim=-1)

        if return_attention:
            return out, torch.stack(alphas, dim=1)  # (batch, n_heads, T)
        return out
```

### Parallel Attention Heads

Each head can focus on different patterns:
- **Head 1**: Short-term price movements
- **Head 2**: Volume spikes
- **Head 3**: Order book imbalances
- **Head 4**: Trend patterns

```
Multi-Head Attention Visualization:

Time:     t-10  t-9   t-8   t-7   t-6   t-5   t-4   t-3   t-2   t-1
Head 1:   ░░░░  ░░░░  ░░░░  ░░░░  ░░░░  ██▓▓  ████  ██▓▓  ░░░░  ░░░░
          (Focuses on mid-term events)

Head 2:   ░░░░  ░░░░  ░░░░  ░░░░  ░░░░  ░░░░  ░░░░  ░░░░  ████  ████
          (Focuses on recent events)

Head 3:   ████  ░░░░  ░░░░  ████  ░░░░  ░░░░  ░░░░  ░░░░  ░░░░  ░░░░
          (Focuses on periodic patterns)

Combined: ▓▓▓▓  ░░░░  ░░░░  ▓▓▓▓  ░░░░  ▓▓▓▓  ████  ▓▓▓▓  ████  ████
          (Aggregated attention)
```

## Data Processing

### Limit Order Book Features

LOB data provides rich information about market microstructure:

```python
def extract_lob_features(lob_snapshot):
    """
    Extract features from Limit Order Book snapshot.

    Returns features for TABL input.
    """
    features = {}

    # Price levels (typically 10 levels each side)
    features['ask_prices'] = lob_snapshot['asks'][:, 0]  # (10,)
    features['bid_prices'] = lob_snapshot['bids'][:, 0]  # (10,)
    features['ask_volumes'] = lob_snapshot['asks'][:, 1]  # (10,)
    features['bid_volumes'] = lob_snapshot['bids'][:, 1]  # (10,)

    # Derived features
    mid_price = (features['ask_prices'][0] + features['bid_prices'][0]) / 2
    spread = features['ask_prices'][0] - features['bid_prices'][0]

    # Order imbalance
    total_ask_vol = features['ask_volumes'].sum()
    total_bid_vol = features['bid_volumes'].sum()
    imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)

    features['mid_price'] = mid_price
    features['spread'] = spread
    features['imbalance'] = imbalance

    return features
```

### OHLCV Features

For crypto/stock data using OHLCV:

```python
def extract_ohlcv_features(df):
    """
    Extract features from OHLCV data.
    """
    features = pd.DataFrame()

    # Price features
    features['log_return'] = np.log(df['close'] / df['close'].shift(1))
    features['high_low_range'] = (df['high'] - df['low']) / df['close']
    features['close_open_diff'] = (df['close'] - df['open']) / df['open']

    # Volume features
    features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    features['volume_std'] = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()

    # Technical indicators
    features['rsi'] = compute_rsi(df['close'], 14)
    features['macd'] = compute_macd(df['close'])

    # Volatility
    features['volatility'] = features['log_return'].rolling(20).std()

    return features.dropna()
```

### Feature Engineering

Recommended features for temporal attention models:

| Feature | Description | Importance |
|---------|-------------|------------|
| `log_return` | Log price change | High |
| `spread` | Bid-ask spread | High |
| `imbalance` | Order book imbalance | High |
| `volume_ratio` | Volume vs MA | Medium |
| `volatility` | Rolling volatility | Medium |
| `price_levels` | LOB price levels | Medium |
| `volume_levels` | LOB volume levels | Medium |

## Practical Examples

### 01: Data Preparation

```python
# python/01_data_preparation.py

import numpy as np
import pandas as pd
from typing import Tuple, List

def prepare_tabl_data(
    df: pd.DataFrame,
    lookback: int = 100,
    horizon: int = 10,
    threshold: float = 0.0002
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for TABL model training.

    Args:
        df: DataFrame with OHLCV data
        lookback: Number of time steps to look back
        horizon: Prediction horizon
        threshold: Threshold for direction classification

    Returns:
        X: Features (n_samples, lookback, n_features)
        y: Labels (n_samples,) - 0: down, 1: stationary, 2: up
    """
    # Extract features
    features = extract_features(df)

    # Normalize features
    features_norm = (features - features.mean()) / features.std()

    # Create sequences
    X, y = [], []

    for i in range(lookback, len(features_norm) - horizon):
        # Input sequence
        x_seq = features_norm.iloc[i-lookback:i].values

        # Target: future return
        future_return = np.log(
            df['close'].iloc[i + horizon] / df['close'].iloc[i]
        )

        # Classify direction
        if future_return > threshold:
            label = 2  # Up
        elif future_return < -threshold:
            label = 0  # Down
        else:
            label = 1  # Stationary

        X.append(x_seq)
        y.append(label)

    return np.array(X), np.array(y)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from OHLCV data."""
    features = pd.DataFrame(index=df.index)

    # Returns
    features['return'] = np.log(df['close'] / df['close'].shift(1))

    # Price position
    features['hl_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

    # Volume
    vol_ma = df['volume'].rolling(20).mean()
    features['vol_ratio'] = df['volume'] / vol_ma

    # Volatility
    features['volatility'] = features['return'].rolling(20).std()

    # Momentum
    features['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    features['momentum_10'] = df['close'] / df['close'].shift(10) - 1

    return features.dropna()
```

### 02: TABL Architecture

See [python/model.py](python/model.py) for complete implementation.

### 03: Model Training

```python
# python/03_train_model.py

import torch
from model import TABLModel, TABLConfig

# Configuration
config = TABLConfig(
    seq_len=100,
    input_dim=6,
    hidden_T=20,
    hidden_D=32,
    attention_dim=64,
    n_heads=4,
    n_classes=3,
    dropout=0.2
)

# Initialize model
model = TABLModel(config)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

# Training loop
best_acc = 0
for epoch in range(100):
    model.train()
    train_loss = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        # Forward pass
        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            logits = model(batch_x)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    val_acc = correct / total
    scheduler.step(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pt')

    print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val Acc={val_acc:.4f}")
```

### 04: Attention Visualization

```python
# python/04_attention_visualization.py

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(model, x, timestamps=None):
    """
    Visualize temporal attention weights.

    Args:
        model: Trained TABL model
        x: Input sequence (1, T, D)
        timestamps: Optional timestamps for x-axis
    """
    model.eval()
    with torch.no_grad():
        logits, attention = model(x, return_attention=True)

    # attention: (1, n_heads, T) or (1, T)
    attention = attention.squeeze(0).numpy()

    if attention.ndim == 1:
        # Single head
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.bar(range(len(attention)), attention)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Attention Weight')
        ax.set_title('Temporal Attention Weights')
    else:
        # Multi-head
        n_heads = attention.shape[0]
        fig, axes = plt.subplots(n_heads + 1, 1, figsize=(12, 3 * (n_heads + 1)))

        for i, ax in enumerate(axes[:-1]):
            ax.bar(range(attention.shape[1]), attention[i])
            ax.set_ylabel(f'Head {i+1}')
            ax.set_title(f'Attention Head {i+1}')

        # Combined attention
        combined = attention.mean(axis=0)
        axes[-1].bar(range(len(combined)), combined)
        axes[-1].set_xlabel('Time Step')
        axes[-1].set_ylabel('Combined')
        axes[-1].set_title('Combined Attention (Average)')

    plt.tight_layout()
    plt.savefig('attention_weights.png', dpi=150)
    plt.show()


def attention_heatmap(model, dataset, n_samples=50):
    """
    Create heatmap of attention patterns across samples.
    """
    model.eval()
    all_attention = []

    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            x, y = dataset[i]
            x = x.unsqueeze(0)
            _, attention = model(x, return_attention=True)
            all_attention.append(attention.squeeze().numpy())

    attention_matrix = np.stack(all_attention)  # (n_samples, T)

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        attention_matrix,
        cmap='Blues',
        xticklabels=5,
        yticklabels=5
    )
    plt.xlabel('Time Step')
    plt.ylabel('Sample')
    plt.title('Attention Patterns Across Samples')
    plt.savefig('attention_heatmap.png', dpi=150)
    plt.show()
```

### 05: Trading Strategy

```python
# python/05_strategy.py

def backtest_tabl_strategy(
    model,
    test_data,
    df_prices,
    initial_capital: float = 100000,
    transaction_cost: float = 0.001,
    confidence_threshold: float = 0.6
):
    """
    Backtest TABL trading strategy.

    Args:
        model: Trained TABL model
        test_data: Test dataset
        df_prices: Price data aligned with test_data
        initial_capital: Starting capital
        transaction_cost: Cost per transaction
        confidence_threshold: Min probability for trade
    """
    model.eval()
    capital = initial_capital
    position = 0  # -1: short, 0: flat, 1: long

    results = []

    with torch.no_grad():
        for i, (x, _) in enumerate(test_data):
            x = x.unsqueeze(0)
            logits = model(x)
            probs = F.softmax(logits, dim=1).squeeze()

            # Get prediction and confidence
            pred = probs.argmax().item()
            confidence = probs.max().item()

            # Trading logic
            if confidence >= confidence_threshold:
                if pred == 2 and position <= 0:  # Up signal
                    # Close short, open long
                    if position == -1:
                        capital *= (1 - transaction_cost)
                    position = 1
                    capital *= (1 - transaction_cost)

                elif pred == 0 and position >= 0:  # Down signal
                    # Close long, open short
                    if position == 1:
                        capital *= (1 - transaction_cost)
                    position = -1
                    capital *= (1 - transaction_cost)

            # Calculate P&L
            if i > 0:
                price_return = df_prices['close'].iloc[i] / df_prices['close'].iloc[i-1] - 1
                capital *= (1 + position * price_return)

            results.append({
                'capital': capital,
                'position': position,
                'prediction': pred,
                'confidence': confidence
            })

    return pd.DataFrame(results)


def calculate_metrics(results_df, df_prices):
    """Calculate strategy metrics."""
    returns = results_df['capital'].pct_change().dropna()

    metrics = {
        'total_return': (results_df['capital'].iloc[-1] / results_df['capital'].iloc[0] - 1) * 100,
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252 * 24),  # Hourly
        'max_drawdown': ((results_df['capital'].cummax() - results_df['capital']) /
                         results_df['capital'].cummax()).max() * 100,
        'win_rate': (returns > 0).mean() * 100,
        'n_trades': (results_df['position'].diff() != 0).sum()
    }

    return metrics
```

## Rust Implementation

See [rust/](rust/) for complete Rust implementation using Bybit data.

```
rust/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs              # Main library exports
│   ├── api/                # Bybit API client
│   │   ├── mod.rs
│   │   ├── client.rs       # HTTP client for Bybit
│   │   └── types.rs        # API response types
│   ├── data/               # Data processing
│   │   ├── mod.rs
│   │   ├── loader.rs       # Data loading utilities
│   │   ├── features.rs     # Feature engineering
│   │   └── dataset.rs      # Dataset for training
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

### Quick Start (Rust)

```bash
# Navigate to Rust project
cd rust

# Fetch data from Bybit
cargo run --example fetch_data -- --symbols BTCUSDT,ETHUSDT

# Train model
cargo run --example train -- --epochs 100 --batch-size 32

# Run backtest
cargo run --example backtest -- --start 2024-01-01 --end 2024-12-31
```

## Python Implementation

See [python/](python/) for Python implementation.

```
python/
├── model.py                # TABL model implementation
├── data.py                 # Data loading and features
├── train.py                # Training script
├── strategy.py             # Trading strategy
├── example_usage.py        # Example usage
├── requirements.txt        # Dependencies
└── __init__.py
```

### Quick Start (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train.py --symbols BTCUSDT --epochs 100

# Run backtest
python strategy.py --model checkpoints/best_model.pt
```

## Best Practices

### When to Use TABL

**Good use cases:**
- LOB mid-price movement prediction
- High-frequency direction forecasting
- Short-horizon predictions (seconds to minutes)
- Interpretability requirements

**Not ideal for:**
- Very long sequences (>500 time steps)
- Multi-horizon forecasting
- Portfolio allocation (use Stockformer)

### Hyperparameter Recommendations

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `seq_len` | 50-200 | Depends on data frequency |
| `hidden_T` | 10-30 | Temporal compression |
| `hidden_D` | 32-128 | Feature compression |
| `attention_dim` | 32-128 | Match with hidden_D |
| `n_heads` | 2-8 | More for complex patterns |
| `dropout` | 0.1-0.3 | Higher for small data |

### Common Pitfalls

1. **Class imbalance**: Use weighted loss or resample data
2. **Overfitting**: Apply dropout, early stopping
3. **Feature scaling**: Normalize inputs to zero mean, unit variance
4. **Threshold selection**: Tune classification thresholds carefully

## Resources

### Papers

- [Temporal Attention augmented Bilinear Network for Financial Time-Series Data Analysis](https://arxiv.org/abs/1712.00975) — Original TABL paper
- [Multi-head Temporal Attention-Augmented Bilinear Network](https://ieeexplore.ieee.org/document/9909957/) — Multi-head extension
- [Augmented Bilinear Network for Incremental Multi-Stock Time-Series Classification](https://www.sciencedirect.com/science/article/pii/S0031320323003059) — Incremental learning

### Implementations

- [TABL Original (Python 2.7)](https://github.com/viebboy/TABL)
- [TABL PyTorch](https://github.com/LeonardoBerti00/TABL-Temporal-Attention-Augmented-Bilinear-Network-for-Financial-Time-Series-Data-Analysis)

### Related Chapters

- [Chapter 26: Temporal Fusion Transformers](../26_temporal_fusion_transformers) — Multi-horizon forecasting
- [Chapter 43: Stockformer Multivariate](../43_stockformer_multivariate) — Cross-asset attention
- [Chapter 42: Dual Attention LOB](../42_dual_attention_lob) — LOB prediction

---

## Difficulty Level

**Intermediate to Advanced**

Prerequisites:
- Neural network fundamentals
- Attention mechanisms basics
- Time series forecasting concepts
- PyTorch/Rust ML libraries
