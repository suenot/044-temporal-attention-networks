"""
Example Usage of TABL for Financial Time-Series Prediction

This script demonstrates:
1. Data loading and preparation
2. Model training
3. Prediction and evaluation
4. Attention visualization
5. Backtesting
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import logging

from model import TABLModel, TABLConfig
from data import (
    BybitDataLoader,
    prepare_tabl_data,
    extract_ohlcv_features,
    create_train_val_test_split
)
from strategy import (
    TABLStrategy,
    backtest_tabl_strategy,
    print_backtest_results
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(n_samples: int = 2000) -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    np.random.seed(42)

    # Simulate price with trend and noise
    base_price = 100
    trend = np.linspace(0, 20, n_samples)
    noise = np.cumsum(np.random.randn(n_samples) * 0.5)
    prices = base_price + trend + noise

    # Add some patterns
    cycle = 5 * np.sin(np.linspace(0, 20 * np.pi, n_samples))
    prices += cycle

    # Volume correlated with volatility
    returns = np.diff(prices, prepend=prices[0]) / prices
    volatility = pd.Series(returns).rolling(10).std().fillna(0.01).values
    base_volume = 10000
    volumes = base_volume * (1 + 5 * volatility + np.random.randn(n_samples) * 0.2)
    volumes = np.abs(volumes)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="h"),
        "open": prices + np.random.randn(n_samples) * 0.1,
        "high": prices + np.abs(np.random.randn(n_samples)) * 0.5,
        "low": prices - np.abs(np.random.randn(n_samples)) * 0.5,
        "close": prices,
        "volume": volumes,
        "turnover": prices * volumes
    })

    # Ensure high >= close >= low
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


def train_model(
    model: TABLModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    patience: int = 10
) -> Dict:
    """
    Train TABL model

    Args:
        model: TABLModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        patience: Early stopping patience

    Returns:
        Dictionary with training history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output['logits'], batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = output['logits'].argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                output = model(batch_x)
                loss = criterion(output['logits'], batch_y)

                val_loss += loss.item()
                preds = output['logits'].argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        # Learning rate scheduling
        scheduler.step(val_acc)

        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        logger.info(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history


def visualize_attention(
    model: TABLModel,
    X: np.ndarray,
    sample_idx: int = 0
):
    """Visualize attention weights for a sample"""
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        x = torch.FloatTensor(X[sample_idx:sample_idx+1]).to(device)
        output = model(x, return_attention=True)

        if output['attention'] is None:
            logger.warning("Model doesn't return attention weights")
            return

        attention = output['attention'].cpu().numpy().squeeze()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Attention weights
    if attention.ndim == 1:
        axes[0].bar(range(len(attention)), attention)
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Attention Weight')
        axes[0].set_title('Temporal Attention Weights')
    else:
        # Multi-head: show heatmap
        import seaborn as sns
        sns.heatmap(attention, cmap='Blues', ax=axes[0])
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Attention Head')
        axes[0].set_title('Multi-Head Attention Weights')

    # Input features
    axes[1].plot(X[sample_idx])
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Feature Value')
    axes[1].set_title('Input Features')
    axes[1].legend([f'Feature {i}' for i in range(X.shape[2])], loc='best')

    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=150)
    plt.show()


def plot_training_history(history: Dict):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()

    # Accuracy
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()


def plot_backtest_results(result):
    """Plot backtest results"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Equity curve
    axes[0].plot(result.equity_curve)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Equity')
    axes[0].set_title('Equity Curve')
    axes[0].grid(True, alpha=0.3)

    # Position history
    axes[1].plot(result.positions, drawstyle='steps-post')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Position')
    axes[1].set_title('Position History')
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_yticklabels(['Short', 'Flat', 'Long'])
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150)
    plt.show()


def main():
    """Main function demonstrating TABL usage"""
    logger.info("=" * 60)
    logger.info("TABL: Temporal Attention-Augmented Bilinear Network")
    logger.info("=" * 60)

    # 1. Create or load data
    logger.info("\n1. Creating sample data...")
    df = create_sample_data(n_samples=3000)
    logger.info(f"Data shape: {df.shape}")

    # 2. Prepare features and labels
    logger.info("\n2. Preparing features and labels...")
    X, y, returns = prepare_tabl_data(
        df,
        lookback=100,
        horizon=10,
        threshold=0.001
    )
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

    # 3. Split data
    logger.info("\n3. Splitting data...")
    splits = create_train_val_test_split(X, y, returns)

    X_train, y_train, ret_train = splits['train']
    X_val, y_val, ret_val = splits['val']
    X_test, y_test, ret_test = splits['test']

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 4. Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # 5. Create model
    logger.info("\n4. Creating TABL model...")
    config = TABLConfig(
        seq_len=X.shape[1],
        input_dim=X.shape[2],
        hidden_T=20,
        hidden_D=32,
        attention_dim=64,
        n_classes=3,
        use_multihead=True,
        n_heads=4,
        dropout=0.2
    )

    model = TABLModel(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 6. Train model
    logger.info("\n5. Training model...")
    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=30,
        lr=0.001,
        patience=10
    )

    # 7. Evaluate on test set
    logger.info("\n6. Evaluating on test set...")
    model.eval()
    device = next(model.parameters()).device

    test_correct = 0
    test_total = 0
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_x)
            preds = output['logits'].argmax(dim=1)

            test_correct += (preds == batch_y).sum().item()
            test_total += batch_y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(output['probs'].cpu().numpy())

    test_acc = test_correct / test_total
    logger.info(f"Test Accuracy: {test_acc:.4f}")

    # 8. Backtest
    logger.info("\n7. Running backtest...")
    result = backtest_tabl_strategy(
        model,
        X_test,
        ret_test,
        confidence_threshold=0.5,
        transaction_cost=0.001
    )
    print_backtest_results(result)

    # 9. Visualizations (comment out if no display)
    try:
        logger.info("\n8. Creating visualizations...")
        plot_training_history(history)
        visualize_attention(model, X_test)
        plot_backtest_results(result)
    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("TABL demonstration complete!")
    logger.info("=" * 60)

    return model, result


if __name__ == "__main__":
    model, result = main()
