"""
Trading Strategy and Backtesting for TABL

Provides:
- TABLStrategy: Trading strategy based on TABL model
- backtest_tabl_strategy: Backtest function
- calculate_metrics: Calculate performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Single trade result"""
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    direction: int  # 1: long, -1: short
    pnl: float
    return_pct: float


@dataclass
class BacktestResult:
    """Backtesting results"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    avg_trade_return: float
    profit_factor: float
    equity_curve: np.ndarray
    positions: np.ndarray
    returns: np.ndarray


class TABLStrategy:
    """
    Trading strategy based on TABL model predictions

    Example:
        strategy = TABLStrategy(
            model=trained_model,
            confidence_threshold=0.6,
            max_position=1.0
        )
        signals = strategy.generate_signals(X_test)
    """

    def __init__(
        self,
        model,
        confidence_threshold: float = 0.6,
        max_position: float = 1.0,
        transaction_cost: float = 0.001,
        use_softmax: bool = True
    ):
        """
        Initialize strategy

        Args:
            model: Trained TABL model
            confidence_threshold: Minimum probability for taking position
            max_position: Maximum position size (fraction of capital)
            transaction_cost: Transaction cost per trade
            use_softmax: Whether to use softmax probabilities
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.use_softmax = use_softmax

    def generate_signals(
        self,
        X: np.ndarray,
        return_confidence: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate trading signals from model predictions

        Args:
            X: Input features (n_samples, seq_len, n_features)
            return_confidence: Whether to return confidence scores

        Returns:
            signals: Trading signals (n_samples,) - -1: short, 0: flat, 1: long
            confidence: Confidence scores (n_samples,) - if return_confidence=True
        """
        self.model.eval()
        signals = []
        confidences = []

        with torch.no_grad():
            # Process in batches
            batch_size = 64
            for i in range(0, len(X), batch_size):
                batch = torch.FloatTensor(X[i:i+batch_size])

                output = self.model(batch)
                probs = output['probs'].numpy()

                for p in probs:
                    # Get prediction and confidence
                    pred = np.argmax(p)
                    conf = p[pred]

                    # Generate signal
                    if conf >= self.confidence_threshold:
                        if pred == 2:  # Up
                            signal = 1
                        elif pred == 0:  # Down
                            signal = -1
                        else:  # Stationary
                            signal = 0
                    else:
                        signal = 0  # Not confident enough

                    signals.append(signal)
                    confidences.append(conf)

        signals = np.array(signals)
        confidences = np.array(confidences)

        if return_confidence:
            return signals, confidences
        return signals, None

    def get_attention_insights(
        self,
        X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Get attention weights for interpretability

        Args:
            X: Input features (n_samples, seq_len, n_features)

        Returns:
            Dictionary with attention weights and analysis
        """
        self.model.eval()
        all_attention = []

        with torch.no_grad():
            batch_size = 64
            for i in range(0, len(X), batch_size):
                batch = torch.FloatTensor(X[i:i+batch_size])
                output = self.model(batch, return_attention=True)

                if output['attention'] is not None:
                    all_attention.append(output['attention'].numpy())

        if not all_attention:
            return {}

        attention = np.concatenate(all_attention, axis=0)

        # Compute statistics
        mean_attention = attention.mean(axis=0)
        std_attention = attention.std(axis=0)

        # Find most important time steps
        if attention.ndim == 2:  # Single head
            important_steps = np.argsort(mean_attention)[-10:][::-1]
        else:  # Multi-head
            combined_attention = attention.mean(axis=1)
            important_steps = np.argsort(combined_attention.mean(axis=0))[-10:][::-1]

        return {
            'attention': attention,
            'mean': mean_attention,
            'std': std_attention,
            'important_steps': important_steps
        }


def backtest_tabl_strategy(
    model,
    X: np.ndarray,
    actual_returns: np.ndarray,
    confidence_threshold: float = 0.6,
    transaction_cost: float = 0.001,
    initial_capital: float = 100000.0
) -> BacktestResult:
    """
    Backtest TABL trading strategy

    Args:
        model: Trained TABL model
        X: Input features (n_samples, seq_len, n_features)
        actual_returns: Actual returns for each period (n_samples,)
        confidence_threshold: Minimum probability for trading
        transaction_cost: Transaction cost per trade
        initial_capital: Starting capital

    Returns:
        BacktestResult with performance metrics
    """
    strategy = TABLStrategy(
        model=model,
        confidence_threshold=confidence_threshold,
        transaction_cost=transaction_cost
    )

    # Generate signals
    signals, confidences = strategy.generate_signals(X, return_confidence=True)

    # Initialize tracking
    capital = initial_capital
    equity_curve = [capital]
    positions = [0]
    realized_returns = []

    prev_position = 0

    for i in range(len(signals)):
        signal = signals[i]
        actual_ret = actual_returns[i]

        # Transaction cost for position changes
        if signal != prev_position:
            cost = transaction_cost * capital
            capital -= cost

        # Apply return based on position
        period_return = prev_position * actual_ret
        capital *= (1 + period_return)

        # Track
        equity_curve.append(capital)
        positions.append(signal)
        realized_returns.append(period_return)

        prev_position = signal

    # Calculate metrics
    equity_curve = np.array(equity_curve)
    positions = np.array(positions)
    realized_returns = np.array(realized_returns)

    metrics = calculate_metrics(equity_curve, realized_returns, positions)

    return BacktestResult(
        total_return=metrics['total_return'],
        sharpe_ratio=metrics['sharpe_ratio'],
        max_drawdown=metrics['max_drawdown'],
        win_rate=metrics['win_rate'],
        n_trades=metrics['n_trades'],
        avg_trade_return=metrics['avg_trade_return'],
        profit_factor=metrics['profit_factor'],
        equity_curve=equity_curve,
        positions=positions,
        returns=realized_returns
    )


def calculate_metrics(
    equity_curve: np.ndarray,
    returns: np.ndarray,
    positions: np.ndarray,
    annualization_factor: float = 252 * 24  # Hourly data
) -> Dict[str, float]:
    """
    Calculate performance metrics

    Args:
        equity_curve: Equity values over time
        returns: Period returns
        positions: Position history
        annualization_factor: Factor for annualizing returns

    Returns:
        Dictionary of performance metrics
    """
    # Total return
    total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100

    # Sharpe ratio
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(annualization_factor)
    else:
        sharpe_ratio = 0

    # Maximum drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_drawdown = np.max(drawdown) * 100

    # Win rate (only count periods with position)
    active_returns = returns[positions[:-1] != 0]
    if len(active_returns) > 0:
        win_rate = np.mean(active_returns > 0) * 100
    else:
        win_rate = 0

    # Number of trades (position changes)
    n_trades = np.sum(np.diff(positions) != 0)

    # Average trade return
    if len(active_returns) > 0:
        avg_trade_return = np.mean(active_returns) * 100
    else:
        avg_trade_return = 0

    # Profit factor
    gains = active_returns[active_returns > 0].sum()
    losses = abs(active_returns[active_returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else np.inf

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'n_trades': n_trades,
        'avg_trade_return': avg_trade_return,
        'profit_factor': profit_factor
    }


def print_backtest_results(result: BacktestResult):
    """Print backtest results in a formatted way"""
    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"Total Return:       {result.total_return:>10.2f}%")
    print(f"Sharpe Ratio:       {result.sharpe_ratio:>10.2f}")
    print(f"Max Drawdown:       {result.max_drawdown:>10.2f}%")
    print(f"Win Rate:           {result.win_rate:>10.2f}%")
    print(f"Number of Trades:   {result.n_trades:>10d}")
    print(f"Avg Trade Return:   {result.avg_trade_return:>10.4f}%")
    print(f"Profit Factor:      {result.profit_factor:>10.2f}")
    print("=" * 50)


def compare_with_benchmark(
    result: BacktestResult,
    benchmark_returns: np.ndarray,
    initial_capital: float = 100000.0
) -> Dict[str, float]:
    """
    Compare strategy with buy-and-hold benchmark

    Args:
        result: BacktestResult from strategy
        benchmark_returns: Buy-and-hold returns
        initial_capital: Starting capital

    Returns:
        Dictionary comparing strategy vs benchmark
    """
    # Calculate benchmark equity
    benchmark_equity = initial_capital * np.cumprod(1 + benchmark_returns)

    # Benchmark metrics
    benchmark_total = (benchmark_equity[-1] / initial_capital - 1) * 100
    benchmark_sharpe = np.mean(benchmark_returns) / np.std(benchmark_returns) * np.sqrt(252 * 24)

    peak = np.maximum.accumulate(benchmark_equity)
    benchmark_dd = np.max((peak - benchmark_equity) / peak) * 100

    return {
        'strategy_return': result.total_return,
        'benchmark_return': benchmark_total,
        'excess_return': result.total_return - benchmark_total,
        'strategy_sharpe': result.sharpe_ratio,
        'benchmark_sharpe': benchmark_sharpe,
        'strategy_drawdown': result.max_drawdown,
        'benchmark_drawdown': benchmark_dd
    }


if __name__ == "__main__":
    # Test strategy with mock model
    print("Testing strategy module...")

    # Create mock data
    np.random.seed(42)
    n_samples = 500
    seq_len = 100
    n_features = 6

    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    actual_returns = np.random.randn(n_samples) * 0.01  # 1% daily vol

    # Create mock model
    class MockModel:
        def __init__(self):
            pass

        def eval(self):
            pass

        def __call__(self, x, return_attention=False):
            batch_size = x.shape[0]
            # Random predictions
            probs = torch.softmax(torch.randn(batch_size, 3), dim=1)
            result = {'probs': probs, 'logits': torch.randn(batch_size, 3)}
            if return_attention:
                result['attention'] = torch.randn(batch_size, 100)
            return result

    mock_model = MockModel()

    # Test strategy
    strategy = TABLStrategy(mock_model, confidence_threshold=0.5)
    signals, confidences = strategy.generate_signals(X, return_confidence=True)
    print(f"Generated {len(signals)} signals")
    print(f"Signal distribution: long={np.sum(signals==1)}, flat={np.sum(signals==0)}, short={np.sum(signals==-1)}")

    # Test backtest
    result = backtest_tabl_strategy(
        mock_model, X, actual_returns,
        confidence_threshold=0.5,
        transaction_cost=0.001
    )

    print_backtest_results(result)

    # Test benchmark comparison
    comparison = compare_with_benchmark(result, actual_returns)
    print("\nBenchmark Comparison:")
    for k, v in comparison.items():
        print(f"  {k}: {v:.2f}")

    print("\nStrategy tests passed!")
