"""
Data Loading and Feature Engineering for TABL

Provides:
- BybitDataLoader: Load cryptocurrency data from Bybit API
- prepare_tabl_data: Prepare data for TABL training
- extract_ohlcv_features: Extract features from OHLCV data
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import requests
from datetime import datetime, timedelta
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Kline:
    """Single candlestick data"""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float


class BybitDataLoader:
    """
    Load cryptocurrency data from Bybit API

    Example:
        loader = BybitDataLoader()
        df = loader.get_klines("BTCUSDT", "1h", 1000)
    """

    BASE_URL = "https://api.bybit.com"

    def __init__(self, testnet: bool = False):
        """
        Initialize Bybit data loader

        Args:
            testnet: Whether to use testnet API
        """
        if testnet:
            self.BASE_URL = "https://api-testnet.bybit.com"

    def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get kline/candlestick data

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            limit: Number of klines to fetch (max 1000)
            start_time: Start timestamp in milliseconds
            end_time: End timestamp in milliseconds

        Returns:
            DataFrame with OHLCV data
        """
        endpoint = f"{self.BASE_URL}/v5/market/kline"

        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": self._convert_interval(interval),
            "limit": min(limit, 1000)
        }

        if start_time:
            params["start"] = start_time
        if end_time:
            params["end"] = end_time

        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data["retCode"] != 0:
                raise ValueError(f"API error: {data['retMsg']}")

            klines = data["result"]["list"]

            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])

            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                df[col] = df[col].astype(float)

            # Sort by timestamp ascending
            df = df.sort_values("timestamp").reset_index(drop=True)

            return df

        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def get_historical_klines(
        self,
        symbol: str,
        interval: str = "1h",
        days: int = 30
    ) -> pd.DataFrame:
        """
        Get historical klines for a specified number of days

        Args:
            symbol: Trading pair
            interval: Kline interval
            days: Number of days of history

        Returns:
            DataFrame with OHLCV data
        """
        all_data = []
        end_time = int(datetime.now().timestamp() * 1000)
        interval_ms = self._interval_to_ms(interval)
        klines_per_request = 1000

        total_klines_needed = int((days * 24 * 60 * 60 * 1000) / interval_ms)
        requests_needed = (total_klines_needed // klines_per_request) + 1

        logger.info(f"Fetching {total_klines_needed} klines for {symbol}...")

        for i in range(requests_needed):
            df = self.get_klines(
                symbol=symbol,
                interval=interval,
                limit=1000,
                end_time=end_time
            )

            if df.empty:
                break

            all_data.append(df)
            end_time = int(df["timestamp"].iloc[0].timestamp() * 1000) - 1

            # Rate limiting
            time.sleep(0.1)

        if not all_data:
            return pd.DataFrame()

        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        return result.reset_index(drop=True)

    def _convert_interval(self, interval: str) -> str:
        """Convert interval string to Bybit format"""
        mapping = {
            "1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
            "1h": "60", "2h": "120", "4h": "240", "6h": "360", "12h": "720",
            "1d": "D", "1w": "W", "1M": "M"
        }
        return mapping.get(interval, interval)

    def _interval_to_ms(self, interval: str) -> int:
        """Convert interval string to milliseconds"""
        mapping = {
            "1m": 60000, "3m": 180000, "5m": 300000, "15m": 900000,
            "30m": 1800000, "1h": 3600000, "2h": 7200000, "4h": 14400000,
            "6h": 21600000, "12h": 43200000, "1d": 86400000, "1w": 604800000
        }
        return mapping.get(interval, 3600000)


def extract_ohlcv_features(
    df: pd.DataFrame,
    include_technical: bool = True
) -> pd.DataFrame:
    """
    Extract features from OHLCV data

    Args:
        df: DataFrame with OHLCV columns
        include_technical: Whether to include technical indicators

    Returns:
        DataFrame with extracted features
    """
    features = pd.DataFrame(index=df.index)

    # Price features
    features["log_return"] = np.log(df["close"] / df["close"].shift(1))
    features["high_low_range"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)
    features["close_open_diff"] = (df["close"] - df["open"]) / (df["open"] + 1e-8)
    features["hl_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-8)

    # Volume features
    vol_ma_20 = df["volume"].rolling(20).mean()
    features["volume_ma_ratio"] = df["volume"] / (vol_ma_20 + 1e-8)
    features["volume_change"] = df["volume"].pct_change()

    # Volatility
    features["volatility_20"] = features["log_return"].rolling(20).std()
    features["volatility_5"] = features["log_return"].rolling(5).std()

    if include_technical:
        # RSI
        features["rsi_14"] = compute_rsi(df["close"], 14)

        # MACD
        macd, signal = compute_macd(df["close"])
        features["macd"] = macd
        features["macd_signal"] = signal
        features["macd_hist"] = macd - signal

        # Bollinger Bands
        bb_upper, bb_lower = compute_bollinger_bands(df["close"])
        features["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-8)

        # Momentum
        features["momentum_5"] = df["close"] / df["close"].shift(5) - 1
        features["momentum_10"] = df["close"] / df["close"].shift(10) - 1
        features["momentum_20"] = df["close"] / df["close"].shift(20) - 1

    return features


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi / 100  # Normalize to [0, 1]


def compute_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series]:
    """Compute MACD indicator"""
    exp_fast = prices.ewm(span=fast, adjust=False).mean()
    exp_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = exp_fast - exp_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    # Normalize by price
    return macd / prices, signal_line / prices


def compute_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series]:
    """Compute Bollinger Bands"""
    ma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    return upper, lower


def prepare_tabl_data(
    df: pd.DataFrame,
    lookback: int = 100,
    horizon: int = 10,
    threshold: float = 0.0002,
    feature_columns: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for TABL model training

    Args:
        df: DataFrame with OHLCV data
        lookback: Number of time steps to look back
        horizon: Prediction horizon
        threshold: Threshold for direction classification
        feature_columns: List of feature columns to use

    Returns:
        X: Features (n_samples, lookback, n_features)
        y: Labels (n_samples,) - 0: down, 1: stationary, 2: up
        returns: Actual returns (n_samples,) - for backtesting
    """
    # Extract features
    features = extract_ohlcv_features(df)

    # Select feature columns
    if feature_columns is None:
        feature_columns = [
            "log_return", "high_low_range", "close_open_diff",
            "volume_ma_ratio", "volatility_20", "rsi_14"
        ]

    # Filter to available columns
    available_cols = [c for c in feature_columns if c in features.columns]
    features = features[available_cols]

    # Fill NaN values
    features = features.fillna(0)

    # Normalize features (z-score)
    features_mean = features.mean()
    features_std = features.std()
    features_norm = (features - features_mean) / (features_std + 1e-8)

    # Create sequences
    X, y, returns = [], [], []

    for i in range(lookback + 20, len(features_norm) - horizon):  # +20 for feature warmup
        # Input sequence
        x_seq = features_norm.iloc[i-lookback:i].values

        # Target: future log return
        future_return = np.log(
            df["close"].iloc[i + horizon] / df["close"].iloc[i]
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
        returns.append(future_return)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    returns = np.array(returns, dtype=np.float32)

    logger.info(f"Prepared {len(X)} samples with {X.shape[2]} features")
    logger.info(f"Class distribution: down={np.sum(y==0)}, stationary={np.sum(y==1)}, up={np.sum(y==2)}")

    return X, y, returns


def create_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    returns: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split data into train/val/test sets (time-based, no shuffle)

    Args:
        X: Features array
        y: Labels array
        returns: Returns array
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set

    Returns:
        Dictionary with 'train', 'val', 'test' splits
    """
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        "train": (X[:train_end], y[:train_end], returns[:train_end]),
        "val": (X[train_end:val_end], y[train_end:val_end], returns[train_end:val_end]),
        "test": (X[val_end:], y[val_end:], returns[val_end:])
    }


class TABLDataset:
    """
    PyTorch-compatible dataset for TABL

    Example:
        dataset = TABLDataset(X, y)
        loader = DataLoader(dataset, batch_size=32)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset

        Args:
            X: Features (n_samples, seq_len, n_features)
            y: Labels (n_samples,)
        """
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading...")

    # Create sample data (simulated since we can't call API in test)
    np.random.seed(42)
    n_samples = 1000

    # Simulate OHLCV data
    base_price = 100
    prices = base_price + np.cumsum(np.random.randn(n_samples) * 0.5)
    volumes = np.abs(np.random.randn(n_samples) * 1000 + 5000)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n_samples, freq="h"),
        "open": prices + np.random.randn(n_samples) * 0.1,
        "high": prices + np.abs(np.random.randn(n_samples)) * 0.5,
        "low": prices - np.abs(np.random.randn(n_samples)) * 0.5,
        "close": prices,
        "volume": volumes,
        "turnover": prices * volumes
    })

    # Test feature extraction
    features = extract_ohlcv_features(df)
    print(f"Extracted features: {list(features.columns)}")
    print(f"Feature shape: {features.shape}")

    # Test data preparation
    X, y, returns = prepare_tabl_data(df, lookback=100, horizon=10)
    print(f"\nData shapes: X={X.shape}, y={y.shape}, returns={returns.shape}")

    # Test split
    splits = create_train_val_test_split(X, y, returns)
    for name, (X_s, y_s, r_s) in splits.items():
        print(f"{name}: X={X_s.shape}, y={y_s.shape}")

    print("\nData loading tests passed!")
