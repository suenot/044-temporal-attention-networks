//! Type definitions for Bybit API responses

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur when interacting with the Bybit API
#[derive(Error, Debug)]
pub enum BybitError {
    #[error("HTTP request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("JSON parsing failed: {0}")]
    ParseError(#[from] serde_json::Error),

    #[error("API error: {code} - {message}")]
    ApiError { code: i32, message: String },

    #[error("Invalid response format")]
    InvalidResponse,

    #[error("Rate limit exceeded")]
    RateLimitExceeded,
}

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    /// Start time of the candle (Unix timestamp in milliseconds)
    pub start_time: i64,
    /// Opening price
    pub open: f64,
    /// Highest price
    pub high: f64,
    /// Lowest price
    pub low: f64,
    /// Closing price
    pub close: f64,
    /// Trading volume
    pub volume: f64,
    /// Turnover (quote volume)
    pub turnover: f64,
}

impl Kline {
    /// Calculate typical price (HLC average)
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    /// Calculate price range
    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    /// Check if this is a bullish candle
    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    /// Calculate body size
    pub fn body_size(&self) -> f64 {
        (self.close - self.open).abs()
    }
}

/// Order book level with price and quantity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

/// Order book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// Timestamp
    pub timestamp: i64,
    /// Bid levels (sorted by price descending)
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (sorted by price ascending)
    pub asks: Vec<OrderBookLevel>,
}

impl OrderBook {
    /// Calculate mid price
    pub fn mid_price(&self) -> Option<f64> {
        if self.bids.is_empty() || self.asks.is_empty() {
            return None;
        }
        Some((self.bids[0].price + self.asks[0].price) / 2.0)
    }

    /// Calculate bid-ask spread
    pub fn spread(&self) -> Option<f64> {
        if self.bids.is_empty() || self.asks.is_empty() {
            return None;
        }
        Some(self.asks[0].price - self.bids[0].price)
    }

    /// Calculate order book imbalance
    pub fn imbalance(&self, depth: usize) -> f64 {
        let bid_volume: f64 = self.bids.iter().take(depth).map(|l| l.quantity).sum();
        let ask_volume: f64 = self.asks.iter().take(depth).map(|l| l.quantity).sum();
        let total = bid_volume + ask_volume;
        if total == 0.0 {
            0.0
        } else {
            (bid_volume - ask_volume) / total
        }
    }

    /// Calculate weighted average price for bids
    pub fn vwap_bid(&self, depth: usize) -> Option<f64> {
        let levels: Vec<_> = self.bids.iter().take(depth).collect();
        if levels.is_empty() {
            return None;
        }
        let total_volume: f64 = levels.iter().map(|l| l.quantity).sum();
        if total_volume == 0.0 {
            return None;
        }
        let weighted_sum: f64 = levels.iter().map(|l| l.price * l.quantity).sum();
        Some(weighted_sum / total_volume)
    }

    /// Calculate weighted average price for asks
    pub fn vwap_ask(&self, depth: usize) -> Option<f64> {
        let levels: Vec<_> = self.asks.iter().take(depth).collect();
        if levels.is_empty() {
            return None;
        }
        let total_volume: f64 = levels.iter().map(|l| l.quantity).sum();
        if total_volume == 0.0 {
            return None;
        }
        let weighted_sum: f64 = levels.iter().map(|l| l.price * l.quantity).sum();
        Some(weighted_sum / total_volume)
    }
}

/// Raw API response wrapper
#[derive(Debug, Deserialize)]
pub(crate) struct ApiResponse<T> {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: T,
}

/// Kline list result
#[derive(Debug, Deserialize)]
pub(crate) struct KlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// Order book result
#[derive(Debug, Deserialize)]
pub(crate) struct OrderBookResult {
    pub s: String,
    pub b: Vec<Vec<String>>,
    pub a: Vec<Vec<String>>,
    pub ts: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_calculations() {
        let kline = Kline {
            start_time: 1000000,
            open: 100.0,
            high: 110.0,
            low: 95.0,
            close: 105.0,
            volume: 1000.0,
            turnover: 100000.0,
        };

        assert!((kline.typical_price() - 103.333).abs() < 0.01);
        assert_eq!(kline.range(), 15.0);
        assert!(kline.is_bullish());
        assert_eq!(kline.body_size(), 5.0);
    }

    #[test]
    fn test_orderbook_calculations() {
        let orderbook = OrderBook {
            symbol: "BTCUSDT".to_string(),
            timestamp: 1000000,
            bids: vec![
                OrderBookLevel { price: 99.0, quantity: 10.0 },
                OrderBookLevel { price: 98.0, quantity: 20.0 },
            ],
            asks: vec![
                OrderBookLevel { price: 101.0, quantity: 15.0 },
                OrderBookLevel { price: 102.0, quantity: 25.0 },
            ],
        };

        assert_eq!(orderbook.mid_price(), Some(100.0));
        assert_eq!(orderbook.spread(), Some(2.0));

        // Imbalance: (30 - 40) / 70 = -0.1428...
        let imbalance = orderbook.imbalance(2);
        assert!((imbalance - (-0.1428)).abs() < 0.01);
    }
}
