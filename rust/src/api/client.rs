//! Bybit API client implementation

use reqwest::Client;
use std::time::Duration;

use super::types::{ApiResponse, BybitError, Kline, KlineResult, OrderBook, OrderBookLevel, OrderBookResult};

/// Bybit API client
pub struct BybitClient {
    client: Client,
    base_url: String,
}

impl Default for BybitClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BybitClient {
    /// Create a new Bybit client
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: "https://api.bybit.com".to_string(),
        }
    }

    /// Create a client with a custom base URL (for testing)
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Failed to create HTTP client"),
            base_url: base_url.to_string(),
        }
    }

    /// Fetch OHLCV candlestick data
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `interval` - Kline interval (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, M, W)
    /// * `limit` - Number of candles to fetch (max 1000)
    ///
    /// # Example
    /// ```no_run
    /// use tabl::BybitClient;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let client = BybitClient::new();
    ///     let klines = client.get_klines("BTCUSDT", "1h", 100).await.unwrap();
    ///     println!("Fetched {} candles", klines.len());
    /// }
    /// ```
    pub async fn get_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: usize,
    ) -> Result<Vec<Kline>, BybitError> {
        let url = format!(
            "{}/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
            self.base_url,
            symbol,
            interval,
            limit.min(1000)
        );

        let response = self.client.get(&url).send().await?;
        let api_response: ApiResponse<KlineResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        let klines: Result<Vec<Kline>, BybitError> = api_response
            .result
            .list
            .iter()
            .map(|row| {
                if row.len() < 7 {
                    return Err(BybitError::InvalidResponse);
                }
                Ok(Kline {
                    start_time: row[0].parse().map_err(|_| BybitError::InvalidResponse)?,
                    open: row[1].parse().map_err(|_| BybitError::InvalidResponse)?,
                    high: row[2].parse().map_err(|_| BybitError::InvalidResponse)?,
                    low: row[3].parse().map_err(|_| BybitError::InvalidResponse)?,
                    close: row[4].parse().map_err(|_| BybitError::InvalidResponse)?,
                    volume: row[5].parse().map_err(|_| BybitError::InvalidResponse)?,
                    turnover: row[6].parse().map_err(|_| BybitError::InvalidResponse)?,
                })
            })
            .collect();

        // Bybit returns data in reverse chronological order, so we reverse it
        let mut klines = klines?;
        klines.reverse();
        Ok(klines)
    }

    /// Fetch order book snapshot
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    /// * `limit` - Order book depth (max 500)
    pub async fn get_orderbook(
        &self,
        symbol: &str,
        limit: usize,
    ) -> Result<OrderBook, BybitError> {
        let url = format!(
            "{}/v5/market/orderbook?category=spot&symbol={}&limit={}",
            self.base_url,
            symbol,
            limit.min(500)
        );

        let response = self.client.get(&url).send().await?;
        let api_response: ApiResponse<OrderBookResult> = response.json().await?;

        if api_response.ret_code != 0 {
            return Err(BybitError::ApiError {
                code: api_response.ret_code,
                message: api_response.ret_msg,
            });
        }

        let bids: Result<Vec<OrderBookLevel>, BybitError> = api_response
            .result
            .b
            .iter()
            .map(|level| {
                if level.len() < 2 {
                    return Err(BybitError::InvalidResponse);
                }
                Ok(OrderBookLevel {
                    price: level[0].parse().map_err(|_| BybitError::InvalidResponse)?,
                    quantity: level[1].parse().map_err(|_| BybitError::InvalidResponse)?,
                })
            })
            .collect();

        let asks: Result<Vec<OrderBookLevel>, BybitError> = api_response
            .result
            .a
            .iter()
            .map(|level| {
                if level.len() < 2 {
                    return Err(BybitError::InvalidResponse);
                }
                Ok(OrderBookLevel {
                    price: level[0].parse().map_err(|_| BybitError::InvalidResponse)?,
                    quantity: level[1].parse().map_err(|_| BybitError::InvalidResponse)?,
                })
            })
            .collect();

        Ok(OrderBook {
            symbol: api_response.result.s,
            timestamp: api_response.result.ts,
            bids: bids?,
            asks: asks?,
        })
    }

    /// Fetch multiple symbols' klines in parallel
    pub async fn get_klines_multi(
        &self,
        symbols: &[&str],
        interval: &str,
        limit: usize,
    ) -> Vec<Result<(String, Vec<Kline>), BybitError>> {
        let futures: Vec<_> = symbols
            .iter()
            .map(|symbol| {
                let symbol = symbol.to_string();
                let client = self.clone();
                let interval = interval.to_string();
                async move {
                    let klines = client.get_klines(&symbol, &interval, limit).await?;
                    Ok((symbol, klines))
                }
            })
            .collect();

        futures::future::join_all(futures).await
    }
}

impl Clone for BybitClient {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            base_url: self.base_url.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = BybitClient::new();
        assert_eq!(client.base_url, "https://api.bybit.com");
    }

    #[test]
    fn test_client_with_custom_url() {
        let client = BybitClient::with_base_url("https://custom.api.com");
        assert_eq!(client.base_url, "https://custom.api.com");
    }
}
