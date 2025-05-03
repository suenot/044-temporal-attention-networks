//! Example: Fetch cryptocurrency data from Bybit
//!
//! This example demonstrates how to use the BybitClient to fetch
//! OHLCV candlestick data and order book snapshots.

use tabl::{BybitClient, Kline};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("TABL - Fetching Data from Bybit\n");
    println!("================================\n");

    let client = BybitClient::new();

    // Fetch BTC/USDT hourly data
    println!("Fetching BTC/USDT 1-hour candles...");
    let klines = client.get_klines("BTCUSDT", "60", 100).await?;
    println!("Fetched {} candles\n", klines.len());

    if klines.is_empty() {
        println!("No candles returned from Bybit.");
        return Ok(());
    }

    // Display first few candles
    println!("First 5 candles:");
    println!("{:<20} {:>12} {:>12} {:>12} {:>12} {:>15}",
        "Time", "Open", "High", "Low", "Close", "Volume");
    println!("{}", "-".repeat(95));

    for kline in klines.iter().take(5) {
        let time = chrono::DateTime::from_timestamp(kline.start_time / 1000, 0)
            .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        println!("{:<20} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>15.2}",
            time, kline.open, kline.high, kline.low, kline.close, kline.volume);
    }

    // Calculate some statistics
    let avg_volume: f64 = klines.iter().map(|k| k.volume).sum::<f64>() / klines.len() as f64;
    let avg_range: f64 = klines.iter().map(|k| k.range()).sum::<f64>() / klines.len() as f64;
    let bullish_count = klines.iter().filter(|k| k.is_bullish()).count();

    println!("\nStatistics:");
    println!("  Average Volume: {:.2}", avg_volume);
    println!("  Average Range: {:.2}", avg_range);
    println!("  Bullish Candles: {} ({:.1}%)",
        bullish_count, 100.0 * bullish_count as f64 / klines.len() as f64);

    // Fetch order book
    println!("\nFetching order book...");
    let orderbook = client.get_orderbook("BTCUSDT", 10).await?;

    println!("\nOrder Book (Top 5 levels):");
    println!("\n{:>15} {:>15}   {:>15} {:>15}",
        "Bid Price", "Bid Qty", "Ask Price", "Ask Qty");
    println!("{}", "-".repeat(65));

    for i in 0..5.min(orderbook.bids.len().min(orderbook.asks.len())) {
        println!("{:>15.2} {:>15.4}   {:>15.2} {:>15.4}",
            orderbook.bids[i].price, orderbook.bids[i].quantity,
            orderbook.asks[i].price, orderbook.asks[i].quantity);
    }

    let mid = orderbook.mid_price();
    if let Some(mid_val) = mid {
        println!("\nMid Price: {:.2}", mid_val);
    }
    if let (Some(spread), Some(mid_val)) = (orderbook.spread(), mid) {
        println!("Spread: {:.2} ({:.4}%)", spread, spread / mid_val * 100.0);
    } else if let Some(spread) = orderbook.spread() {
        println!("Spread: {:.2}", spread);
    }
    println!("Imbalance (5 levels): {:.4}", orderbook.imbalance(5));

    println!("\nDone!");
    Ok(())
}
