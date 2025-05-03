//! Signal generation for trading strategies

use crate::model::TABLModel;
use ndarray::Array2;

/// Trading signal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    /// Buy signal
    Buy,
    /// Sell signal
    Sell,
    /// Hold position
    Hold,
}

impl Signal {
    /// Convert to numeric value for calculations
    pub fn to_position(&self) -> f64 {
        match self {
            Signal::Buy => 1.0,
            Signal::Sell => -1.0,
            Signal::Hold => 0.0,
        }
    }

    /// Create from class prediction
    pub fn from_class(class: usize) -> Self {
        match class {
            0 => Signal::Sell,
            2 => Signal::Buy,
            _ => Signal::Hold,
        }
    }
}

/// Signal generator trait
pub trait SignalGenerator {
    /// Generate a trading signal from input data
    fn generate_signal(&self, x: &Array2<f64>) -> Signal;

    /// Generate signals for a batch
    fn generate_signals(&self, batch: &ndarray::Array3<f64>) -> Vec<Signal>;
}

/// TABL-based trading strategy
pub struct TABLStrategy {
    /// TABL model for predictions
    model: TABLModel,
    /// Probability threshold for generating signals
    threshold: f64,
    /// Minimum confidence for signal generation
    min_confidence: f64,
}

impl TABLStrategy {
    /// Create a new TABL strategy
    pub fn new(model: TABLModel) -> Self {
        Self {
            model,
            threshold: 0.5,
            min_confidence: 0.4,
        }
    }

    /// Set the probability threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the minimum confidence
    pub fn with_min_confidence(mut self, min_confidence: f64) -> Self {
        self.min_confidence = min_confidence;
        self
    }

    /// Get the underlying model
    pub fn model(&self) -> &TABLModel {
        &self.model
    }

    /// Get attention weights for interpretability
    pub fn get_attention_weights(&self, x: &Array2<f64>) -> ndarray::Array1<f64> {
        self.model.get_attention_weights(x)
    }
}

impl SignalGenerator for TABLStrategy {
    fn generate_signal(&self, x: &Array2<f64>) -> Signal {
        let proba = self.model.predict_proba(x);

        // Find max probability and corresponding class
        // Handle NaN values by treating them as smaller than any valid value
        let (max_class, max_prob) = proba
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((1, &0.0)); // Default to Hold class if all values are NaN

        // Only generate signal if confidence is high enough
        if *max_prob < self.min_confidence {
            return Signal::Hold;
        }

        // Apply threshold for non-hold signals
        match max_class {
            0 if proba[0] > self.threshold => Signal::Sell,
            2 if proba[2] > self.threshold => Signal::Buy,
            _ => Signal::Hold,
        }
    }

    fn generate_signals(&self, batch: &ndarray::Array3<f64>) -> Vec<Signal> {
        let batch_size = batch.shape()[0];

        (0..batch_size)
            .map(|i| {
                let x = batch.slice(ndarray::s![i, .., ..]).to_owned();
                self.generate_signal(&x)
            })
            .collect()
    }
}

/// Simple threshold-based strategy for comparison
pub struct ThresholdStrategy {
    /// Buy threshold (positive return)
    buy_threshold: f64,
    /// Sell threshold (negative return)
    sell_threshold: f64,
}

impl ThresholdStrategy {
    /// Create a new threshold strategy
    pub fn new(buy_threshold: f64, sell_threshold: f64) -> Self {
        Self {
            buy_threshold,
            sell_threshold,
        }
    }
}

impl SignalGenerator for ThresholdStrategy {
    fn generate_signal(&self, x: &Array2<f64>) -> Signal {
        // Use the last row's first feature (assumed to be returns)
        let last_return = x[[x.nrows() - 1, 0]];

        if last_return > self.buy_threshold {
            Signal::Buy
        } else if last_return < self.sell_threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }

    fn generate_signals(&self, batch: &ndarray::Array3<f64>) -> Vec<Signal> {
        let batch_size = batch.shape()[0];

        (0..batch_size)
            .map(|i| {
                let x = batch.slice(ndarray::s![i, .., ..]).to_owned();
                self.generate_signal(&x)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TABLConfig;

    #[test]
    fn test_signal_conversion() {
        assert_eq!(Signal::Buy.to_position(), 1.0);
        assert_eq!(Signal::Sell.to_position(), -1.0);
        assert_eq!(Signal::Hold.to_position(), 0.0);
    }

    #[test]
    fn test_signal_from_class() {
        assert_eq!(Signal::from_class(0), Signal::Sell);
        assert_eq!(Signal::from_class(1), Signal::Hold);
        assert_eq!(Signal::from_class(2), Signal::Buy);
    }

    #[test]
    fn test_tabl_strategy() {
        let config = TABLConfig::new(50, 6, 10, 16, 3);
        let model = TABLModel::new(config.clone());
        let strategy = TABLStrategy::new(model);

        let x = Array2::from_shape_fn(
            (config.seq_len, config.input_dim),
            |_| rand::random::<f64>(),
        );

        let signal = strategy.generate_signal(&x);
        assert!(matches!(signal, Signal::Buy | Signal::Sell | Signal::Hold));
    }

    #[test]
    fn test_threshold_strategy() {
        let strategy = ThresholdStrategy::new(0.001, -0.001);

        // Positive return
        let mut x = Array2::zeros((10, 6));
        x[[9, 0]] = 0.002;
        assert_eq!(strategy.generate_signal(&x), Signal::Buy);

        // Negative return
        x[[9, 0]] = -0.002;
        assert_eq!(strategy.generate_signal(&x), Signal::Sell);

        // Neutral
        x[[9, 0]] = 0.0005;
        assert_eq!(strategy.generate_signal(&x), Signal::Hold);
    }
}
