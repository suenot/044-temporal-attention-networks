//! TABL (Temporal Attention-Augmented Bilinear Network) model

use ndarray::{Array1, Array2, Array3, Axis};
use rand::Rng;
use rand_distr::Normal;

use super::attention::{AttentionType, TemporalAttention};
use super::bilinear::BilinearLayer;
use crate::defaults;

/// Configuration for TABL model
#[derive(Debug, Clone)]
pub struct TABLConfig {
    /// Input sequence length (T)
    pub seq_len: usize,
    /// Number of input features (D)
    pub input_dim: usize,
    /// Compressed temporal dimension (T')
    pub hidden_t: usize,
    /// Compressed feature dimension (D')
    pub hidden_d: usize,
    /// Attention hidden dimension
    pub attention_dim: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of output classes
    pub n_classes: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Attention type
    pub attention_type: AttentionType,
}

impl Default for TABLConfig {
    fn default() -> Self {
        Self {
            seq_len: defaults::SEQ_LEN,
            input_dim: defaults::INPUT_DIM,
            hidden_t: defaults::HIDDEN_T,
            hidden_d: defaults::HIDDEN_D,
            attention_dim: defaults::ATTENTION_DIM,
            n_heads: defaults::N_HEADS,
            n_classes: defaults::N_CLASSES,
            dropout: defaults::DROPOUT,
            attention_type: AttentionType::Additive,
        }
    }
}

impl TABLConfig {
    /// Create a new configuration with custom parameters
    pub fn new(
        seq_len: usize,
        input_dim: usize,
        hidden_t: usize,
        hidden_d: usize,
        n_classes: usize,
    ) -> Self {
        Self {
            seq_len,
            input_dim,
            hidden_t,
            hidden_d,
            attention_dim: hidden_d * 2,
            n_heads: 4,
            n_classes,
            dropout: 0.2,
            attention_type: AttentionType::Additive,
        }
    }

    /// Set the attention type
    pub fn with_attention_type(mut self, attention_type: AttentionType) -> Self {
        self.attention_type = attention_type;
        self
    }

    /// Set the number of attention heads
    pub fn with_heads(mut self, n_heads: usize) -> Self {
        self.n_heads = n_heads;
        self
    }

    /// Set the dropout rate
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }
}

/// TABL (Temporal Attention-Augmented Bilinear Network) model
#[derive(Debug)]
pub struct TABLModel {
    /// Model configuration
    config: TABLConfig,
    /// Bilinear projection layer
    bilinear: BilinearLayer,
    /// Temporal attention layer
    attention: TemporalAttention,
    /// Output layer weights [n_classes, hidden_d]
    output_weights: Array2<f64>,
    /// Output layer bias [n_classes]
    output_bias: Array1<f64>,
}

impl TABLModel {
    /// Create a new TABL model
    pub fn new(config: TABLConfig) -> Self {
        let bilinear = BilinearLayer::new(
            config.seq_len,
            config.hidden_t,
            config.input_dim,
            config.hidden_d,
            config.dropout,
            true,
        );

        let attention = TemporalAttention::with_heads(
            config.hidden_d,
            config.attention_dim,
            config.attention_type,
            config.n_heads,
        );

        // Initialize output layer
        let mut rng = rand::thread_rng();
        let std = (2.0 / (config.hidden_d + config.n_classes) as f64).sqrt();
        let dist = Normal::new(0.0, std).unwrap();

        let output_weights = Array2::from_shape_fn(
            (config.n_classes, config.hidden_d),
            |_| rng.sample(dist),
        );
        let output_bias = Array1::zeros(config.n_classes);

        Self {
            config,
            bilinear,
            attention,
            output_weights,
            output_bias,
        }
    }

    /// Forward pass for a single sample
    ///
    /// # Arguments
    /// * `x` - Input tensor [seq_len, input_dim]
    /// * `training` - Whether in training mode
    ///
    /// # Returns
    /// * `(logits, attention_weights)` - Class logits and attention weights
    pub fn forward(
        &self,
        x: &Array2<f64>,
        training: bool,
    ) -> (Array1<f64>, Array1<f64>) {
        // Bilinear projection: [seq_len, input_dim] -> [hidden_t, hidden_d]
        let h = self.bilinear.forward(x, training);

        // Temporal attention: [hidden_t, hidden_d] -> [hidden_d]
        let (context, attention_weights) = self.attention.forward(&h, None);

        // Output layer: [hidden_d] -> [n_classes]
        let logits = self.output_weights.dot(&context) + &self.output_bias;

        (logits, attention_weights)
    }

    /// Forward pass for a batch
    ///
    /// # Arguments
    /// * `batch` - Batch of inputs [batch_size, seq_len, input_dim]
    /// * `training` - Whether in training mode
    ///
    /// # Returns
    /// * `(logits, attention_weights)` - [batch_size, n_classes] and [batch_size, hidden_t]
    pub fn forward_batch(
        &self,
        batch: &Array3<f64>,
        training: bool,
    ) -> (Array2<f64>, Array2<f64>) {
        let batch_size = batch.shape()[0];

        let mut logits = Array2::zeros((batch_size, self.config.n_classes));
        let mut all_weights = Array2::zeros((batch_size, self.config.hidden_t));

        for i in 0..batch_size {
            let x = batch.slice(ndarray::s![i, .., ..]).to_owned();
            let (sample_logits, weights) = self.forward(&x, training);

            logits.row_mut(i).assign(&sample_logits);
            all_weights.row_mut(i).assign(&weights);
        }

        (logits, all_weights)
    }

    /// Compute softmax probabilities from logits
    ///
    /// Handles edge cases where sum is zero or non-finite (NaN/Inf) by
    /// returning a uniform distribution.
    pub fn softmax(logits: &Array1<f64>) -> Array1<f64> {
        let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_logits = logits.mapv(|v| (v - max_val).exp());
        let sum: f64 = exp_logits.sum();
        if !sum.is_finite() || sum == 0.0 {
            let n = logits.len().max(1) as f64;
            return Array1::from_elem(logits.len(), 1.0 / n);
        }
        exp_logits / sum
    }

    /// Predict class probabilities for a single sample
    pub fn predict_proba(&self, x: &Array2<f64>) -> Array1<f64> {
        let (logits, _) = self.forward(x, false);
        Self::softmax(&logits)
    }

    /// Predict class for a single sample
    pub fn predict(&self, x: &Array2<f64>) -> usize {
        let proba = self.predict_proba(x);
        proba
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Predict classes for a batch
    pub fn predict_batch(&self, batch: &Array3<f64>) -> Vec<usize> {
        let (logits, _) = self.forward_batch(batch, false);

        (0..logits.nrows())
            .map(|i| {
                let row = logits.row(i);
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Compute cross-entropy loss
    pub fn compute_loss(&self, logits: &Array2<f64>, targets: &[i32]) -> f64 {
        let batch_size = logits.nrows();
        let mut total_loss = 0.0;

        for i in 0..batch_size {
            let row = logits.row(i);
            let proba = Self::softmax(&row.to_owned());
            let target = targets[i] as usize;

            // Cross-entropy: -log(p[target])
            let p = proba[target].max(1e-10);
            total_loss -= p.ln();
        }

        total_loss / batch_size as f64
    }

    /// Compute accuracy
    pub fn compute_accuracy(&self, predictions: &[usize], targets: &[i32]) -> f64 {
        let correct = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(&pred, &target)| pred == target as usize)
            .count();

        correct as f64 / predictions.len() as f64
    }

    /// Get the model configuration
    pub fn config(&self) -> &TABLConfig {
        &self.config
    }

    /// Get the total number of parameters
    pub fn num_params(&self) -> usize {
        self.bilinear.num_params()
            + self.attention.num_params()
            + self.output_weights.len()
            + self.output_bias.len()
    }

    /// Get attention weights for interpretability
    pub fn get_attention_weights(&self, x: &Array2<f64>) -> Array1<f64> {
        let h = self.bilinear.forward(x, false);
        let (_, attention_weights) = self.attention.forward(&h, None);
        attention_weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tabl_forward() {
        let config = TABLConfig::default();
        let model = TABLModel::new(config.clone());

        let x = Array2::from_shape_fn(
            (config.seq_len, config.input_dim),
            |_| rand::random::<f64>(),
        );

        let (logits, weights) = model.forward(&x, false);

        assert_eq!(logits.len(), config.n_classes);
        assert_eq!(weights.len(), config.hidden_t);
    }

    #[test]
    fn test_tabl_batch() {
        let config = TABLConfig::new(50, 6, 10, 16, 3);
        let model = TABLModel::new(config.clone());

        let batch = Array3::from_shape_fn(
            (16, config.seq_len, config.input_dim),
            |_| rand::random::<f64>(),
        );

        let (logits, weights) = model.forward_batch(&batch, false);

        assert_eq!(logits.shape(), &[16, 3]);
        assert_eq!(weights.shape(), &[16, 10]);
    }

    #[test]
    fn test_tabl_predict() {
        let config = TABLConfig::new(50, 6, 10, 16, 3);
        let model = TABLModel::new(config.clone());

        let x = Array2::from_shape_fn(
            (config.seq_len, config.input_dim),
            |_| rand::random::<f64>(),
        );

        let pred = model.predict(&x);
        assert!(pred < 3);

        let proba = model.predict_proba(&x);
        assert_eq!(proba.len(), 3);

        // Probabilities should sum to 1
        let sum: f64 = proba.sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tabl_loss() {
        let config = TABLConfig::new(50, 6, 10, 16, 3);
        let model = TABLModel::new(config);

        let logits = Array2::from_shape_fn((4, 3), |_| rand::random::<f64>());
        let targets = vec![0, 1, 2, 1];

        let loss = model.compute_loss(&logits, &targets);
        assert!(loss.is_finite());
        assert!(loss > 0.0);
    }

    #[test]
    fn test_tabl_accuracy() {
        let config = TABLConfig::new(50, 6, 10, 16, 3);
        let model = TABLModel::new(config);

        let predictions = vec![0, 1, 2, 1];
        let targets = vec![0, 1, 0, 1];

        let accuracy = model.compute_accuracy(&predictions, &targets);
        assert!((accuracy - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_tabl_with_multi_head_attention() {
        let config = TABLConfig::new(50, 6, 10, 16, 3)
            .with_attention_type(AttentionType::MultiHead)
            .with_heads(4);

        let model = TABLModel::new(config.clone());

        let x = Array2::from_shape_fn(
            (config.seq_len, config.input_dim),
            |_| rand::random::<f64>(),
        );

        let (logits, _) = model.forward(&x, false);
        assert_eq!(logits.len(), 3);
    }

    #[test]
    fn test_num_params() {
        let config = TABLConfig::new(50, 6, 10, 16, 3);
        let model = TABLModel::new(config);

        let params = model.num_params();
        assert!(params > 0);
    }
}
