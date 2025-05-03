//! Temporal attention mechanism

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::Normal;

/// Type of attention mechanism
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttentionType {
    /// Additive attention (Bahdanau style)
    Additive,
    /// Scaled dot-product attention
    ScaledDotProduct,
    /// Multi-head attention
    MultiHead,
}

/// Temporal attention layer
///
/// Computes attention weights over time steps:
/// α = softmax(w · tanh(U · X^T))
#[derive(Debug, Clone)]
pub struct TemporalAttention {
    /// Hidden dimension for attention computation
    hidden_dim: usize,
    /// Input dimension
    input_dim: usize,
    /// Query projection [hidden_dim, input_dim]
    u: Array2<f64>,
    /// Attention weights [hidden_dim]
    w: Array1<f64>,
    /// Attention type
    attention_type: AttentionType,
    /// Number of heads (for multi-head attention)
    n_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Query projections for multi-head [n_heads, head_dim, input_dim]
    wq: Option<Vec<Array2<f64>>>,
    /// Key projections for multi-head [n_heads, head_dim, input_dim]
    wk: Option<Vec<Array2<f64>>>,
    /// Value projections for multi-head [n_heads, head_dim, input_dim]
    wv: Option<Vec<Array2<f64>>>,
    /// Output projection [input_dim, n_heads * head_dim]
    wo: Option<Array2<f64>>,
}

impl TemporalAttention {
    /// Create a new temporal attention layer
    ///
    /// # Arguments
    /// * `input_dim` - Dimension of input features
    /// * `hidden_dim` - Hidden dimension for attention computation
    /// * `attention_type` - Type of attention mechanism
    pub fn new(input_dim: usize, hidden_dim: usize, attention_type: AttentionType) -> Self {
        Self::with_heads(input_dim, hidden_dim, attention_type, 1)
    }

    /// Create a multi-head attention layer
    ///
    /// # Arguments
    /// * `input_dim` - Dimension of input features
    /// * `hidden_dim` - Hidden dimension for attention computation
    /// * `attention_type` - Type of attention mechanism
    /// * `n_heads` - Number of attention heads
    pub fn with_heads(
        input_dim: usize,
        hidden_dim: usize,
        attention_type: AttentionType,
        n_heads: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let std = (2.0 / (input_dim + hidden_dim) as f64).sqrt();
        let dist = Normal::new(0.0, std).unwrap();

        let u = Array2::from_shape_fn((hidden_dim, input_dim), |_| rng.sample(dist));
        let w = Array1::from_shape_fn(hidden_dim, |_| rng.sample(dist));

        let (wq, wk, wv, wo, head_dim) = if attention_type == AttentionType::MultiHead && n_heads > 1 {
            // Note: For best results, input_dim should be divisible by n_heads.
            // If not divisible, the last head_dim * n_heads < input_dim, which may
            // result in some information loss in the output projection.
            let head_dim = input_dim / n_heads;
            let head_std = (2.0 / (input_dim + head_dim) as f64).sqrt();
            let head_dist = Normal::new(0.0, head_std).unwrap();

            let wq: Vec<Array2<f64>> = (0..n_heads)
                .map(|_| Array2::from_shape_fn((head_dim, input_dim), |_| rng.sample(head_dist)))
                .collect();
            let wk: Vec<Array2<f64>> = (0..n_heads)
                .map(|_| Array2::from_shape_fn((head_dim, input_dim), |_| rng.sample(head_dist)))
                .collect();
            let wv: Vec<Array2<f64>> = (0..n_heads)
                .map(|_| Array2::from_shape_fn((head_dim, input_dim), |_| rng.sample(head_dist)))
                .collect();
            let wo = Array2::from_shape_fn((input_dim, n_heads * head_dim), |_| rng.sample(head_dist));

            (Some(wq), Some(wk), Some(wv), Some(wo), head_dim)
        } else {
            (None, None, None, None, hidden_dim)
        };

        Self {
            hidden_dim,
            input_dim,
            u,
            w,
            attention_type,
            n_heads,
            head_dim,
            wq,
            wk,
            wv,
            wo,
        }
    }

    /// Compute softmax over a vector
    ///
    /// Handles edge cases where sum is zero or non-finite (e.g., when all inputs
    /// are masked or contain NaN/Inf) by returning zeros.
    fn softmax(x: &Array1<f64>) -> Array1<f64> {
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_x = x.mapv(|v| (v - max_val).exp());
        let sum: f64 = exp_x.sum();
        if sum == 0.0 || !sum.is_finite() {
            return Array1::zeros(x.len());
        }
        exp_x / sum
    }

    /// Forward pass with additive attention
    ///
    /// # Arguments
    /// * `x` - Input tensor [T, D]
    /// * `mask` - Optional mask [T] where true values are masked out
    ///
    /// # Returns
    /// * `(context, attention_weights)` - Context vector [D] and attention weights [T]
    fn forward_additive(
        &self,
        x: &Array2<f64>,
        mask: Option<&Array1<bool>>,
    ) -> (Array1<f64>, Array1<f64>) {
        let t = x.nrows();

        // U · X^T -> [hidden_dim, T]
        let projected = self.u.dot(&x.t());

        // tanh(U · X^T) -> [hidden_dim, T]
        let activated = projected.mapv(|v| v.tanh());

        // w · tanh(U · X^T) -> [T]
        let mut scores = self.w.dot(&activated);

        // Apply mask if provided
        if let Some(m) = mask {
            for (i, &masked) in m.iter().enumerate() {
                if masked {
                    scores[i] = f64::NEG_INFINITY;
                }
            }
        }

        // softmax -> attention weights [T]
        let attention_weights = Self::softmax(&scores);

        // Context = X^T · α -> [D]
        let context = x.t().dot(&attention_weights);

        (context, attention_weights)
    }

    /// Forward pass with scaled dot-product attention
    fn forward_scaled_dot_product(
        &self,
        x: &Array2<f64>,
        mask: Option<&Array1<bool>>,
    ) -> (Array1<f64>, Array1<f64>) {
        let t = x.nrows();
        let d = x.ncols();

        // Use the last time step as query
        let query = x.row(t - 1).to_owned();

        // Compute dot products: Q · K^T / sqrt(d)
        let scale = (d as f64).sqrt();
        let mut scores = Array1::zeros(t);

        for i in 0..t {
            let key = x.row(i);
            scores[i] = query.dot(&key) / scale;
        }

        // Apply mask
        if let Some(m) = mask {
            for (i, &masked) in m.iter().enumerate() {
                if masked {
                    scores[i] = f64::NEG_INFINITY;
                }
            }
        }

        let attention_weights = Self::softmax(&scores);

        // Weighted sum of values
        let mut context = Array1::zeros(d);
        for i in 0..t {
            context = &context + &(x.row(i).to_owned() * attention_weights[i]);
        }

        (context, attention_weights)
    }

    /// Forward pass with multi-head attention
    fn forward_multi_head(
        &self,
        x: &Array2<f64>,
        mask: Option<&Array1<bool>>,
    ) -> (Array1<f64>, Array1<f64>) {
        let t = x.nrows();

        if self.wq.is_none() || self.wk.is_none() || self.wv.is_none() || self.wo.is_none() {
            // Fall back to additive if multi-head weights aren't set
            return self.forward_additive(x, mask);
        }

        let wq = self.wq.as_ref().unwrap();
        let wk = self.wk.as_ref().unwrap();
        let wv = self.wv.as_ref().unwrap();
        let wo = self.wo.as_ref().unwrap();

        // Compute attention for each head
        let mut head_outputs = Vec::with_capacity(self.n_heads);
        let mut all_weights = Array1::zeros(t);

        for head in 0..self.n_heads {
            // Project to query, key, value spaces
            let q = wq[head].dot(&x.row(t - 1).t()); // [head_dim]
            let scale = (self.head_dim as f64).sqrt();

            let mut scores = Array1::zeros(t);
            for i in 0..t {
                let k = wk[head].dot(&x.row(i).t()); // [head_dim]
                scores[i] = q.dot(&k) / scale;
            }

            // Apply mask
            if let Some(m) = mask {
                for (i, &masked) in m.iter().enumerate() {
                    if masked {
                        scores[i] = f64::NEG_INFINITY;
                    }
                }
            }

            let weights = Self::softmax(&scores);
            all_weights = &all_weights + &weights;

            // Compute weighted value
            let mut head_output = Array1::zeros(self.head_dim);
            for i in 0..t {
                let v = wv[head].dot(&x.row(i).t()); // [head_dim]
                head_output = &head_output + &(v * weights[i]);
            }

            head_outputs.push(head_output);
        }

        // Concatenate heads
        let mut concat = Array1::zeros(self.n_heads * self.head_dim);
        for (head, output) in head_outputs.iter().enumerate() {
            for (i, &val) in output.iter().enumerate() {
                concat[head * self.head_dim + i] = val;
            }
        }

        // Output projection
        let context = wo.dot(&concat);

        // Average attention weights
        all_weights = all_weights / self.n_heads as f64;

        (context, all_weights)
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input tensor [T, D]
    /// * `mask` - Optional mask [T] where true values are masked out
    ///
    /// # Returns
    /// * `(context, attention_weights)` - Context vector [D] and attention weights [T]
    pub fn forward(
        &self,
        x: &Array2<f64>,
        mask: Option<&Array1<bool>>,
    ) -> (Array1<f64>, Array1<f64>) {
        match self.attention_type {
            AttentionType::Additive => self.forward_additive(x, mask),
            AttentionType::ScaledDotProduct => self.forward_scaled_dot_product(x, mask),
            AttentionType::MultiHead => self.forward_multi_head(x, mask),
        }
    }

    /// Forward pass for a batch
    ///
    /// # Arguments
    /// * `batch` - Batch of inputs [batch_size, T, D]
    /// * `masks` - Optional masks [batch_size, T]
    ///
    /// # Returns
    /// * `(contexts, attention_weights)` - [batch_size, D] and [batch_size, T]
    pub fn forward_batch(
        &self,
        batch: &ndarray::Array3<f64>,
        masks: Option<&ndarray::Array2<bool>>,
    ) -> (Array2<f64>, Array2<f64>) {
        let batch_size = batch.shape()[0];
        let t = batch.shape()[1];
        let d = batch.shape()[2];

        let mut contexts = Array2::zeros((batch_size, d));
        let mut all_weights = Array2::zeros((batch_size, t));

        for i in 0..batch_size {
            let x = batch.slice(ndarray::s![i, .., ..]).to_owned();
            let mask = masks.map(|m| m.row(i).to_owned());

            let (context, weights) = self.forward(&x, mask.as_ref());

            contexts.row_mut(i).assign(&context);
            all_weights.row_mut(i).assign(&weights);
        }

        (contexts, all_weights)
    }

    /// Get the number of parameters
    pub fn num_params(&self) -> usize {
        let base = self.u.len() + self.w.len();

        if let (Some(wq), Some(wk), Some(wv), Some(wo)) = (&self.wq, &self.wk, &self.wv, &self.wo) {
            let multi_head_params: usize = wq.iter().map(|w| w.len()).sum::<usize>()
                + wk.iter().map(|w| w.len()).sum::<usize>()
                + wv.iter().map(|w| w.len()).sum::<usize>()
                + wo.len();
            base + multi_head_params
        } else {
            base
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_additive_attention() {
        let attn = TemporalAttention::new(32, 64, AttentionType::Additive);
        let x = Array2::from_shape_fn((100, 32), |_| rand::random::<f64>());

        let (context, weights) = attn.forward(&x, None);

        assert_eq!(context.len(), 32);
        assert_eq!(weights.len(), 100);

        // Weights should sum to 1
        let sum: f64 = weights.sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_dot_product_attention() {
        let attn = TemporalAttention::new(32, 64, AttentionType::ScaledDotProduct);
        let x = Array2::from_shape_fn((100, 32), |_| rand::random::<f64>());

        let (context, weights) = attn.forward(&x, None);

        assert_eq!(context.len(), 32);
        assert_eq!(weights.len(), 100);

        // Weights should sum to 1
        let sum: f64 = weights.sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_multi_head_attention() {
        let attn = TemporalAttention::with_heads(32, 64, AttentionType::MultiHead, 4);
        let x = Array2::from_shape_fn((100, 32), |_| rand::random::<f64>());

        let (context, weights) = attn.forward(&x, None);

        assert_eq!(context.len(), 32);
        assert_eq!(weights.len(), 100);

        // Weights should sum to 1
        let sum: f64 = weights.sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_attention_with_mask() {
        let attn = TemporalAttention::new(32, 64, AttentionType::Additive);
        let x = Array2::from_shape_fn((10, 32), |_| rand::random::<f64>());

        // Mask out first 5 time steps
        let mut mask = Array1::from_elem(10, false);
        for i in 0..5 {
            mask[i] = true;
        }

        let (_, weights) = attn.forward(&x, Some(&mask));

        // Masked positions should have ~0 weight
        for i in 0..5 {
            assert!(weights[i] < 1e-6);
        }

        // Remaining weights should sum to ~1
        let sum: f64 = weights.iter().skip(5).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_forward() {
        let attn = TemporalAttention::new(32, 64, AttentionType::Additive);
        let batch = ndarray::Array3::from_shape_fn((16, 100, 32), |_| rand::random::<f64>());

        let (contexts, weights) = attn.forward_batch(&batch, None);

        assert_eq!(contexts.shape(), &[16, 32]);
        assert_eq!(weights.shape(), &[16, 100]);
    }
}
