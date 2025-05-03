//! Bilinear projection layer for temporal data

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand_distr::Normal;

/// Bilinear layer for joint temporal-feature projection
///
/// Performs the operation: H = σ(W₁ · X · W₂ + b)
/// - W₁ projects along temporal dimension
/// - W₂ projects along feature dimension
#[derive(Debug, Clone)]
pub struct BilinearLayer {
    /// Temporal projection matrix [T_out, T_in]
    w1: Array2<f64>,
    /// Feature projection matrix [D_in, D_out]
    w2: Array2<f64>,
    /// Bias term [T_out, D_out]
    bias: Array2<f64>,
    /// Input temporal dimension
    t_in: usize,
    /// Output temporal dimension
    t_out: usize,
    /// Input feature dimension
    d_in: usize,
    /// Output feature dimension
    d_out: usize,
    /// Dropout rate
    dropout: f64,
    /// Use batch normalization
    use_batch_norm: bool,
    /// Running mean for batch norm
    running_mean: Option<Array2<f64>>,
    /// Running variance for batch norm
    running_var: Option<Array2<f64>>,
}

impl BilinearLayer {
    /// Create a new bilinear layer
    ///
    /// # Arguments
    /// * `t_in` - Input temporal dimension
    /// * `t_out` - Output temporal dimension
    /// * `d_in` - Input feature dimension
    /// * `d_out` - Output feature dimension
    /// * `dropout` - Dropout rate (0.0 - 1.0, exclusive of 1.0)
    /// * `use_batch_norm` - Whether to use batch normalization
    ///
    /// # Panics
    /// Panics if dropout is not in the range [0.0, 1.0)
    pub fn new(
        t_in: usize,
        t_out: usize,
        d_in: usize,
        d_out: usize,
        dropout: f64,
        use_batch_norm: bool,
    ) -> Self {
        assert!(
            (0.0..1.0).contains(&dropout),
            "dropout must be in [0, 1), got {}",
            dropout
        );
        let mut rng = rand::thread_rng();

        // Xavier/Glorot initialization
        let w1_std = (2.0 / (t_in + t_out) as f64).sqrt();
        let w2_std = (2.0 / (d_in + d_out) as f64).sqrt();

        let w1_dist = Normal::new(0.0, w1_std).unwrap();
        let w2_dist = Normal::new(0.0, w2_std).unwrap();

        let w1 = Array2::from_shape_fn((t_out, t_in), |_| rng.sample(w1_dist));
        let w2 = Array2::from_shape_fn((d_in, d_out), |_| rng.sample(w2_dist));
        let bias = Array2::zeros((t_out, d_out));

        let (running_mean, running_var) = if use_batch_norm {
            (
                Some(Array2::zeros((t_out, d_out))),
                Some(Array2::ones((t_out, d_out))),
            )
        } else {
            (None, None)
        };

        Self {
            w1,
            w2,
            bias,
            t_in,
            t_out,
            d_in,
            d_out,
            dropout,
            use_batch_norm,
            running_mean,
            running_var,
        }
    }

    /// Forward pass
    ///
    /// # Arguments
    /// * `x` - Input tensor [T_in, D_in]
    /// * `training` - Whether in training mode (applies dropout)
    ///
    /// # Returns
    /// * Output tensor [T_out, D_out]
    pub fn forward(&self, x: &Array2<f64>, training: bool) -> Array2<f64> {
        assert_eq!(x.nrows(), self.t_in, "Temporal dimension mismatch");
        assert_eq!(x.ncols(), self.d_in, "Feature dimension mismatch");

        // H = W₁ · X · W₂ + b
        let h1 = self.w1.dot(x);          // [T_out, D_in]
        let mut h = h1.dot(&self.w2);     // [T_out, D_out]
        h = &h + &self.bias;

        // Batch normalization (simplified - inference-only)
        // Note: This implementation uses pre-computed running statistics and does not
        // update them during training. For proper training, batch statistics should be
        // computed and running statistics updated with exponential moving average.
        // This simplified version is suitable for inference with pre-trained models.
        if self.use_batch_norm {
            if let (Some(running_mean), Some(running_var)) = (&self.running_mean, &self.running_var) {
                h = (&h - running_mean) / (running_var.mapv(|v| (v + 1e-5).sqrt()));
            }
        }

        // ReLU activation
        h.mapv_inplace(|v| v.max(0.0));

        // Dropout during training
        if training && self.dropout > 0.0 {
            let mut rng = rand::thread_rng();
            h.mapv_inplace(|v| {
                if rng.gen::<f64>() < self.dropout {
                    0.0
                } else {
                    v / (1.0 - self.dropout)
                }
            });
        }

        h
    }

    /// Forward pass for a batch
    ///
    /// # Arguments
    /// * `batch` - Batch of inputs [batch_size, T_in, D_in]
    /// * `training` - Whether in training mode
    ///
    /// # Returns
    /// * Batch of outputs [batch_size, T_out, D_out]
    pub fn forward_batch(
        &self,
        batch: &ndarray::Array3<f64>,
        training: bool,
    ) -> ndarray::Array3<f64> {
        let batch_size = batch.shape()[0];
        let mut output = ndarray::Array3::zeros((batch_size, self.t_out, self.d_out));

        for i in 0..batch_size {
            let x = batch.slice(ndarray::s![i, .., ..]).to_owned();
            let y = self.forward(&x, training);
            output.slice_mut(ndarray::s![i, .., ..]).assign(&y);
        }

        output
    }

    /// Get the output shape
    pub fn output_shape(&self) -> (usize, usize) {
        (self.t_out, self.d_out)
    }

    /// Get the number of parameters
    pub fn num_params(&self) -> usize {
        self.w1.len() + self.w2.len() + self.bias.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilinear_forward() {
        let layer = BilinearLayer::new(100, 20, 6, 32, 0.1, false);
        let x = Array2::from_shape_fn((100, 6), |_| rand::random::<f64>());

        let y = layer.forward(&x, false);

        assert_eq!(y.shape(), &[20, 32]);
    }

    #[test]
    fn test_bilinear_batch() {
        let layer = BilinearLayer::new(50, 10, 4, 16, 0.0, false);
        let batch = ndarray::Array3::from_shape_fn((8, 50, 4), |_| rand::random::<f64>());

        let output = layer.forward_batch(&batch, false);

        assert_eq!(output.shape(), &[8, 10, 16]);
    }

    #[test]
    fn test_bilinear_with_batch_norm() {
        let layer = BilinearLayer::new(100, 20, 6, 32, 0.1, true);
        let x = Array2::from_shape_fn((100, 6), |_| rand::random::<f64>());

        let y = layer.forward(&x, false);

        assert_eq!(y.shape(), &[20, 32]);
    }

    #[test]
    fn test_num_params() {
        let layer = BilinearLayer::new(100, 20, 6, 32, 0.1, false);
        let expected = 100 * 20 + 6 * 32 + 20 * 32;
        assert_eq!(layer.num_params(), expected);
    }
}
