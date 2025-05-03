"""
TABL Model Implementation in PyTorch

Provides:
- TABLConfig: Model configuration
- TABLModel: Main TABL model
- MultiHeadTABL: Multi-head attention variant
- BilinearLayer: Bilinear projection layer
- TemporalAttention: Temporal attention mechanism
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass, field
from enum import Enum


class OutputType(Enum):
    """Type of model output"""
    DIRECTION = "direction"      # 3-class: up/stationary/down
    REGRESSION = "regression"    # Continuous returns
    BINARY = "binary"           # 2-class: up/down


@dataclass
class TABLConfig:
    """
    Configuration for TABL model

    Example:
        config = TABLConfig(
            seq_len=100,
            input_dim=6,
            hidden_T=20,
            hidden_D=32
        )
    """
    # Input dimensions
    seq_len: int = 100          # Number of time steps
    input_dim: int = 6          # Number of input features

    # Bilinear layer dimensions
    hidden_T: int = 20          # Compressed temporal dimension
    hidden_D: int = 32          # Compressed feature dimension

    # Attention configuration
    attention_dim: int = 64     # Dimension for attention computation
    n_heads: int = 4            # Number of attention heads (for MultiHeadTABL)
    use_multihead: bool = False # Whether to use multi-head attention

    # Output configuration
    output_type: OutputType = OutputType.DIRECTION
    n_classes: int = 3          # Number of output classes (for classification)

    # Regularization
    dropout: float = 0.2
    use_batch_norm: bool = True

    # Training
    learning_rate: float = 0.001
    weight_decay: float = 1e-5

    def validate(self):
        """Validate configuration"""
        assert self.seq_len > 0, "seq_len must be positive"
        assert self.input_dim > 0, "input_dim must be positive"
        assert self.hidden_T > 0 and self.hidden_T <= self.seq_len, \
            "hidden_T must be positive and <= seq_len"
        assert self.hidden_D > 0, "hidden_D must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
        if self.use_multihead:
            assert self.n_heads > 0, "n_heads must be > 0 when use_multihead=True"
        if self.output_type in (OutputType.BINARY, OutputType.REGRESSION):
            assert self.n_classes == 1, \
                f"n_classes must be 1 for {self.output_type.value} outputs, got {self.n_classes}"

    @property
    def bilinear_output_dim(self) -> int:
        """Output dimension of bilinear layer"""
        return self.hidden_T * self.hidden_D

    @property
    def total_output_dim(self) -> int:
        """Total output dimension before classification head"""
        # Both single-head and multi-head variants produce same output dimension
        return self.bilinear_output_dim + self.input_dim


class BilinearLayer(nn.Module):
    """
    Bilinear projection layer: H = σ(W₁ · X · W₂ + b)

    Transforms (batch, T, D) → (batch, T', D') by projecting both
    temporal and feature dimensions simultaneously.

    Args:
        T_in: Input temporal dimension
        T_out: Output temporal dimension
        D_in: Input feature dimension
        D_out: Output feature dimension
        dropout: Dropout rate
        use_batch_norm: Whether to use batch normalization
    """

    def __init__(
        self,
        T_in: int,
        T_out: int,
        D_in: int,
        D_out: int,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()
        self.T_in = T_in
        self.T_out = T_out
        self.D_in = D_in
        self.D_out = D_out

        # Temporal projection: (T_out, T_in)
        self.W1 = nn.Parameter(torch.empty(T_out, T_in))
        # Feature projection: (D_in, D_out)
        self.W2 = nn.Parameter(torch.empty(D_in, D_out))
        # Bias: (T_out, D_out)
        self.bias = nn.Parameter(torch.zeros(T_out, D_out))

        # Initialize weights
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

        # Normalization and regularization
        self.batch_norm = nn.BatchNorm1d(T_out * D_out) if use_batch_norm else None
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor (batch, T_in, D_in)

        Returns:
            Output tensor (batch, T_out, D_out)
        """
        batch_size = x.size(0)

        # Bilinear transformation: H = W1 · X · W2 + b
        # Step 1: W1 · X: (batch, T_out, D_in)
        out = torch.matmul(self.W1, x)

        # Step 2: (W1 · X) · W2: (batch, T_out, D_out)
        out = torch.matmul(out, self.W2)

        # Add bias
        out = out + self.bias

        # Apply batch normalization (flatten → normalize → reshape)
        if self.batch_norm is not None:
            out = out.view(batch_size, -1)
            out = self.batch_norm(out)
            out = out.view(batch_size, self.T_out, self.D_out)

        # Activation and dropout
        out = self.activation(out)
        out = self.dropout(out)

        return out


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism: α = softmax(w · tanh(U · X^T))

    Learns to focus on important time steps by computing attention weights
    over the temporal dimension.

    Args:
        input_dim: Dimension of input features
        attention_dim: Dimension of attention hidden layer
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        attention_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        # Project features to attention space
        self.U = nn.Linear(input_dim, attention_dim, bias=True)

        # Attention query vector
        self.w = nn.Linear(attention_dim, 1, bias=False)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor (batch, T, D)
            mask: Optional mask tensor (batch, T) - 1 for valid, 0 for padding

        Returns:
            context: Context vector (batch, D)
            alpha: Attention weights (batch, T)
        """
        # Compute attention scores
        # h = tanh(U · x): (batch, T, attention_dim)
        h = torch.tanh(self.U(x))

        # scores = w · h: (batch, T)
        scores = self.w(h).squeeze(-1)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax over time dimension
        alpha = F.softmax(scores, dim=-1)
        alpha = self.dropout(alpha)

        # Weighted sum: context = X^T · α
        # alpha: (batch, T) → (batch, 1, T)
        # x: (batch, T, D)
        # context: (batch, D)
        context = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)

        return context, alpha


class TABL(nn.Module):
    """
    Temporal Attention-Augmented Bilinear Layer

    Combines bilinear projection with temporal attention.

    Args:
        config: TABLConfig configuration object
    """

    def __init__(self, config: TABLConfig):
        super().__init__()
        self.config = config

        # Bilinear projection
        self.bilinear = BilinearLayer(
            T_in=config.seq_len,
            T_out=config.hidden_T,
            D_in=config.input_dim,
            D_out=config.hidden_D,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm
        )

        # Temporal attention
        self.attention = TemporalAttention(
            input_dim=config.input_dim,
            attention_dim=config.attention_dim,
            dropout=config.dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass

        Args:
            x: Input tensor (batch, T, D)
            mask: Optional mask tensor (batch, T)
            return_attention: Whether to return attention weights

        Returns:
            out: Output features (batch, hidden_T*hidden_D + D)
            alpha: Attention weights (batch, T) - only if return_attention=True
        """
        batch_size = x.size(0)

        # Bilinear projection: (batch, T, D) → (batch, T', D')
        h = self.bilinear(x)

        # Flatten bilinear output: (batch, T' * D')
        h_flat = h.view(batch_size, -1)

        # Temporal attention: (batch, D), (batch, T)
        context, alpha = self.attention(x, mask)

        # Concatenate bilinear output and context
        out = torch.cat([h_flat, context], dim=-1)

        if return_attention:
            return out, alpha
        return out


class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-head temporal attention mechanism

    Uses multiple attention heads to capture different temporal patterns.

    Args:
        input_dim: Dimension of input features
        attention_dim: Dimension of attention hidden layer per head
        n_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        attention_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_heads = n_heads

        # Multiple attention heads
        self.attention_heads = nn.ModuleList([
            TemporalAttention(input_dim, attention_dim, dropout)
            for _ in range(n_heads)
        ])

        # Head combination layer
        self.head_combine = nn.Sequential(
            nn.Linear(n_heads * input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass

        Args:
            x: Input tensor (batch, T, D)
            mask: Optional mask tensor (batch, T)
            return_attention: Whether to return attention weights

        Returns:
            context: Combined context (batch, D)
            alphas: Attention weights (batch, n_heads, T) - only if return_attention=True
        """
        contexts = []
        alphas = []

        for head in self.attention_heads:
            ctx, alpha = head(x, mask)
            contexts.append(ctx)
            alphas.append(alpha)

        # Combine heads: (batch, n_heads * D) → (batch, D)
        multi_context = torch.cat(contexts, dim=-1)
        combined = self.head_combine(multi_context)

        if return_attention:
            # Stack alphas: (batch, n_heads, T)
            stacked_alphas = torch.stack(alphas, dim=1)
            return combined, stacked_alphas
        return combined


class MultiHeadTABL(nn.Module):
    """
    Multi-Head Temporal Attention Bilinear Layer

    Uses multiple attention heads to capture different temporal patterns.

    Args:
        config: TABLConfig configuration object
    """

    def __init__(self, config: TABLConfig):
        super().__init__()
        self.config = config

        # Shared bilinear projection
        self.bilinear = BilinearLayer(
            T_in=config.seq_len,
            T_out=config.hidden_T,
            D_in=config.input_dim,
            D_out=config.hidden_D,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm
        )

        # Multi-head temporal attention
        self.attention = MultiHeadTemporalAttention(
            input_dim=config.input_dim,
            attention_dim=config.attention_dim,
            n_heads=config.n_heads,
            dropout=config.dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass

        Args:
            x: Input tensor (batch, T, D)
            mask: Optional mask tensor (batch, T)
            return_attention: Whether to return attention weights

        Returns:
            out: Output features (batch, hidden_T*hidden_D + D)
            alpha: Attention weights (batch, n_heads, T) - only if return_attention=True
        """
        batch_size = x.size(0)

        # Bilinear projection
        h = self.bilinear(x)
        h_flat = h.view(batch_size, -1)

        # Multi-head temporal attention
        if return_attention:
            context, alphas = self.attention(x, mask, return_attention=True)
        else:
            context = self.attention(x, mask, return_attention=False)

        # Concatenate
        out = torch.cat([h_flat, context], dim=-1)

        if return_attention:
            return out, alphas
        return out


class TABLModel(nn.Module):
    """
    Complete TABL Model for Financial Time-Series Prediction

    Example:
        config = TABLConfig(seq_len=100, input_dim=6)
        model = TABLModel(config)

        x = torch.randn(32, 100, 6)  # [batch, seq_len, features]
        output = model(x)
        print(output['logits'].shape)  # [32, 3] for 3-class classification
    """

    def __init__(self, config: TABLConfig):
        super().__init__()
        config.validate()
        self.config = config

        # Input normalization
        self.input_norm = nn.LayerNorm(config.input_dim)

        # TABL layer (single or multi-head)
        if config.use_multihead:
            self.tabl = MultiHeadTABL(config)
        else:
            self.tabl = TABL(config)

        # Classification head
        hidden_dim = config.total_output_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim // 2, config.n_classes)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> dict:
        """
        Forward pass

        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mask: Optional mask tensor (batch, seq_len)
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with:
                - logits: Classification logits (batch, n_classes)
                - probs: Class probabilities (batch, n_classes)
                - attention: Attention weights (if return_attention=True)
        """
        # Input normalization
        x = self.input_norm(x)

        # TABL layer
        if return_attention:
            features, attention = self.tabl(x, mask, return_attention=True)
        else:
            features = self.tabl(x, mask, return_attention=False)
            attention = None

        # Classification
        logits = self.classifier(features)

        # Compute probabilities based on output type
        if self.config.output_type == OutputType.BINARY:
            probs = torch.sigmoid(logits)
        elif self.config.output_type == OutputType.REGRESSION:
            # For regression, probabilities don't apply - use logits directly
            probs = logits
        else:
            probs = F.softmax(logits, dim=-1)

        result = {
            'logits': logits,
            'probs': probs,
        }

        if return_attention:
            result['attention'] = attention

        return result

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions

        Args:
            x: Input tensor (batch, seq_len, input_dim)

        Returns:
            For classification: Predicted class labels (batch,)
            For regression: Continuous predictions (batch,)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            if self.config.output_type == OutputType.BINARY:
                return (output['probs'] > 0.5).long().squeeze(-1)
            if self.config.output_type == OutputType.REGRESSION:
                return output['logits'].squeeze(-1)
            return output['logits'].argmax(dim=-1)


def create_tabl_model(
    seq_len: int = 100,
    input_dim: int = 6,
    n_classes: int = 3,
    use_multihead: bool = False,
    n_heads: int = 4,
    dropout: float = 0.2
) -> TABLModel:
    """
    Factory function to create a TABL model

    Args:
        seq_len: Number of time steps
        input_dim: Number of input features
        n_classes: Number of output classes
        use_multihead: Whether to use multi-head attention
        n_heads: Number of attention heads
        dropout: Dropout rate

    Returns:
        TABLModel instance
    """
    config = TABLConfig(
        seq_len=seq_len,
        input_dim=input_dim,
        n_classes=n_classes,
        use_multihead=use_multihead,
        n_heads=n_heads,
        dropout=dropout
    )
    return TABLModel(config)


if __name__ == "__main__":
    # Test the model
    print("Testing TABL model...")

    # Single-head configuration
    config = TABLConfig(
        seq_len=100,
        input_dim=6,
        hidden_T=20,
        hidden_D=32,
        attention_dim=64,
        n_classes=3,
        use_multihead=False
    )

    model = TABLModel(config)
    print(f"Single-head parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(4, 100, 6)
    output = model(x, return_attention=True)

    print(f"Logits shape: {output['logits'].shape}")
    print(f"Probs shape: {output['probs'].shape}")
    print(f"Attention shape: {output['attention'].shape}")

    # Multi-head configuration
    config_mh = TABLConfig(
        seq_len=100,
        input_dim=6,
        hidden_T=20,
        hidden_D=32,
        attention_dim=64,
        n_heads=4,
        n_classes=3,
        use_multihead=True
    )

    model_mh = TABLModel(config_mh)
    print(f"\nMulti-head parameters: {sum(p.numel() for p in model_mh.parameters()):,}")

    output_mh = model_mh(x, return_attention=True)
    print(f"Multi-head logits shape: {output_mh['logits'].shape}")
    print(f"Multi-head attention shape: {output_mh['attention'].shape}")

    # Test predictions
    preds = model.predict(x)
    print(f"\nPredictions shape: {preds.shape}")
    print(f"Predictions: {preds}")
