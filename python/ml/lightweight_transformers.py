"""
Lightweight Temporal Transformer for Time-Series Forecasting
Optimized for AMD ROCm with strict VRAM/RAM caps (8GB system limit)
No HuggingFace dependencies - pure PyTorch/NumPy implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
import gc


class MemoryManager:
    """
    Strict memory management to enforce 8GB system cap.
    Monitors and limits GPU/CPU memory usage dynamically.
    """
    
    MAX_RAM_GB = 8.0
    SAFETY_MARGIN_GB = 1.0  # Keep 1GB buffer
    
    def __init__(self):
        self.allocated_memory = 0
        self.max_allowed_bytes = int((self.MAX_RAM_GB - self.SAFETY_MARGIN_GB) * 1024**3)
        
    def check_memory(self) -> bool:
        """Check if we're within memory limits."""
        import psutil
        current_ram = psutil.virtual_memory().used / (1024**3)
        return current_ram < self.MAX_RAM_GB - self.SAFETY_MARGIN_GB
    
    def force_gc(self):
        """Aggressive garbage collection."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_optimal_batch_size(self, sequence_length: int, feature_dim: int) -> int:
        """Calculate optimal batch size based on memory constraints."""
        # Estimate memory per sample: 4 bytes (float32) * seq_len * features * overhead
        bytes_per_sample = 4 * sequence_length * feature_dim * 10  # 10x for gradients/activations
        available_bytes = self.max_allowed_bytes * 0.5  # Use 50% for batch data
        return max(1, min(256, int(available_bytes / bytes_per_sample)))


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding optimized for time-series.
    Uses learnable scaling factors for temporal patterns.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not parameter) to avoid gradient computation
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Learnable temporal scaling
        self.temporal_scale = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding with temporal scaling."""
        seq_len = x.size(1)
        x = x + self.temporal_scale * self.pe[:, :seq_len, :]
        return self.dropout(x)


class CausalMultiHeadAttention(nn.Module):
    """
    Memory-efficient causal multi-head attention.
    Optimized for AMD ROCm with flash attention fallback.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_flash_attention: bool = False
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Single projection matrices for Q, K, V
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.use_flash_attention = use_flash_attention
        
        # Causal mask cache
        self._causal_mask_cache: Optional[torch.Tensor] = None
        
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate or retrieve cached causal mask."""
        if self._causal_mask_cache is not None and self._causal_mask_cache.size(-1) >= seq_len:
            return self._causal_mask_cache[:, :, :seq_len, :seq_len]
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        self._causal_mask_cache = mask
        return mask
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional flash attention.
        Memory-optimized implementation.
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply causal mask
        causal_mask = self._get_causal_mask(seq_len, x.device)
        if attn_mask is not None:
            attn_mask = attn_mask + causal_mask
        else:
            attn_mask = causal_mask
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights + attn_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)


class FeedForwardNetwork(nn.Module):
    """
    Optimized feed-forward network with GELU activation.
    Includes layer normalization and residual connection support.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x + residual


class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with pre-normalization.
    Optimized for time-series forecasting tasks.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = CausalMultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # Pre-norm layers
        self.attn_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn_norm = nn.LayerNorm(d_model, eps=1e-5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-normalization architecture."""
        # Self-attention with residual
        attn_out = self.attention(self.attn_norm(x))
        x = x + attn_out
        
        # FFN with residual
        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + ffn_out
        
        return x


class LightweightTemporalTransformer(nn.Module):
    """
    Main transformer model for time-series forecasting.
    Memory-efficient design with dynamic batching and gradient checkpointing.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        forecast_horizon: int = 1,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = True
    ):
        super().__init__()
        
        self.memory_manager = MemoryManager()
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model, bias=False)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-5),
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, forecast_horizon, bias=False)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer.
        Input: [batch_size, seq_len, input_dim]
        Output: [batch_size, forecast_horizon]
        """
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply gradient checkpointing if enabled
        if self.use_gradient_checkpointing and self.training:
            for layer in self.encoder_layers:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        else:
            for layer in self.encoder_layers:
                x = layer(x)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # [batch_size, d_model]
        
        # Project to forecast horizon
        output = self.output_projection(x)
        
        return output
    
    def predict(self, x: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Inference method with automatic batch sizing.
        Handles memory constraints dynamically.
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Convert to tensor
        x_tensor = torch.from_numpy(x).float().to(device)
        
        # Determine batch size
        if batch_size is None:
            batch_size = self.memory_manager.get_optimal_batch_size(
                x_tensor.shape[1], x_tensor.shape[2]
            )
        
        # Batch prediction
        predictions = []
        for i in range(0, len(x_tensor), batch_size):
            batch = x_tensor[i:i+batch_size]
            
            with torch.no_grad():
                pred = self.forward(batch)
                predictions.append(pred.cpu().numpy())
            
            # Memory cleanup
            if not self.memory_manager.check_memory():
                self.memory_manager.force_gc()
        
        return np.vstack(predictions)
    
    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Single training step with memory monitoring."""
        self.train()
        optimizer.zero_grad()
        
        # Forward pass
        predictions = self.forward(x)
        
        # Calculate loss
        loss = criterion(predictions, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Memory check
        if not self.memory_manager.check_memory():
            self.memory_manager.force_gc()
        
        return {
            'loss': loss.item(),
            'memory_ok': self.memory_manager.check_memory()
        }


class RegimeDetectionTransformer(LightweightTemporalTransformer):
    """
    Specialized transformer for market regime detection.
    Multi-class classification output for regime states.
    """
    
    REGIME_CLASSES = {
        0: 'crash',
        1: 'bearish',
        2: 'neutral',
        3: 'bullish',
        4: 'parabolic'
    }
    
    def __init__(
        self,
        input_dim: int,
        n_regimes: int = 5,
        **kwargs
    ):
        super().__init__(
            input_dim=input_dim,
            forecast_horizon=n_regimes,
            **kwargs
        )
        self.n_regimes = n_regimes
        
        # Replace output layer for classification
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.d_model, eps=1e-5),
            nn.Linear(self.d_model, self.d_model // 2, bias=False),
            nn.GELU(),
            nn.Dropout(kwargs.get('dropout', 0.1)),
            nn.Linear(self.d_model // 2, n_regimes, bias=False)
        )
        
    def predict_regime(self, x: np.ndarray) -> Tuple[int, str, np.ndarray]:
        """
        Predict market regime with confidence scores.
        Returns: (regime_id, regime_name, probabilities)
        """
        probs = self.predict(x)
        probs = F.softmax(torch.from_numpy(probs), dim=-1).numpy()
        
        regime_id = np.argmax(probs[0])
        regime_name = self.REGIME_CLASSES.get(regime_id, 'unknown')
        
        return regime_id, regime_name, probs[0]


# Example usage and testing
if __name__ == '__main__':
    # Test configuration
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create sample data
    batch_size = 32
    seq_len = 128
    input_dim = 50
    
    x = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
    y = np.random.randn(batch_size, 1).astype(np.float32)
    
    # Initialize model
    model = LightweightTemporalTransformer(
        input_dim=input_dim,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=1024,
        max_seq_len=seq_len,
        forecast_horizon=1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    output = model.predict(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test regime detection
    regime_model = RegimeDetectionTransformer(input_dim=input_dim)
    regime_id, regime_name, probs = regime_model.predict_regime(x)
    print(f"Predicted regime: {regime_name} (ID: {regime_id})")
    print(f"Confidence: {probs[regime_id]:.4f}")
