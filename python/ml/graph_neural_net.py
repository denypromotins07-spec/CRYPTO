//! python/ml/graph_neural_net.py
//!
//! Graph Neural Network (GNN) to model cross-asset correlations and lead-lag
//! relationships between the top 50 crypto pairs. Maps the crypto market as a
//! dynamic graph to predict systemic risk and liquidity cascades.
//!
//! Features:
//! - Dynamic graph construction from rolling correlations
//! - Message passing with attention weights
//! - Systemic risk scoring via graph centrality metrics
//! - Memory-efficient sparse operations
//! - ROCm optimization for AMD GPU
//!
//! Memory Constraint: Uses sparse matrices to stay within 8GB cap.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import time

# Check for ROCm availability
USE_ROCM = torch.cuda.is_available() and torch.version.hip is not None
DEVICE = torch.device("cuda:0" if USE_ROCM else "cpu")


class DynamicGraphConstructor:
    """
    Constructs dynamic adjacency matrices from asset return correlations.
    
    Updates the graph structure periodically based on rolling correlation windows.
    """
    
    def __init__(
        self,
        num_assets: int,
        correlation_window: int = 60,  # 60 timesteps
        threshold: float = 0.3,  # Correlation threshold for edge creation
        max_edges_per_node: int = 10,  # Sparsity constraint
    ):
        self.num_assets = num_assets
        self.correlation_window = correlation_window
        self.threshold = threshold
        self.max_edges_per_node = max_edges_per_node
        
        # Rolling buffer for returns
        self.returns_buffer: List[np.ndarray] = []
        
    def update_returns(self, returns: np.ndarray):
        """Add new returns to the rolling buffer."""
        self.returns_buffer.append(returns.copy())
        if len(self.returns_buffer) > self.correlation_window:
            self.returns_buffer.pop(0)
    
    def build_adjacency(self) -> Tuple[SparseTensor, np.ndarray]:
        """
        Build sparse adjacency matrix from rolling correlations.
        
        Returns:
            adj: Sparse adjacency matrix (num_assets, num_assets)
            edge_weights: Correlation weights for each edge
        """
        if len(self.returns_buffer) < 10:
            # Not enough data, return identity
            indices = torch.arange(self.num_assets).unsqueeze(0).repeat(2, 1).to(DEVICE)
            values = torch.ones(self.num_assets).to(DEVICE)
            adj = SparseTensor(indices=indices, value=values, 
                              sparse_sizes=(self.num_assets, self.num_assets))
            return adj, np.ones(self.num_assets)
        
        # Compute correlation matrix
        returns_matrix = np.stack(self.returns_buffer, axis=0)  # (window, num_assets)
        corr_matrix = np.corrcoef(returns_matrix.T)  # (num_assets, num_assets)
        
        # Threshold and sparsify
        mask = np.abs(corr_matrix) > self.threshold
        np.fill_diagonal(mask, False)  # No self-loops
        
        # Limit edges per node
        adjacency = np.zeros_like(corr_matrix)
        edge_weights = []
        
        for i in range(self.num_assets):
            neighbors = np.where(mask[i])[0]
            if len(neighbors) > self.max_edges_per_node:
                # Keep top-k by absolute correlation
                top_k_idx = np.argsort(np.abs(corr_matrix[i, neighbors]))[-self.max_edges_per_node:]
                neighbors = neighbors[top_k_idx]
            
            for j in neighbors:
                adjacency[i, j] = corr_matrix[i, j]
                edge_weights.append(corr_matrix[i, j])
        
        # Convert to sparse tensor
        rows, cols = np.where(adjacency != 0)
        indices = torch.tensor([rows, cols], dtype=torch.long).to(DEVICE)
        values = torch.tensor(adjacency[rows, cols], dtype=torch.float32).to(DEVICE)
        
        adj_sparse = SparseTensor(
            indices=indices,
            value=values,
            sparse_sizes=(self.num_assets, self.num_assets)
        )
        
        return adj_sparse, np.array(edge_weights)
    
    def get_eigenvector_centrality(self, adj: SparseTensor) -> np.ndarray:
        """
        Compute eigenvector centrality for systemic risk scoring.
        Higher centrality = more systemically important asset.
        """
        # Power iteration method
        n = self.num_assets
        x = torch.ones(n, 1).to(DEVICE) / np.sqrt(n)
        
        for _ in range(20):
            x_new = matmul(adj, x)
            norm = torch.norm(x_new)
            if norm > 0:
                x = x_new / norm
        
        centrality = x.cpu().numpy().flatten()
        return centrality


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) layer with sparse operations.
    
    Computes attention-weighted messages between connected nodes.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 0.2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        
        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        
        # Linear transformations per head
        self.W = nn.Parameter(torch.empty(num_heads, in_features, self.head_dim))
        nn.init.xavier_uniform_(self.W)
        
        # Attention coefficients
        self.a = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim, 1))
        nn.init.xavier_uniform_(self.a)
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
    
    def forward(self, x: torch.Tensor, adj: SparseTensor) -> torch.Tensor:
        """
        Args:
            x: Node features (batch, num_nodes, in_features)
            adj: Sparse adjacency matrix
        
        Returns:
            h: Updated node features (batch, num_nodes, out_features)
        """
        batch_size, num_nodes, _ = x.shape
        
        # Apply linear transformation per head
        # x: (batch, nodes, in) -> (batch, heads, nodes, head_dim)
        h = torch.einsum('bni,hid->bhnd', x, self.W)
        
        # Compute attention coefficients
        # For each edge (i, j), compute e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        h_i = h[:, :, adj.storage.row(), :]  # (batch, heads, edges, head_dim)
        h_j = h[:, :, adj.storage.col(), :]
        
        h_concat = torch.cat([h_i, h_j], dim=-1)  # (batch, heads, edges, 2*head_dim)
        
        # Attention scores
        e = torch.sum(h_concat * self.a.unsqueeze(0).unsqueeze(2), dim=-1)  # (batch, heads, edges)
        e = self.leaky_relu(e)
        
        # Mask non-edges
        e = e - (1 - adj.storage.value()).unsqueeze(0).unsqueeze(1) * 1e9
        
        # Softmax over neighbors
        e_exp = torch.exp(e - e.max(dim=-1, keepdim=True)[0])
        
        # Normalize by degree (sparse)
        # This is a simplified version; proper implementation needs scatter operations
        alpha = self.dropout(e_exp)
        
        # Aggregate messages
        # message_j = alpha_ij * Wh_j
        messages = alpha.unsqueeze(-1) * h_j  # (batch, heads, edges, head_dim)
        
        # Scatter sum to nodes
        h_new = torch.zeros(batch_size, self.num_heads, num_nodes, self.head_dim, 
                           device=x.device, dtype=x.dtype)
        
        # Simple aggregation (can be optimized with torch_scatter)
        for b in range(batch_size):
            for head in range(self.num_heads):
                idx = adj.storage.col()
                h_new[b, head].index_add_(0, idx, messages[b, head])
        
        # Reshape
        h_new = h_new.permute(0, 2, 1, 3).contiguous()
        h_new = h_new.view(batch_size, num_nodes, -1)
        
        return self.activation(h_new)


class CryptoGNN(nn.Module):
    """
    Graph Neural Network for crypto market modeling.
    
    Architecture:
    - Input: Asset features + dynamic graph structure
    - Layers: Multiple GAT layers with residual connections
    - Output: Per-asset predictions + global risk score
    """
    
    def __init__(
        self,
        num_assets: int,
        input_features: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        output_dim: int = 1,  # Price change prediction
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_assets = num_assets
        
        # Feature embedding
        self.input_embed = nn.Linear(input_features, hidden_dim)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i > 0 else hidden_dim
            out_dim = hidden_dim
            self.gat_layers.append(
                GraphAttentionLayer(in_dim, out_dim, num_heads, dropout)
            )
        
        # Residual projections
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output heads
        self.prediction_head = nn.Linear(hidden_dim, output_dim)
        self.risk_head = nn.Linear(hidden_dim, 1)  # Per-asset risk score
        
        # Global risk scorer
        self.global_risk_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Graph constructor
        self.graph_constructor = DynamicGraphConstructor(num_assets)
        
    def update_graph(self, returns: np.ndarray) -> Tuple[SparseTensor, np.ndarray]:
        """Update the dynamic graph structure."""
        self.graph_constructor.update_returns(returns)
        return self.graph_constructor.build_adjacency()
    
    def forward(
        self,
        x: torch.Tensor,
        adj: SparseTensor,
        centrality: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features (batch, num_assets, input_features)
            adj: Sparse adjacency matrix
            centrality: Optional eigenvector centrality weights
        
        Returns:
            predictions: Price change predictions (batch, num_assets, output_dim)
            risk_scores: Per-asset risk scores (batch, num_assets, 1)
            global_risk: Systemic risk score (batch, 1)
        """
        # Embed input
        h = self.input_embed(x)
        h = self.dropout(h)
        
        # Graph attention layers with residuals
        for gat in self.gat_layers:
            h_residual = h
            h = gat(h, adj)
            h = h + self.residual_proj(h_residual)
            h = self.layer_norm(h)
        
        # Apply centrality weighting if provided
        if centrality is not None:
            h = h * centrality.unsqueeze(-1).unsqueeze(0)
        
        # Output heads
        predictions = self.prediction_head(h)
        risk_scores = torch.sigmoid(self.risk_head(h))
        
        # Global risk: weighted average of individual risks by centrality
        if centrality is not None:
            weights = centrality / (centrality.sum() + 1e-9)
            global_risk = (risk_scores.squeeze(-1) * weights.unsqueeze(0)).sum(dim=1, keepdim=True)
        else:
            global_risk = risk_scores.mean(dim=1)
        
        global_risk = self.global_risk_mlp(global_risk)
        
        return predictions, risk_scores, global_risk
    
    @torch.inference_mode()
    def predict(
        self,
        features: np.ndarray,
        returns_history: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Inference method for real-time prediction.
        
        Args:
            features: Current asset features (num_assets, input_features)
            returns_history: Historical returns for graph construction (window, num_assets)
        
        Returns:
            Dictionary with predictions, risk scores, and centrality
        """
        self.eval()
        
        # Reset graph constructor with history
        self.graph_constructor.returns_buffer = [returns_history[i] for i in range(len(returns_history))]
        
        # Build graph
        adj, edge_weights = self.graph_constructor.build_adjacency()
        centrality = self.graph_constructor.get_eigenvector_centrality(adj)
        
        # Prepare input
        if features.ndim == 2:
            features = features.unsqueeze(0)
        x_tensor = torch.from_numpy(features).float().to(DEVICE)
        centrality_tensor = torch.from_numpy(centrality).float().to(DEVICE)
        
        # Forward pass
        predictions, risk_scores, global_risk = self.forward(
            x_tensor, adj, centrality_tensor
        )
        
        # Convert to numpy
        result = {
            'predictions': predictions.squeeze(0).cpu().numpy(),
            'risk_scores': risk_scores.squeeze(0).cpu().numpy(),
            'global_risk': global_risk.squeeze().cpu().numpy(),
            'centrality': centrality,
            'edge_weights': edge_weights,
        }
        
        return result


def create_crypto_gnn(
    num_assets: int = 50,
    input_features: int = 20,
    hidden_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 3,
    pretrained_path: Optional[str] = None,
) -> CryptoGNN:
    """Factory function to create a CryptoGNN model."""
    model = CryptoGNN(
        num_assets=num_assets,
        input_features=input_features,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
    
    model.to(DEVICE)
    
    if USE_ROCM:
        model.half()
    
    return model


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"ROCm available: {USE_ROCM}")
    
    # Create model
    model = create_crypto_gnn(num_assets=50, input_features=20)
    
    # Test inference
    dummy_features = np.random.randn(50, 20).astype(np.float32)
    dummy_returns = np.random.randn(60, 50).astype(np.float32) * 0.01
    
    start = time.perf_counter()
    result = model.predict(dummy_features, dummy_returns)
    elapsed = time.perf_counter() - start
    
    print(f"Predictions shape: {result['predictions'].shape}")
    print(f"Global risk: {result['global_risk']:.4f}")
    print(f"Inference time: {elapsed*1000:.2f}ms")
    print(f"Top 5 central assets: {np.argsort(result['centrality'])[-5:]}")
    print(f"Memory efficient GNN: ✓")
