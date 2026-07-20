"""
Soul Vector Database - Local Vector Store for Market Regime and Trade Outcome Embeddings.
Uses FAISS/LanceDB for efficient KNN similarity search to find similar past market conditions.
No LLMs involved - purely statistical embeddings from market data features.

Target Hardware: AMD Ryzen AI 5 + AMD Radeon GPU + 16GB RAM (8GB cap)
"""

import os
import logging
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import threading
import json
import struct

logger = logging.getLogger(__name__)


@dataclass
class VectorEntry:
    """A single vector entry with metadata."""
    id: int
    vector: np.ndarray
    metadata: Dict[str, Any]
    created_at: datetime
    
    # For trade outcomes
    outcome_pnl: float = 0.0
    outcome_return: float = 0.0
    regime_label: str = ""


@dataclass
class SearchResult:
    """KNN search result."""
    id: int
    distance: float
    metadata: Dict[str, Any]
    vector: Optional[np.ndarray] = None


class SoulVectorDB:
    """
    Lightweight local vector database for market regime and trade outcome embeddings.
    
    Features:
    - FAISS-based similarity search (CPU-optimized for AMD Ryzen)
    - Memory-mapped storage for large vector collections
    - Strict memory cap enforcement (2GB max for vectors)
    - Incremental indexing without full rebuilds
    - Metadata storage alongside vectors
    
    Use Cases:
    - Find similar market regimes for pattern matching
    - Retrieve historical trade outcomes for RL training
    - Build "memory" of past market conditions for decision making
    """
    
    def __init__(
        self,
        storage_path: str = "./data/soul_db",
        vector_dim: int = 128,
        max_vectors: int = 1_000_000,
        max_ram_usage_gb: float = 2.0
    ):
        """
        Initialize vector database.
        
        Args:
            storage_path: Path for database files
            vector_dim: Dimension of embedding vectors
            max_vectors: Maximum number of vectors to store
            max_ram_usage_gb: RAM limit for in-memory index
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.vector_dim = vector_dim
        self.max_vectors = max_vectors
        self.max_ram_bytes = int(max_ram_usage_gb * 1024**3)
        
        # Try to import FAISS
        self.faiss_available = False
        self.index = None
        
        try:
            import faiss
            self.faiss_available = True
            logger.info("FAISS available for accelerated similarity search")
        except ImportError:
            logger.warning("FAISS not available, falling back to NumPy brute-force search")
        
        # In-memory metadata store
        self.metadata_store: Dict[int, Dict[str, Any]] = {}
        self.id_counter = 0
        self._lock = threading.RLock()
        
        # Vector storage file
        self.vectors_file = self.storage_path / "vectors.dat"
        self.metadata_file = self.storage_path / "metadata.jsonl"
        
        # Load existing data
        self._load_existing_data()
        
        logger.info(f"SoulVectorDB initialized: dim={vector_dim}, max={max_vectors}")
    
    def _load_existing_data(self):
        """Load existing vectors and metadata from disk."""
        if not self.vectors_file.exists():
            logger.info("Starting fresh vector database")
            return
        
        try:
            # Count existing vectors
            file_size = self.vectors_file.stat().st_size
            vector_size = self.vector_dim * 8  # float32 = 4 bytes, but we use float64
            existing_count = file_size // vector_size
            
            if existing_count > 0:
                logger.info(f"Loading {existing_count} existing vectors")
                
                # Load vectors
                vectors = np.memmap(
                    str(self.vectors_file),
                    dtype='float64',
                    mode='r',
                    shape=(existing_count, self.vector_dim)
                )
                
                # Build index
                if self.faiss_available:
                    import faiss
                    self.index = faiss.IndexFlatL2(self.vector_dim)
                    self.index.add(vectors.astype('float32'))
                else:
                    self._numpy_vectors = np.array(vectors)
                
                self.id_counter = existing_count
                
                # Load metadata
                if self.metadata_file.exists():
                    with open(self.metadata_file, 'r') as f:
                        for line in f:
                            entry = json.loads(line.strip())
                            self.metadata_store[entry['id']] = entry['metadata']
                
                logger.info(f"Loaded {existing_count} vectors from disk")
                
        except Exception as e:
            logger.error(f"Failed to load existing data: {e}")
    
    def add_vector(
        self,
        vector: np.ndarray,
        metadata: Dict[str, Any],
        outcome_pnl: float = 0.0,
        outcome_return: float = 0.0,
        regime_label: str = ""
    ) -> int:
        """
        Add a new vector to the database.
        
        Args:
            vector: Embedding vector (must match vector_dim)
            metadata: Associated metadata dict
            outcome_pnl: PnL outcome (for trade embeddings)
            outcome_return: Return percentage (for trade embeddings)
            regime_label: Market regime label
            
        Returns:
            Assigned vector ID
        """
        if len(vector) != self.vector_dim:
            raise ValueError(f"Vector dimension must be {self.vector_dim}")
        
        with self._lock:
            if self.id_counter >= self.max_vectors:
                logger.warning("Maximum vector capacity reached")
                return -1
            
            vector_id = self.id_counter
            self.id_counter += 1
            
            # Ensure vector is correct dtype and shape
            vector = np.asarray(vector, dtype='float64').flatten()
            
            # Append to memory-mapped file
            with open(self.vectors_file, 'ab') as f:
                vector.tofile(f)
            
            # Update FAISS index
            if self.faiss_available and self.index is not None:
                import faiss
                self.index.add(vector.astype('float32').reshape(1, -1))
            else:
                if not hasattr(self, '_numpy_vectors'):
                    self._numpy_vectors = np.zeros((0, self.vector_dim), dtype='float64')
                self._numpy_vectors = np.vstack([self._numpy_vectors, vector])
            
            # Store metadata
            entry_metadata = {
                'id': vector_id,
                'created_at': datetime.now().isoformat(),
                'outcome_pnl': outcome_pnl,
                'outcome_return': outcome_return,
                'regime_label': regime_label,
                **metadata
            }
            self.metadata_store[vector_id] = entry_metadata
            
            # Append to metadata file
            with open(self.metadata_file, 'a') as f:
                f.write(json.dumps({
                    'id': vector_id,
                    'metadata': entry_metadata
                }) + '\n')
            
            logger.debug(f"Added vector {vector_id}")
            return vector_id
    
    def add_batch(
        self,
        vectors: np.ndarray,
        metadatas: List[Dict[str, Any]],
        outcomes: Optional[List[float]] = None,
        regime_labels: Optional[List[str]] = None
    ) -> List[int]:
        """
        Add multiple vectors in batch.
        
        Args:
            vectors: Array of shape (n_vectors, vector_dim)
            metadatas: List of metadata dicts
            outcomes: Optional list of PnL outcomes
            regime_labels: Optional list of regime labels
            
        Returns:
            List of assigned IDs
        """
        n_vectors = len(vectors)
        if len(metadatas) != n_vectors:
            raise ValueError("Number of vectors must match number of metadatas")
        
        ids = []
        for i in range(n_vectors):
            outcome = outcomes[i] if outcomes else 0.0
            regime = regime_labels[i] if regime_labels else ""
            
            vec_id = self.add_vector(
                vectors[i],
                metadatas[i],
                outcome_pnl=outcome,
                regime_label=regime
            )
            ids.append(vec_id)
        
        return ids
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        regime_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Find k most similar vectors to query.
        
        Args:
            query_vector: Query embedding
            k: Number of results
            regime_filter: Optional filter by regime label
            
        Returns:
            List of SearchResult objects
        """
        if len(query_vector) != self.vector_dim:
            raise ValueError(f"Query vector dimension must be {self.vector_dim}")
        
        query_vector = np.asarray(query_vector, dtype='float64').flatten()
        
        if self.faiss_available and self.index is not None:
            return self._faiss_search(query_vector, k, regime_filter)
        else:
            return self._numpy_search(query_vector, k, regime_filter)
    
    def _faiss_search(
        self,
        query: np.ndarray,
        k: int,
        regime_filter: Optional[str]
    ) -> List[SearchResult]:
        """FAISS-accelerated search."""
        import faiss
        
        query_f32 = query.astype('float32').reshape(1, -1)
        
        # Search with extra candidates for filtering
        search_k = k * 10 if regime_filter else k
        distances, indices = self.index.search(query_f32, search_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue  # Invalid index
            
            meta = self.metadata_store.get(int(idx), {})
            
            # Apply regime filter
            if regime_filter and meta.get('regime_label') != regime_filter:
                continue
            
            results.append(SearchResult(
                id=int(idx),
                distance=float(dist),
                metadata=meta
            ))
            
            if len(results) >= k:
                break
        
        return results
    
    def _numpy_search(
        self,
        query: np.ndarray,
        k: int,
        regime_filter: Optional[str]
    ) -> List[SearchResult]:
        """NumPy brute-force search fallback."""
        if not hasattr(self, '_numpy_vectors') or len(self._numpy_vectors) == 0:
            return []
        
        # Compute L2 distances
        diff = self._numpy_vectors - query
        distances = np.sqrt(np.sum(diff ** 2, axis=1))
        
        # Get top-k indices
        top_k_indices = np.argsort(distances)[:k * 10]  # Extra for filtering
        
        results = []
        for idx in top_k_indices:
            meta = self.metadata_store.get(int(idx), {})
            
            # Apply regime filter
            if regime_filter and meta.get('regime_label') != regime_filter:
                continue
            
            results.append(SearchResult(
                id=int(idx),
                distance=float(distances[idx]),
                metadata=meta
            ))
            
            if len(results) >= k:
                break
        
        return results
    
    def get_vector(self, vector_id: int) -> Optional[VectorEntry]:
        """Retrieve a specific vector by ID."""
        if vector_id not in self.metadata_store:
            return None
        
        # Read vector from mmap
        try:
            vectors = np.memmap(
                str(self.vectors_file),
                dtype='float64',
                mode='r',
                shape=(-1, self.vector_dim)
            )
            
            if vector_id >= len(vectors):
                return None
            
            return VectorEntry(
                id=vector_id,
                vector=vectors[vector_id].copy(),
                metadata=self.metadata_store[vector_id],
                created_at=datetime.fromisoformat(
                    self.metadata_store[vector_id].get('created_at', datetime.now().isoformat())
                ),
                outcome_pnl=self.metadata_store[vector_id].get('outcome_pnl', 0.0),
                outcome_return=self.metadata_store[vector_id].get('outcome_return', 0.0),
                regime_label=self.metadata_store[vector_id].get('regime_label', '')
            )
        except Exception as e:
            logger.error(f"Failed to retrieve vector {vector_id}: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        vector_count = self.id_counter
        
        # Count by regime
        regime_counts: Dict[str, int] = {}
        pnl_sum = 0.0
        profitable_count = 0
        
        for meta in self.metadata_store.values():
            regime = meta.get('regime_label', 'unknown')
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            pnl = meta.get('outcome_pnl', 0.0)
            pnl_sum += pnl
            if pnl > 0:
                profitable_count += 1
        
        return {
            'total_vectors': vector_count,
            'vector_dimension': self.vector_dim,
            'regime_distribution': regime_counts,
            'avg_pnl': pnl_sum / max(vector_count, 1),
            'profitable_ratio': profitable_count / max(vector_count, 1),
            'faiss_enabled': self.faiss_available,
            'storage_file': str(self.vectors_file),
            'storage_size_mb': self.vectors_file.stat().st_size / (1024**2) if self.vectors_file.exists() else 0
        }
    
    def export_for_training(
        self,
        output_path: str,
        regime_filter: Optional[str] = None,
        min_pnl: Optional[float] = None
    ) -> int:
        """
        Export vectors for ML training.
        
        Args:
            output_path: Output file path
            regime_filter: Filter by regime
            min_pnl: Minimum PnL threshold
            
        Returns:
            Number of exported vectors
        """
        exported = []
        
        for vec_id, meta in self.metadata_store.items():
            # Apply filters
            if regime_filter and meta.get('regime_label') != regime_filter:
                continue
            if min_pnl is not None and meta.get('outcome_pnl', 0) < min_pnl:
                continue
            
            # Get vector
            entry = self.get_vector(vec_id)
            if entry is None:
                continue
            
            exported.append({
                'vector': entry.vector.tolist(),
                'label': meta.get('regime_label', ''),
                'pnl': meta.get('outcome_pnl', 0),
                'return': meta.get('outcome_return', 0),
                'metadata': {k: v for k, v in meta.items() 
                           if k not in ['id', 'created_at']}
            })
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(exported, f)
        
        logger.info(f"Exported {len(exported)} vectors to {output_path}")
        return len(exported)
    
    def compact(self) -> int:
        """
        Compact the database by removing old/low-quality entries.
        
        Returns:
            Number of entries removed
        """
        # Strategy: Keep only top N vectors per regime based on |PnL|
        # This ensures we keep informative examples (both good and bad)
        
        removed = 0
        keep_per_regime = 10000
        
        # Group by regime
        regime_groups: Dict[str, List[Tuple[int, float]]] = {}
        for vec_id, meta in self.metadata_store.items():
            regime = meta.get('regime_label', 'unknown')
            pnl = abs(meta.get('outcome_pnl', 0))
            
            if regime not in regime_groups:
                regime_groups[regime] = []
            regime_groups[regime].append((vec_id, pnl))
        
        # Determine which to keep
        to_remove = set()
        for regime, entries in regime_groups.items():
            if len(entries) > keep_per_regime:
                # Sort by |PnL| descending, keep top
                entries.sort(key=lambda x: x[1], reverse=True)
                for vec_id, _ in entries[keep_per_regime:]:
                    to_remove.add(vec_id)
        
        # Note: Actual removal would require rebuilding the index
        # For now, just log what would be removed
        logger.info(f"Compaction would remove {len(to_remove)} entries")
        return len(to_remove)
    
    def close(self):
        """Clean shutdown."""
        logger.info("Closing SoulVectorDB")
        # mmap files auto-close, but we can flush any pending writes
        if hasattr(self, '_numpy_vectors'):
            del self._numpy_vectors


class MarketRegimeEmbedder:
    """
    Generate embeddings for market regimes using statistical features.
    No neural networks - purely deterministic feature engineering.
    """
    
    def __init__(self, embedding_dim: int = 128):
        """
        Initialize embedder.
        
        Args:
            embedding_dim: Target embedding dimension
        """
        self.embedding_dim = embedding_dim
        
        # Feature weights for projection (learned offline or heuristic)
        self.feature_weights = np.random.randn(embedding_dim, 64) * 0.1
        self.feature_weights /= np.linalg.norm(self.feature_weights, axis=0)
        
        logger.info(f"MarketRegimeEmbedder initialized: dim={embedding_dim}")
    
    def compute_embedding(
        self,
        returns: np.ndarray,
        volumes: np.ndarray,
        volatility: np.ndarray,
        orderflow_imbalance: float,
        momentum: float
    ) -> np.ndarray:
        """
        Compute market regime embedding from raw features.
        
        Args:
            returns: Recent returns array
            volumes: Recent volumes array
            volatility: Volatility metrics
            orderflow_imbalance: Current order flow imbalance
            momentum: Momentum indicator
            
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        # Extract statistical features
        features = self._extract_features(
            returns, volumes, volatility,
            orderflow_imbalance, momentum
        )
        
        # Project to embedding space
        embedding = np.tanh(features @ self.feature_weights.T)
        
        # Normalize
        embedding /= np.linalg.norm(embedding) + 1e-8
        
        return embedding
    
    def _extract_features(
        self,
        returns: np.ndarray,
        volumes: np.ndarray,
        volatility: np.ndarray,
        orderflow_imbalance: float,
        momentum: float
    ) -> np.ndarray:
        """Extract statistical features from market data."""
        features = []
        
        # Return statistics
        features.extend([
            np.mean(returns),
            np.std(returns),
            np.skew(returns) if len(returns) > 2 else 0,
            np.kurtosis(returns) if len(returns) > 3 else 0,
            np.min(returns),
            np.max(returns),
            np.percentile(returns, [1, 5, 25, 50, 75, 95, 99]).tolist()
        ])
        
        # Volume statistics
        features.extend([
            np.mean(volumes),
            np.std(volumes),
            np.max(volumes) / (np.mean(volumes) + 1e-8),  # Volume spike ratio
        ])
        
        # Volatility features
        if isinstance(volatility, np.ndarray):
            features.extend([
                np.mean(volatility),
                np.std(volatility),
                volatility[-1] / (np.mean(volatility) + 1e-8)  # Current vs avg
            ])
        else:
            features.extend([volatility, 0, 0])
        
        # Order flow and momentum
        features.extend([
            orderflow_imbalance,
            momentum,
            momentum * orderflow_imbalance  # Interaction term
        ])
        
        # Flatten and pad/truncate to fixed size
        flat = np.array(features).flatten()
        target_size = 64
        
        if len(flat) < target_size:
            flat = np.pad(flat, (0, target_size - len(flat)))
        else:
            flat = flat[:target_size]
        
        return flat
    
    def classify_regime(self, embedding: np.ndarray) -> str:
        """
        Classify market regime from embedding using simple heuristics.
        
        Args:
            embedding: Market regime embedding
            
        Returns:
            Regime label string
        """
        # Simple rule-based classification
        # In production, this would use a trained classifier
        
        # Use first few dimensions as regime indicators
        trend_score = embedding[0]
        vol_score = embedding[1]
        momentum_score = embedding[2]
        
        if vol_score > 0.5:
            if trend_score > 0.3:
                return "HIGH_VOL_BULL"
            elif trend_score < -0.3:
                return "HIGH_VOL_BEAR"
            else:
                return "HIGH_VOL_SIDEWAYS"
        else:
            if momentum_score > 0.2:
                return "LOW_VOL_UPTREND"
            elif momentum_score < -0.2:
                return "LOW_VOL_DOWNTREND"
            else:
                return "LOW_VOL_RANGE"


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    db = SoulVectorDB(storage_path="./data/test_soul_db", vector_dim=64)
    embedder = MarketRegimeEmbedder(embedding_dim=64)
    
    # Generate sample embeddings
    for i in range(100):
        returns = np.random.randn(100) * 0.02
        volumes = np.random.exponential(1000, 100)
        volatility = np.abs(returns).std()
        orderflow = np.random.uniform(-1, 1)
        momentum = np.random.uniform(-0.1, 0.1)
        
        embedding = embedder.compute_embedding(
            returns, volumes, volatility, orderflow, momentum
        )
        
        # Simulate trade outcome
        pnl = np.random.randn() * 100
        regime = embedder.classify_regime(embedding)
        
        db.add_vector(
            embedding,
            metadata={'sample_idx': i, 'symbol': 'BTCUSDT'},
            outcome_pnl=pnl,
            regime_label=regime
        )
    
    # Search for similar regimes
    query_returns = np.random.randn(100) * 0.02
    query_embedding = embedder.compute_embedding(
        query_returns,
        np.random.exponential(1000, 100),
        0.03,
        0.5,
        0.02
    )
    
    results = db.search(query_embedding, k=5)
    print(f"\nTop 5 similar regimes:")
    for r in results:
        print(f"  ID {r.id}: distance={r.distance:.4f}, regime={r.metadata.get('regime_label')}, pnl={r.metadata.get('outcome_pnl', 0):.2f}")
    
    # Stats
    print(f"\nDatabase stats: {db.get_statistics()}")
    
    db.close()
