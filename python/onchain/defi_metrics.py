"""
DeFi Metrics Tracker for On-Chain Analytics
Tracks TVL, staking activity, and gas usage across L1/L2 networks
Aggregates data into optimized time-series arrays for feature vectors
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, defaultdict
from dataclasses import dataclass, field
import numpy as np
import time
import struct


@dataclass
class TVLMetrics:
    """TVL metrics for a protocol or chain."""
    protocol_name: str
    chain: str
    timestamp: float
    tvl_usd: float
    tvl_change_24h: float
    volume_24h: float
    unique_users_24h: int
    transactions_24h: int


@dataclass
class StakingMetrics:
    """Staking activity metrics."""
    protocol_name: str
    chain: str
    timestamp: float
    total_staked: float
    staking_ratio: float  # % of supply staked
    apr: float
    validators_count: int
    delegation_changes_24h: float


@dataclass
class GasMetrics:
    """Gas usage metrics."""
    chain: str
    timestamp: float
    avg_gas_price_gwei: float
    median_gas_price_gwei: float
    gas_used: int
    gas_limit: int
    utilization_rate: float
    base_fee: Optional[float] = None  # EIP-1559 base fee


class TimeSeriesBuffer:
    """
    Memory-efficient circular buffer for time-series data.
    Uses NumPy arrays for optimal performance.
    """
    
    def __init__(self, capacity: int = 10000, dtype: np.dtype = np.float32):
        self.capacity = capacity
        self.dtype = dtype
        self.buffer = np.zeros(capacity, dtype=dtype)
        self.timestamps = np.zeros(capacity, dtype=np.float64)
        self.index = 0
        self.count = 0
        self.lock = asyncio.Lock()
    
    async def append(self, value: float, timestamp: float):
        """Append value to buffer in a thread-safe manner."""
        async with self.lock:
            self.buffer[self.index] = value
            self.timestamps[self.index] = timestamp
            self.index = (self.index + 1) % self.capacity
            self.count = min(self.count + 1, self.capacity)
    
    def get_recent(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get n most recent values and timestamps."""
        if self.count == 0:
            return np.array([], dtype=self.dtype), np.array([], dtype=np.float64)
        
        n = min(n, self.count)
        
        if self.index >= n:
            start_idx = self.index - n
            values = self.buffer[start_idx:self.index]
            timestamps = self.timestamps[start_idx:self.index]
        else:
            # Wrap around case
            values = np.concatenate([
                self.buffer[self.index - n:],
                self.buffer[:self.index]
            ])
            timestamps = np.concatenate([
                self.timestamps[self.index - n:],
                self.timestamps[:self.index]
            ])
        
        return values, timestamps
    
    def get_statistics(self, window: int = 100) -> Dict[str, float]:
        """Calculate statistics over recent window."""
        if self.count < window:
            window = self.count
        
        if window == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'percentile_90': 0.0,
                'percentile_99': 0.0
            }
        
        values, _ = self.get_recent(window)
        
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'percentile_90': float(np.percentile(values, 90)),
            'percentile_99': float(np.percentile(values, 99)),
            'skewness': float(self._calculate_skewness(values)),
            'kurtosis': float(self._calculate_kurtosis(values))
        }
    
    @staticmethod
    def _calculate_skewness(data: np.ndarray) -> float:
        """Calculate sample skewness."""
        n = len(data)
        if n < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        return float((n / ((n-1) * (n-2))) * np.sum(((data - mean) / std) ** 3))
    
    @staticmethod
    def _calculate_kurtosis(data: np.ndarray) -> float:
        """Calculate excess kurtosis."""
        n = len(data)
        if n < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        m4 = np.mean((data - mean) ** 4)
        m2 = np.mean((data - mean) ** 2)
        if m2 == 0:
            return 0.0
        return float((m4 / (m2 ** 2)) - 3)
    
    def to_feature_vector(self, windows: List[int] = [10, 50, 100, 500]) -> np.ndarray:
        """
        Convert buffer to feature vector for ML models.
        Includes multi-scale statistics.
        """
        features = []
        
        for window in windows:
            stats = self.get_statistics(window)
            features.extend([
                stats['mean'],
                stats['std'],
                stats['min'],
                stats['max'],
                stats['median'],
                stats['percentile_90'],
                stats['percentile_99'],
                stats['skewness'],
                stats['kurtosis']
            ])
        
        # Add trend features
        recent_100, _ = self.get_recent(100)
        if len(recent_100) > 10:
            # Simple linear regression slope
            x = np.arange(len(recent_100))
            slope = np.polyfit(x, recent_100, 1)[0]
            features.append(float(slope))
            
            # Momentum (rate of change)
            momentum = (recent_100[-1] - recent_100[0]) / recent_100[0] if recent_100[0] != 0 else 0
            features.append(float(momentum))
        else:
            features.extend([0.0, 0.0])
        
        return np.array(features, dtype=np.float32)


class DefiMetricsTracker:
    """
    Main DeFi metrics tracker with async data ingestion.
    Collects TVL, staking, and gas metrics across chains.
    """
    
    # Known DeFi protocols per chain
    PROTOCOLS = {
        'ethereum': [
            'aave', 'compound', 'uniswap', 'curve', 'lido',
            'makerdao', 'convex', 'rocket_pool', 'frax'
        ],
        'bsc': [
            'pancakeswap', 'venus', 'alpaca', 'biswap'
        ],
        'arbitrum': [
            'gmx', 'camelot', 'radiant', 'pendle'
        ],
        'optimism': [
            'velodrome', 'synthetix', 'aave'
        ]
    }
    
    def __init__(
        self,
        chains: Optional[List[str]] = None,
        buffer_capacity: int = 10000,
        update_interval: float = 60.0
    ):
        self.chains = chains or ['ethereum', 'bsc', 'arbitrum', 'optimism']
        self.buffer_capacity = buffer_capacity
        self.update_interval = update_interval
        
        # Time-series buffers per metric type per chain
        self.tvl_buffers: Dict[str, Dict[str, TimeSeriesBuffer]] = defaultdict(dict)
        self.staking_buffers: Dict[str, Dict[str, TimeSeriesBuffer]] = defaultdict(dict)
        self.gas_buffers: Dict[str, TimeSeriesBuffer] = {}
        
        # Initialize buffers
        for chain in self.chains:
            self.gas_buffers[chain] = TimeSeriesBuffer(buffer_capacity)
            for protocol in self.PROTOCOLS.get(chain, []):
                self.tvl_buffers[chain][protocol] = TimeSeriesBuffer(buffer_capacity)
                self.staking_buffers[chain][protocol] = TimeSeriesBuffer(buffer_capacity)
        
        # Latest metrics cache
        self.latest_tvl: Dict[str, Dict[str, TVLMetrics]] = defaultdict(dict)
        self.latest_staking: Dict[str, Dict[str, StakingMetrics]] = defaultdict(dict)
        self.latest_gas: Dict[str, GasMetrics] = {}
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Running state
        self.running = False
        self.tasks: List[asyncio.Task] = []
        
        # Statistics
        self.stats = {
            'updates_completed': 0,
            'errors': 0,
            'last_update': 0.0
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close(self):
        """Cleanup resources."""
        self.running = False
        
        for task in self.tasks:
            task.cancel()
        
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def fetch_tvl_data(self, chain: str, protocol: str) -> Optional[TVLMetrics]:
        """
        Fetch TVL data for a protocol.
        In production, this would call actual APIs (DefiLlama, etc.)
        """
        try:
            session = await self._get_session()
            
            # Simulated API call - replace with actual DefiLlama API
            # url = f"https://api.llama.fi/tvl/{protocol}"
            # async with session.get(url) as response:
            #     data = await response.json()
            
            # Simulated data for demonstration
            base_tvl = np.random.uniform(1e8, 1e10)  # $100M to $10B
            change_24h = np.random.uniform(-0.1, 0.1)
            
            return TVLMetrics(
                protocol_name=protocol,
                chain=chain,
                timestamp=time.time(),
                tvl_usd=base_tvl,
                tvl_change_24h=change_24h,
                volume_24h=base_tvl * np.random.uniform(0.01, 0.1),
                unique_users_24h=int(np.random.uniform(1000, 100000)),
                transactions_24h=int(np.random.uniform(5000, 500000))
            )
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"[{chain}/{protocol}] TVL fetch error: {e}")
            return None
    
    async def fetch_staking_data(self, chain: str, protocol: str) -> Optional[StakingMetrics]:
        """
        Fetch staking metrics for a protocol.
        In production, this would call protocol-specific APIs.
        """
        try:
            # Simulated staking data
            total_staked = np.random.uniform(1e7, 1e9)
            
            return StakingMetrics(
                protocol_name=protocol,
                chain=chain,
                timestamp=time.time(),
                total_staked=total_staked,
                staking_ratio=np.random.uniform(0.3, 0.8),
                apr=np.random.uniform(0.02, 0.15),
                validators_count=int(np.random.uniform(50, 500)),
                delegation_changes_24h=np.random.uniform(-0.05, 0.05)
            )
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"[{chain}/{protocol}] Staking fetch error: {e}")
            return None
    
    async def fetch_gas_data(self, chain: str) -> Optional[GasMetrics]:
        """
        Fetch gas metrics for a chain.
        In production, this would call RPC nodes.
        """
        try:
            session = await self._get_session()
            
            # Simulated gas data
            avg_gas = np.random.uniform(10, 200)  # gwei
            
            return GasMetrics(
                chain=chain,
                timestamp=time.time(),
                avg_gas_price_gwei=avg_gas,
                median_gas_price_gwei=avg_gas * np.random.uniform(0.8, 1.0),
                gas_used=int(np.random.uniform(10e6, 30e6)),
                gas_limit=int(np.random.uniform(30e6, 50e6)),
                utilization_rate=np.random.uniform(0.3, 0.9),
                base_fee=avg_gas * 0.8 if chain == 'ethereum' else None
            )
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"[{chain}] Gas fetch error: {e}")
            return None
    
    async def update_tvl_metrics(self, chain: str, protocol: str):
        """Update TVL metrics and store in buffer."""
        metrics = await self.fetch_tvl_data(chain, protocol)
        
        if metrics:
            self.latest_tvl[chain][protocol] = metrics
            
            # Store in buffer
            buffer = self.tvl_buffers[chain][protocol]
            await buffer.append(metrics.tvl_usd, metrics.timestamp)
    
    async def update_staking_metrics(self, chain: str, protocol: str):
        """Update staking metrics and store in buffer."""
        metrics = await self.fetch_staking_data(chain, protocol)
        
        if metrics:
            self.latest_staking[chain][protocol] = metrics
            
            # Store in buffer
            buffer = self.staking_buffers[chain][protocol]
            await buffer.append(metrics.total_staked, metrics.timestamp)
    
    async def update_gas_metrics(self, chain: str):
        """Update gas metrics and store in buffer."""
        metrics = await self.fetch_gas_data(chain)
        
        if metrics:
            self.latest_gas[chain] = metrics
            
            # Store in buffer
            buffer = self.gas_buffers[chain]
            await buffer.append(metrics.avg_gas_price_gwei, metrics.timestamp)
    
    async def update_all_metrics(self):
        """Update all metrics for all chains and protocols."""
        tasks = []
        
        # TVL updates
        for chain in self.chains:
            for protocol in self.PROTOCOLS.get(chain, []):
                tasks.append(self.update_tvl_metrics(chain, protocol))
        
        # Staking updates
        for chain in self.chains:
            for protocol in self.PROTOCOLS.get(chain, []):
                tasks.append(self.update_staking_metrics(chain, protocol))
        
        # Gas updates
        for chain in self.chains:
            tasks.append(self.update_gas_metrics(chain))
        
        # Execute all updates concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.stats['updates_completed'] += 1
        self.stats['last_update'] = time.time()
    
    async def run_update_loop(self):
        """Continuous update loop."""
        while self.running:
            try:
                await self.update_all_metrics()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Update loop error: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def start(self):
        """Start the metrics tracker."""
        self.running = True
        
        # Initial update
        await self.update_all_metrics()
        
        # Start continuous update loop
        task = asyncio.create_task(self.run_update_loop())
        self.tasks.append(task)
        
        print(f"DeFi metrics tracker started for {len(self.chains)} chains")
    
    def get_tvl_features(self, chain: str, protocol: str) -> np.ndarray:
        """Get TVL feature vector for ML model."""
        if chain not in self.tvl_buffers or protocol not in self.tvl_buffers[chain]:
            return np.zeros(20, dtype=np.float32)
        
        return self.tvl_buffers[chain][protocol].to_feature_vector()
    
    def get_staking_features(self, chain: str, protocol: str) -> np.ndarray:
        """Get staking feature vector for ML model."""
        if chain not in self.staking_buffers or protocol not in self.staking_buffers[chain]:
            return np.zeros(20, dtype=np.float32)
        
        return self.staking_buffers[chain][protocol].to_feature_vector()
    
    def get_gas_features(self, chain: str) -> np.ndarray:
        """Get gas feature vector for ML model."""
        if chain not in self.gas_buffers:
            return np.zeros(20, dtype=np.float32)
        
        return self.gas_buffers[chain].to_feature_vector()
    
    def get_aggregated_features(self, chain: str) -> np.ndarray:
        """
        Get aggregated feature vector combining all metrics for a chain.
        This is the primary input for ML models.
        """
        features = []
        
        # Gas features
        gas_features = self.get_gas_features(chain)
        features.extend(gas_features)
        
        # Aggregate TVL features across protocols
        tvl_values = []
        for protocol in self.PROTOCOLS.get(chain, []):
            if chain in self.tvl_buffers and protocol in self.tvl_buffers[chain]:
                buffer = self.tvl_buffers[chain][protocol]
                values, _ = buffer.get_recent(100)
                if len(values) > 0:
                    tvl_values.append(np.mean(values))
        
        if tvl_values:
            features.extend([
                np.mean(tvl_values),
                np.std(tvl_values),
                np.sum(tvl_values),  # Total TVL
                len(tvl_values)  # Active protocols count
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Aggregate staking features
        staking_ratios = []
        for protocol in self.PROTOCOLS.get(chain, []):
            if protocol in self.latest_staking.get(chain, {}):
                staking_ratios.append(self.latest_staking[chain][protocol].staking_ratio)
        
        if staking_ratios:
            features.extend([
                np.mean(staking_ratios),
                np.std(staking_ratios),
                np.max(staking_ratios)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of tracked metrics."""
        summary = {
            'chains': self.chains,
            'stats': self.stats,
            'tvl_protocols': sum(len(p) for p in self.tvl_buffers.values()),
            'gas_chains': len(self.gas_buffers)
        }
        
        # Add latest values
        summary['latest_gas'] = {
            chain: metrics.avg_gas_price_gwei 
            for chain, metrics in self.latest_gas.items()
        }
        
        return summary


# Example usage
async def main():
    tracker = DefiMetricsTracker(
        chains=['ethereum', 'arbitrum'],
        update_interval=30.0
    )
    
    await tracker.start()
    
    # Let it run for a bit
    await asyncio.sleep(60)
    
    # Get features
    eth_features = tracker.get_aggregated_features('ethereum')
    print(f"Ethereum feature vector shape: {eth_features.shape}")
    print(f"Feature vector: {eth_features}")
    
    # Get summary
    print("\nSummary:", tracker.get_summary())
    
    await tracker.close()


if __name__ == '__main__':
    asyncio.run(main())
