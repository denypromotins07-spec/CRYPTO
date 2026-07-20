"""
Collateral Optimizer Module - Ray-based Background Worker

Ray-based background worker that periodically rebalances collateral between
spot and futures wallets to maintain optimal margin ratios, using pre-allocated
memory pools to ensure it never exceeds its memory budget.

Hardware Target: AMD Ryzen AI 5 with Ray distributed computing
Memory Constraint: Strict 8GB global limit, pre-allocated memory pools
"""

import ray
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import time
import logging
import numpy as np
from enum import Enum


# Configure Ray for memory efficiency
ray.init(
    object_store_memory=2 * 1024 * 1024 * 1024,  # 2GB object store
    _system_config={"max_object_store_size": 2 * 1024 * 1024 * 1024}
)


class RebalanceAction(Enum):
    """Types of rebalancing actions"""
    SPOT_TO_FUTURES = "spot_to_futures"
    FUTURES_TO_SPOT = "futures_to_spot"
    ASSET_SWAP = "asset_swap"
    NO_ACTION = "no_action"


@dataclass
class WalletBalance:
    """Balance information for a wallet"""
    wallet_type: str  # 'spot' or 'futures'
    asset: str
    free_balance: float
    locked_balance: float
    total_balance: float
    usd_value: float
    last_updated_ns: int = field(default_factory=lambda: time.time_ns())


@dataclass
class MarginState:
    """Current margin state of the portfolio"""
    total_collateral_usd: float
    used_margin_usd: float
    available_margin_usd: float
    margin_ratio: float
    maintenance_margin_usd: float
    liquidation_risk: float  # 0.0 to 1.0
    timestamp_ns: int = field(default_factory=lambda: time.time_ns())


@dataclass
class RebalanceRecommendation:
    """Recommended rebalancing action"""
    action: RebalanceAction
    from_wallet: str
    to_wallet: str
    asset: str
    amount: float
    expected_improvement: float
    priority: int
    reason: str
    created_at_ns: int = field(default_factory=lambda: time.time_ns())


@ray.remote(max_calls=1000)  # Restart actor after 1000 calls to prevent memory leaks
class CollateralOptimizerWorker:
    """
    Ray worker for collateral optimization calculations.
    Uses pre-allocated memory pools to stay within budget.
    """
    
    def __init__(self, worker_id: int, memory_pool_size: int = 1000):
        self.worker_id = worker_id
        self.memory_pool_size = memory_pool_size
        
        # Pre-allocated memory pools (circular buffers)
        self._balance_history: deque[Dict[str, float]] = deque(maxlen=memory_pool_size)
        self._margin_history: deque[float] = deque(maxlen=memory_pool_size)
        self._recommendations: deque[RebalanceRecommendation] = deque(maxlen=100)
        
        # Statistics
        self.optimizations_run = 0
        self.recommendations_made = 0
        
        # Logger
        self.logger = logging.getLogger(f"OptimizerWorker-{worker_id}")
    
    def analyze_margin_state(
        self,
        spot_balances: List[WalletBalance],
        futures_balances: List[WalletBalance],
        margin_state: MarginState,
    ) -> Dict[str, Any]:
        """Analyze current margin state and return metrics"""
        start_ns = time.time_ns()
        
        # Calculate key metrics
        collateral_utilization = margin_state.used_margin_usd / max(1, margin_state.total_collateral_usd)
        margin_headroom = margin_state.available_margin_usd / max(1, margin_state.total_collateral_usd)
        
        # Record history
        self._margin_history.append(margin_state.margin_ratio)
        self._balance_history.append({
            'spot_total': sum(b.usd_value for b in spot_balances),
            'futures_total': sum(b.usd_value for b in futures_balances),
            'utilization': collateral_utilization,
        })
        
        processing_time_ns = time.time_ns() - start_ns
        self.optimizations_run += 1
        
        return {
            'collateral_utilization': collateral_utilization,
            'margin_headroom': margin_headroom,
            'avg_margin_ratio': np.mean(list(self._margin_history)) if self._margin_history else 0,
            'margin_trend': self._calculate_trend(),
            'processing_time_ns': processing_time_ns,
            'worker_id': self.worker_id,
        }
    
    def generate_recommendations(
        self,
        spot_balances: List[WalletBalance],
        futures_balances: List[WalletBalance],
        margin_state: MarginState,
        target_margin_ratio: float = 2.0,
        min_buffer_pct: float = 0.2,
    ) -> List[RebalanceRecommendation]:
        """Generate rebalancing recommendations"""
        recommendations = []
        
        # Check if margin ratio is below target
        if margin_state.margin_ratio < target_margin_ratio:
            # Need more collateral in futures
            deficit = (target_margin_ratio * margin_state.used_margin_usd) - margin_state.total_collateral_usd
            
            # Find available assets in spot to transfer
            for balance in spot_balances:
                if balance.free_balance > 0 and balance.usd_value > 0:
                    transfer_amount = min(
                        balance.free_balance,
                        deficit / balance.usd_value
                    )
                    
                    if transfer_amount > 0.01:  # Minimum transfer threshold
                        rec = RebalanceRecommendation(
                            action=RebalanceAction.SPOT_TO_FUTURES,
                            from_wallet='spot',
                            to_wallet='futures',
                            asset=balance.asset,
                            amount=transfer_amount,
                            expected_improvement=transfer_amount * balance.usd_value / max(1, margin_state.total_collateral_usd),
                            priority=1,
                            reason=f"Margin ratio {margin_state.margin_ratio:.2f} below target {target_margin_ratio}",
                        )
                        recommendations.append(rec)
                        self._recommendations.append(rec)
                        self.recommendations_made += 1
                        break
        
        # Check for excess collateral in futures
        elif margin_state.margin_ratio > target_margin_ratio * 1.5:
            excess = margin_state.total_collateral_usd - (target_margin_ratio * margin_state.used_margin_usd)
            
            # Find assets in futures to transfer back to spot
            for balance in futures_balances:
                if balance.free_balance > 0 and balance.usd_value > 0:
                    transfer_amount = min(
                        balance.free_balance,
                        excess * 0.5 / balance.usd_value  # Only move half of excess
                    )
                    
                    if transfer_amount > 0.01:
                        rec = RebalanceRecommendation(
                            action=RebalanceAction.FUTURES_TO_SPOT,
                            from_wallet='futures',
                            to_wallet='spot',
                            asset=balance.asset,
                            amount=transfer_amount,
                            expected_improvement=0.1,  # Improves capital efficiency
                            priority=3,
                            reason=f"Excess collateral: margin ratio {margin_state.margin_ratio:.2f}",
                        )
                        recommendations.append(rec)
                        self._recommendations.append(rec)
                        self.recommendations_made += 1
                        break
        
        # Sort by priority
        recommendations.sort(key=lambda r: r.priority)
        
        return recommendations
    
    def _calculate_trend(self) -> str:
        """Calculate margin ratio trend"""
        if len(self._margin_history) < 5:
            return "insufficient_data"
        
        recent = list(self._margin_history)[-5:]
        avg_recent = sum(recent[-3:]) / 3
        avg_older = sum(recent[:2]) / 2
        
        if avg_recent > avg_older * 1.05:
            return "improving"
        elif avg_recent < avg_older * 0.95:
            return "worsening"
        else:
            return "stable"
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        return {
            'worker_id': self.worker_id,
            'optimizations_run': self.optimizations_run,
            'recommendations_made': self.recommendations_made,
            'history_size': len(self._margin_history),
            'memory_pool_usage': len(self._balance_history) / self.memory_pool_size,
        }


@ray.remote
class CollateralOrchestrator:
    """
    Main orchestrator for collateral optimization.
    Coordinates workers and aggregates results.
    """
    
    def __init__(
        self,
        num_workers: int = 4,
        optimization_interval_s: float = 60.0,
        target_margin_ratio: float = 2.0,
    ):
        self.num_workers = num_workers
        self.optimization_interval_s = optimization_interval_s
        self.target_margin_ratio = target_margin_ratio
        
        # Create worker pool
        self.workers = [
            CollateralOptimizerWorker.remote(i) 
            for i in range(num_workers)
        ]
        
        # State tracking
        self.last_optimization_ns = 0
        self.is_running = False
        
        # Aggregated recommendations
        self.pending_recommendations: List[RebalanceRecommendation] = []
        
        # Statistics
        self.total_optimizations = 0
        self.total_rebalances = 0
        
        self.logger = logging.getLogger("CollateralOrchestrator")
    
    async def start(self):
        """Start the orchestrator"""
        self.is_running = True
        self.logger.info(f"[CollateralOrchestrator] Started with {self.num_workers} workers")
    
    async def stop(self):
        """Stop the orchestrator"""
        self.is_running = False
        
        # Cleanup workers
        for worker in self.workers:
            ray.kill(worker)
        
        self.logger.info("[CollateralOrchestrator] Stopped")
    
    async def run_optimization_cycle(
        self,
        spot_balances: List[WalletBalance],
        futures_balances: List[WalletBalance],
        margin_state: MarginState,
    ) -> List[RebalanceRecommendation]:
        """Run a single optimization cycle across all workers"""
        if not self.is_running:
            return []
        
        now_ns = time.time_ns()
        
        # Check if enough time has passed
        if now_ns - self.last_optimization_ns < self.optimization_interval_s * 1e9:
            return self.pending_recommendations
        
        self.last_optimization_ns = now_ns
        
        # Distribute work across workers
        worker_tasks = []
        for i, worker in enumerate(self.workers):
            task = worker.analyze_margin_state.remote(
                spot_balances,
                futures_balances,
                margin_state,
            )
            worker_tasks.append(task)
        
        # Wait for all workers
        results = await asyncio.gather(*worker_tasks, return_exceptions=True)
        
        # Aggregate results
        all_recommendations = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Worker {i} error: {result}")
                continue
            
            # Generate recommendations from each worker
            recs = await self.workers[i].generate_recommendations.remote(
                spot_balances,
                futures_balances,
                margin_state,
                self.target_margin_ratio,
            )
            all_recommendations.extend(recs)
        
        # Deduplicate and sort recommendations
        seen_assets = set()
        unique_recommendations = []
        for rec in sorted(all_recommendations, key=lambda r: r.priority):
            if rec.asset not in seen_assets:
                unique_recommendations.append(rec)
                seen_assets.add(rec.asset)
        
        self.pending_recommendations = unique_recommendations[:10]  # Limit to top 10
        self.total_optimizations += 1
        
        self.logger.info(
            f"[CollateralOrchestrator] Optimization complete. "
            f"{len(self.pending_recommendations)} recommendations"
        )
        
        return self.pending_recommendations
    
    async def execute_rebalance(self, recommendation: RebalanceRecommendation) -> bool:
        """Execute a rebalancing recommendation"""
        try:
            # In production, this would call exchange APIs
            self.logger.info(
                f"[REBALANCE] {recommendation.action.value}: "
                f"{recommendation.amount:.4f} {recommendation.asset} "
                f"from {recommendation.from_wallet} to {recommendation.to_wallet}"
            )
            
            self.total_rebalances += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Rebalance failed: {e}")
            return False
    
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            'total_optimizations': self.total_optimizations,
            'total_rebalances': self.total_rebalances,
            'pending_recommendations': len(self.pending_recommendations),
            'num_workers': self.num_workers,
            'is_running': self.is_running,
        }


# Helper functions for integration
def create_optimizer_cluster(num_workers: int = 4) -> Tuple[ray.ObjectRef, ray.ObjectRef]:
    """Create an optimizer cluster and return handles"""
    orchestrator = CollateralOrchestrator.remote(num_workers=num_workers)
    return orchestrator


async def run_collateral_optimization_loop(
    orchestrator: ray.ObjectRef,
    balance_provider: callable,
    margin_provider: callable,
):
    """Run continuous collateral optimization loop"""
    import asyncio
    
    while True:
        try:
            # Get current state
            spot_balances, futures_balances = balance_provider()
            margin_state = margin_provider()
            
            # Run optimization
            recommendations = await orchestrator.run_optimization_cycle.remote(
                spot_balances,
                futures_balances,
                margin_state,
            )
            
            # Execute top recommendation if any
            recs = ray.get(recommendations)
            if recs:
                await orchestrator.execute_rebalance.remote(recs[0])
            
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            logging.error(f"Optimization loop error: {e}")
            await asyncio.sleep(5)


if __name__ == '__main__':
    import asyncio
    
    async def main():
        # Create orchestrator
        orchestrator = CollateralOrchestrator.remote(num_workers=4)
        await orchestrator.start.remote()
        
        # Mock data for testing
        spot_balances = [
            WalletBalance('spot', 'USDT', 100000, 0, 100000, 100000),
            WalletBalance('spot', 'BTC', 1.5, 0, 1.5, 75000),
        ]
        
        futures_balances = [
            WalletBalance('futures', 'USDT', 50000, 10000, 60000, 60000),
        ]
        
        margin_state = MarginState(
            total_collateral_usd=160000,
            used_margin_usd=50000,
            available_margin_usd=110000,
            margin_ratio=3.2,
            maintenance_margin_usd=25000,
            liquidation_risk=0.1,
        )
        
        # Run optimization
        recommendations = await orchestrator.run_optimization_cycle.remote(
            spot_balances,
            futures_balances,
            margin_state,
        )
        
        recs = ray.get(recommendations)
        print(f"Generated {len(recs)} recommendations")
        
        # Get stats
        stats = await orchestrator.get_orchestrator_stats.remote()
        print(f"Stats: {ray.get(stats)}")
        
        await orchestrator.stop.remote()
    
    asyncio.run(main())
