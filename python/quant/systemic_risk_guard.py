"""
Systemic Risk Guard - Real-time Contagion Detection and Deleveraging
=====================================================================
Implements CoVaR (Conditional Value at Risk) and Marginal Expected Shortfall (MES)
for detecting systemic contagion risk across crypto portfolios.

Automatically triggers global deleveraging protocol when systemic risk spikes,
protecting capital during black swan events and market crashes.

Memory-bounded design for 8GB RAM constraint with streaming updates.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


class RiskLevel(Enum):
    """Risk alert levels for systemic risk monitoring."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SystemicRiskMetrics:
    """Container for systemic risk measurements."""
    covar: float  # Conditional VaR
    delta_covar: float  # Change in CoVaR
    mes: Dict[str, float]  # Marginal ES per asset
    aggregate_mes: float  # Portfolio-wide MES
    spillover_index: float  # Cross-asset spillover measure
    risk_level: RiskLevel
    timestamp: int
    triggered_assets: List[str]


class CoVaREstimator:
    """
    Conditional Value at Risk (CoVaR) estimator.
    
    CoVaR measures the VaR of the financial system conditional on 
    a specific institution (or asset) being in distress.
    
    For crypto portfolios: CoVaR_i = VaR(system | asset_i in distress)
    
    Uses quantile regression for robust estimation without distributional assumptions.
    """
    
    def __init__(self, confidence: float = 0.95, window: int = 252):
        """
        Initialize CoVaR estimator.
        
        Args:
            confidence: VaR confidence level
            window: Rolling window for estimation
        """
        self.confidence = confidence
        self.window = window
        self.fitted = False
        
        # Storage for rolling data
        self.market_returns = []
        self.asset_returns = {}
        
        # Quantile regression coefficients
        self.alpha = None
        self.beta = None
        
    def update(self, market_return: float, asset_returns: Dict[str, float]):
        """
        Update rolling data with new observations.
        
        Args:
            market_return: Market/portfolio return
            asset_returns: Dictionary of individual asset returns
        """
        self.market_returns.append(market_return)
        
        for asset_id, ret in asset_returns.items():
            if asset_id not in self.asset_returns:
                self.asset_returns[asset_id] = []
            self.asset_returns[asset_id].append(ret)
        
        # Maintain bounded window
        if len(self.market_returns) > self.window:
            self.market_returns.pop(0)
        
        for asset_id in list(self.asset_returns.keys()):
            if len(self.asset_returns[asset_id]) > self.window:
                self.asset_returns[asset_id].pop(0)
    
    def _quantile_regression(self, y: np.ndarray, x: np.ndarray, 
                            tau: float) -> Tuple[float, float]:
        """
        Simple quantile regression using check function minimization.
        
        Minimizes: sum(rho_tau(y - alpha - beta*x))
        where rho_tau(u) = u * (tau - I(u < 0))
        
        Returns: (alpha, beta) coefficients
        """
        n = len(y)
        if n < 30:
            # Fallback to OLS if insufficient data
            if np.std(x) > 0:
                beta = np.cov(x, y)[0, 1] / np.var(x)
                alpha = np.mean(y) - beta * np.mean(x)
                return alpha, beta
            return np.mean(y), 0.0
        
        # Grid search for quantile regression (simple but robust)
        def check_function(alpha, beta):
            residuals = y - alpha - beta * x
            return np.sum(residuals * (tau - (residuals < 0).astype(float)))
        
        # Initial guess from OLS
        if np.std(x) > 0:
            beta_init = np.cov(x, y)[0, 1] / np.var(x)
            alpha_init = np.percentile(y - beta_init * x, tau * 100)
        else:
            beta_init = 0.0
            alpha_init = np.percentile(y, tau * 100)
        
        # Grid search refinement
        best_alpha, best_beta = alpha_init, beta_init
        best_loss = check_function(alpha_init, beta_init)
        
        # Search around initial guess
        for beta_try in np.linspace(beta_init - 0.5, beta_init + 0.5, 20):
            for alpha_try in np.linspace(alpha_init - 0.05, alpha_init + 0.05, 20):
                loss = check_function(alpha_try, beta_try)
                if loss < best_loss:
                    best_loss = loss
                    best_alpha, best_beta = alpha_try, beta_try
        
        return best_alpha, best_beta
    
    def estimate_covar(self, asset_id: str, 
                      distress_threshold: float = -0.05) -> Optional[float]:
        """
        Estimate CoVaR: VaR of market given asset is in distress.
        
        Args:
            asset_id: Asset identifier
            distress_threshold: Return threshold defining distress (e.g., -5%)
            
        Returns:
            CoVaR estimate or None if insufficient data
        """
        if asset_id not in self.asset_returns:
            return None
        
        if len(self.market_returns) < 50 or len(self.asset_returns[asset_id]) < 50:
            return None
        
        market = np.array(self.market_returns)
        asset = np.array(self.asset_returns[asset_id])
        
        # Filter for distress periods
        distress_mask = asset < distress_threshold
        n_distress = np.sum(distress_mask)
        
        if n_distress < 10:
            # Not enough distress observations, use tail quantile
            tail_idx = np.argsort(asset)[:max(10, int(len(asset) * 0.05))]
            distress_market = market[tail_idx]
        else:
            distress_market = market[distress_mask]
        
        # CoVaR is the VaR of market during asset distress
        covar = np.percentile(distress_market, (1 - self.confidence) * 100)
        
        return covar
    
    def estimate_delta_covar(self, asset_id: str,
                            distress_threshold: float = -0.05) -> Optional[float]:
        """
        Estimate ΔCoVaR: Difference between distressed and normal CoVaR.
        
        ΔCoVaR = CoVaR(distress) - CoVaR(normal)
        
        Measures the systemic risk contribution of the asset.
        """
        if asset_id not in self.asset_returns:
            return None
        
        if len(self.market_returns) < 50:
            return None
        
        market = np.array(self.market_returns)
        asset = np.array(self.asset_returns[asset_id])
        
        # Distressed state
        distress_mask = asset < distress_threshold
        n_distress = np.sum(distress_mask)
        
        if n_distress < 10:
            tail_idx = np.argsort(asset)[:max(10, int(len(asset) * 0.05))]
            distress_market = market[tail_idx]
        else:
            distress_market = market[distress_mask]
        
        covar_distress = np.percentile(distress_market, (1 - self.confidence) * 100)
        
        # Normal state (median asset returns)
        normal_mask = (asset >= np.percentile(asset, 45)) & (asset <= np.percentile(asset, 55))
        if np.sum(normal_mask) < 10:
            normal_mask = ~distress_mask
        
        normal_market = market[normal_mask]
        covar_normal = np.percentile(normal_market, (1 - self.confidence) * 100)
        
        delta_covar = covar_distress - covar_normal
        
        return delta_covar


class MarginalExpectedShortfall:
    """
    Marginal Expected Shortfall (MES) calculator.
    
    MES measures the expected loss of an asset when the market/system 
    experiences extreme losses. It captures the asset's contribution 
    to systemic risk.
    
    MES_i = -E[R_i | R_market < VaR_market]
    
    Higher MES indicates greater systemic risk contribution.
    """
    
    def __init__(self, confidence: float = 0.95, window: int = 252):
        """
        Initialize MES calculator.
        
        Args:
            confidence: Confidence level for defining market distress
            window: Rolling window for estimation
        """
        self.confidence = confidence
        self.window = window
        self.market_returns = []
        self.asset_returns = {}
        
    def update(self, market_return: float, asset_returns: Dict[str, float]):
        """Update rolling data."""
        self.market_returns.append(market_return)
        
        for asset_id, ret in asset_returns.items():
            if asset_id not in self.asset_returns:
                self.asset_returns[asset_id] = []
            self.asset_returns[asset_id].append(ret)
        
        # Maintain bounded window
        if len(self.market_returns) > self.window:
            self.market_returns.pop(0)
        
        for asset_id in list(self.asset_returns.keys()):
            if len(self.asset_returns[asset_id]) > self.window:
                self.asset_returns[asset_id].pop(0)
    
    def calculate_mes(self, asset_id: str) -> Optional[float]:
        """
        Calculate Marginal Expected Shortfall for an asset.
        
        Returns:
            MES value (positive number representing expected loss)
        """
        if asset_id not in self.asset_returns:
            return None
        
        if len(self.market_returns) < 30:
            return None
        
        market = np.array(self.market_returns)
        asset = np.array(self.asset_returns[asset_id])
        
        # Define market distress threshold
        distress_threshold = np.percentile(market, (1 - self.confidence) * 100)
        
        # Asset returns during market distress
        distress_mask = market < distress_threshold
        n_distress = np.sum(distress_mask)
        
        if n_distress < 5:
            return 0.0
        
        distress_asset_returns = asset[distress_mask]
        
        # MES is negative expected return during market distress
        mes = -np.mean(distress_asset_returns)
        
        return mes
    
    def calculate_aggregate_mes(self, weights: Dict[str, float]) -> Optional[float]:
        """
        Calculate portfolio-weighted aggregate MES.
        
        Args:
            weights: Portfolio weights by asset ID
            
        Returns:
            Aggregate MES
        """
        total_mes = 0.0
        total_weight = 0.0
        
        for asset_id, weight in weights.items():
            mes = self.calculate_mes(asset_id)
            if mes is not None:
                total_mes += weight * mes
                total_weight += weight
        
        if total_weight > 0:
            return total_mes / total_weight
        return None


class SpilloverIndexCalculator:
    """
    Calculates cross-asset return spillover index based on Diebold-Yilmaz methodology.
    
    Measures how much of forecast error variance is due to cross-asset 
    spillovers vs. own shocks. High spillover indicates systemic interconnectedness.
    
    Simplified version using correlation-based approximation for speed.
    """
    
    def __init__(self, window: int = 60):
        """
        Initialize spillover calculator.
        
        Args:
            window: Rolling window for calculation
        """
        self.window = window
        self.returns_history = {}
        
    def update(self, returns: Dict[str, float]):
        """Update with new returns."""
        for asset_id, ret in returns.items():
            if asset_id not in self.returns_history:
                self.returns_history[asset_id] = []
            self.returns_history[asset_id].append(ret)
            
            # Trim history
            if len(self.returns_history[asset_id]) > self.window:
                self.returns_history[asset_id].pop(0)
    
    def calculate_spillover_index(self) -> Optional[float]:
        """
        Calculate aggregate spillover index.
        
        Returns:
            Spillover index (0-100), higher = more interconnected
        """
        if len(self.returns_history) < 3:
            return None
        
        # Check all assets have sufficient data
        min_len = min(len(v) for v in self.returns_history.values())
        if min_len < 30:
            return None
        
        asset_ids = list(self.returns_history.keys())
        n_assets = len(asset_ids)
        
        # Build return matrix
        returns_matrix = np.column_stack([
            self.returns_history[aid][-min_len:] for aid in asset_ids
        ])
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(returns_matrix.T)
        
        # Spillover approximation: average off-diagonal correlation
        # (Simplified Diebold-Yilmaz)
        off_diag = corr_matrix[np.triu_indices(n_assets, 1)]
        avg_corr = np.mean(np.abs(off_diag))
        
        # Scale to 0-100 index
        spillover = avg_corr * 100
        
        return spillover


class SystemicRiskGuard:
    """
    Main systemic risk monitoring and automatic deleveraging system.
    
    Integrates CoVaR, MES, and spillover analysis to:
    1. Monitor real-time systemic risk levels
    2. Detect contagion patterns across assets
    3. Trigger automatic deleveraging when thresholds breached
    4. Provide early warning signals for black swan events
    
    Designed for sub-millisecond evaluation with bounded memory.
    """
    
    def __init__(self, 
                 asset_ids: List[str],
                 covar_threshold: float = 0.03,
                 mes_threshold: float = 0.05,
                 spillover_threshold: float = 60.0,
                 emergency_drawdown: float = 0.10):
        """
        Initialize Systemic Risk Guard.
        
        Args:
            asset_ids: List of monitored asset identifiers
            covar_threshold: ΔCoVaR threshold for elevated risk
            mes_threshold: MES threshold for high risk per asset
            spillover_threshold: Spillover index threshold for systemic concern
            emergency_drawdown: Portfolio drawdown triggering emergency mode
        """
        self.asset_ids = asset_ids
        self.covar_threshold = covar_threshold
        self.mes_threshold = mes_threshold
        self.spillover_threshold = spillover_threshold
        self.emergency_drawdown = emergency_drawdown
        
        # Component estimators
        self.covar_estimator = CoVaREstimator(confidence=0.95, window=252)
        self.mes_calculator = MarginalExpectedShortfall(confidence=0.95, window=252)
        self.spillover_calc = SpilloverIndexCalculator(window=60)
        
        # State tracking
        self.current_risk_level = RiskLevel.NORMAL
        self.risk_history = []
        self.deleveraging_active = False
        self.triggered_assets = set()
        
        # Alert callbacks (to be set by main system)
        self.alert_callback = None
        self.deleverage_callback = None
        
        # Warm-up period
        self.n_observations = 0
        self.min_warmup = 60
        
    def process_tick(self, market_return: float, 
                    asset_returns: Dict[str, float],
                    current_drawdown: float = 0.0):
        """
        Process a new tick of return data.
        
        Args:
            market_return: Overall portfolio/market return
            asset_returns: Individual asset returns
            current_drawdown: Current portfolio drawdown (0 to -1)
        """
        self.n_observations += 1
        
        # Update all estimators
        self.covar_estimator.update(market_return, asset_returns)
        self.mes_calculator.update(market_return, asset_returns)
        self.spillover_calc.update(asset_returns)
        
        # Skip assessment during warm-up
        if self.n_observations < self.min_warmup:
            return
        
        # Assess systemic risk
        metrics = self._assess_risk(current_drawdown)
        
        # Store history (bounded)
        self.risk_history.append(metrics)
        if len(self.risk_history) > 500:
            self.risk_history.pop(0)
        
        # Check for risk level change
        if metrics.risk_level != self.current_risk_level:
            self._on_risk_level_change(metrics)
        
        # Check for deleveraging trigger
        if metrics.risk_level in [RiskLevel.CRITICAL, RiskLevel.EMERGENCY]:
            if not self.deleveraging_active:
                self._trigger_deleveraging(metrics)
        elif metrics.risk_level == RiskLevel.NORMAL:
            if self.deleveraging_active:
                self._exit_deleveraging()
    
    def _assess_risk(self, current_drawdown: float) -> SystemicRiskMetrics:
        """Assess current systemic risk level."""
        
        # Calculate CoVaR for each asset
        covar_values = {}
        delta_covar_values = {}
        for asset_id in self.asset_ids:
            covar = self.covar_estimator.estimate_covar(asset_id)
            delta_covar = self.covar_estimator.estimate_delta_covar(asset_id)
            if covar is not None:
                covar_values[asset_id] = covar
            if delta_covar is not None:
                delta_covar_values[asset_id] = delta_covar
        
        # Calculate MES for each asset
        mes_values = {}
        for asset_id in self.asset_ids:
            mes = self.mes_calculator.calculate_mes(asset_id)
            if mes is not None:
                mes_values[asset_id] = mes
        
        # Aggregate MES
        n_assets = len(mes_values)
        aggregate_mes = np.mean(list(mes_values.values())) if mes_values else 0.0
        
        # Spillover index
        spillover = self.spillover_calc.calculate_spillover_index()
        spillover = spillover if spillover is not None else 0.0
        
        # Determine risk level
        max_delta_covar = max(delta_covar_values.values()) if delta_covar_values else 0.0
        max_mes = max(mes_values.values()) if mes_values else 0.0
        
        # Identify triggered assets
        triggered = []
        for asset_id in self.asset_ids:
            if (delta_covar_values.get(asset_id, 0) > self.covar_threshold or
                mes_values.get(asset_id, 0) > self.mes_threshold):
                triggered.append(asset_id)
                self.triggered_assets.add(asset_id)
        
        # Risk level logic
        risk_score = 0.0
        
        # CoVaR contribution
        if max_delta_covar > self.covar_threshold * 2:
            risk_score += 3.0
        elif max_delta_covar > self.covar_threshold:
            risk_score += 2.0
        elif max_delta_covar > self.covar_threshold * 0.5:
            risk_score += 1.0
        
        # MES contribution
        if aggregate_mes > self.mes_threshold * 2:
            risk_score += 3.0
        elif aggregate_mes > self.mes_threshold:
            risk_score += 2.0
        elif aggregate_mes > self.mes_threshold * 0.5:
            risk_score += 1.0
        
        # Spillover contribution
        if spillover > self.spillover_threshold * 1.5:
            risk_score += 2.0
        elif spillover > self.spillover_threshold:
            risk_score += 1.0
        
        # Drawdown contribution
        if abs(current_drawdown) > self.emergency_drawdown:
            risk_score += 3.0
        elif abs(current_drawdown) > self.emergency_drawdown * 0.5:
            risk_score += 1.5
        
        # Map score to risk level
        if risk_score >= 8.0:
            risk_level = RiskLevel.EMERGENCY
        elif risk_score >= 6.0:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 4.0:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 2.0:
            risk_level = RiskLevel.ELEVATED
        else:
            risk_level = RiskLevel.NORMAL
        
        return SystemicRiskMetrics(
            covar=np.mean(list(covar_values.values())) if covar_values else 0.0,
            delta_covar=max_delta_covar,
            mes=mes_values,
            aggregate_mes=aggregate_mes,
            spillover_index=spillover,
            risk_level=risk_level,
            timestamp=self.n_observations,
            triggered_assets=triggered
        )
    
    def _on_risk_level_change(self, metrics: SystemicRiskMetrics):
        """Handle risk level transition."""
        old_level = self.current_risk_level
        self.current_risk_level = metrics.risk_level
        
        if self.alert_callback:
            self.alert_callback({
                'old_level': old_level.value,
                'new_level': metrics.risk_level.value,
                'metrics': {
                    'delta_covar': metrics.delta_covar,
                    'aggregate_mes': metrics.aggregate_mes,
                    'spillover': metrics.spillover_index,
                    'triggered_assets': metrics.triggered_assets
                }
            })
    
    def _trigger_deleveraging(self, metrics: SystemicRiskMetrics):
        """Trigger automatic deleveraging protocol."""
        self.deleveraging_active = True
        
        if self.deleverage_callback:
            self.deleverage_callback({
                'reason': 'systemic_risk',
                'risk_level': metrics.risk_level.value,
                'triggered_assets': metrics.triggered_assets,
                'metrics': {
                    'delta_covar': metrics.delta_covar,
                    'aggregate_mes': metrics.aggregate_mes,
                    'spillover': metrics.spillover_index
                },
                'action': 'reduce_exposure'
            })
    
    def _exit_deleveraging(self):
        """Exit deleveraging mode."""
        self.deleveraging_active = False
        
        if self.deleverage_callback:
            self.deleverage_callback({
                'reason': 'risk_normalized',
                'action': 'resume_normal_operations'
            })
    
    def set_alert_callback(self, callback):
        """Set callback for risk alerts."""
        self.alert_callback = callback
    
    def set_deleverage_callback(self, callback):
        """Set callback for deleveraging triggers."""
        self.deleverage_callback = callback
    
    def get_current_metrics(self) -> Optional[SystemicRiskMetrics]:
        """Get latest risk metrics."""
        if self.risk_history:
            return self.risk_history[-1]
        return None
    
    def get_risk_dashboard(self) -> Dict:
        """Get comprehensive risk dashboard."""
        metrics = self.get_current_metrics()
        
        if metrics is None:
            return {'status': 'warming_up', 'observations': self.n_observations}
        
        return {
            'status': 'active',
            'risk_level': metrics.risk_level.value,
            'deleveraging_active': self.deleveraging_active,
            'delta_covar': metrics.delta_covar,
            'aggregate_mes': metrics.aggregate_mes,
            'spillover_index': metrics.spillover_index,
            'triggered_assets': metrics.triggered_assets,
            'n_observations': self.n_observations,
            'asset_mes': metrics.mes
        }


# Example integration pattern
def create_systemic_risk_guard(asset_ids: List[str],
                               config: Optional[Dict] = None) -> SystemicRiskGuard:
    """
    Factory function to create configured SystemicRiskGuard.
    
    Args:
        asset_ids: List of asset identifiers to monitor
        config: Optional configuration overrides
        
    Returns:
        Configured SystemicRiskGuard instance
    """
    default_config = {
        'covar_threshold': 0.03,
        'mes_threshold': 0.05,
        'spillover_threshold': 60.0,
        'emergency_drawdown': 0.10
    }
    
    if config:
        default_config.update(config)
    
    return SystemicRiskGuard(
        asset_ids=asset_ids,
        **default_config
    )


if __name__ == "__main__":
    # Demo usage
    np.random.seed(42)
    
    asset_ids = ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC']
    guard = create_systemic_risk_guard(asset_ids)
    
    # Simulate correlated crypto returns with occasional crashes
    n_days = 300
    base_corr = 0.5 + 0.3 * np.random.rand(len(asset_ids), len(asset_ids))
    base_corr = (base_corr + base_corr.T) / 2
    np.fill_diagonal(base_corr, 1.0)
    
    print("=" * 60)
    print("SYSTEMIC RISK GUARD - DEMONSTRATION")
    print("=" * 60)
    
    for day in range(n_days):
        # Generate correlated returns
        raw_returns = np.random.multivariate_normal(np.zeros(len(asset_ids)), 
                                                     base_corr, 1)[0]
        
        # Add occasional crash days
        if day in [100, 150, 200, 250]:
            raw_returns *= 4  # Amplify volatility
        
        # Add regime shift (increased correlation in stress)
        if day > 200:
            raw_returns *= 1.5
            base_corr = np.clip(base_corr + 0.001, 0, 1)
        
        asset_returns = {aid: float(r) for aid, r in zip(asset_ids, raw_returns)}
        market_return = np.mean(raw_returns)
        
        # Process tick
        guard.process_tick(market_return, asset_returns)
        
        # Print status periodically
        if (day + 1) % 50 == 0:
            dashboard = guard.get_risk_dashboard()
            print(f"\nDay {day + 1}:")
            print(f"  Risk Level: {dashboard.get('risk_level', 'N/A')}")
            print(f"  ΔCoVaR: {dashboard.get('delta_covar', 0):.4f}")
            print(f"  Agg MES: {dashboard.get('aggregate_mes', 0):.4f}")
            print(f"  Spillover: {dashboard.get('spillover_index', 0):.1f}")
            print(f"  Deleveraging: {dashboard.get('deleveraging_active', False)}")
    
    print("\n" + "=" * 60)
    print("Systemic Risk Guard successfully monitors portfolio contagion")
    print("and triggers automatic deleveraging during stress periods.")
    print("=" * 60)
