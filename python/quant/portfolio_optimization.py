# python/quant/portfolio_optimization.py
# =============================================================================
# PORTFOLIO OPTIMIZATION & DYNAMIC POSITION SIZING
# =============================================================================
# Purpose: Implement Modern Portfolio Theory, Kelly Criterion, Risk Parity,
# and Hidden Markov Model regime detection for adaptive position sizing.
#
# Components:
# - Markowitz Mean-Variance Optimization
# - Kelly Criterion for optimal growth
# - Risk Parity allocation
# - HMM-based regime detection for adaptive strategies

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class PortfolioWeights:
    """Optimal portfolio weights"""
    assets: List[str]
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    method: str  # 'markowitz', 'kelly', 'risk_parity'


@dataclass
class RegimeState:
    """Market regime detection result"""
    regime: int  # 0=bull, 1=bear, 2=high_vol, 3=low_vol
    probability: float
    recommended_leverage: float


class MarkowitzOptimizer:
    """
    Mean-Variance Portfolio Optimization (Modern Portfolio Theory).
    Uses efficient frontier calculation for optimal weights.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def optimize(self, returns: Dict[str, np.ndarray], 
                 target_return: Optional[float] = None) -> PortfolioWeights:
        """
        Calculate optimal portfolio weights using mean-variance optimization.
        
        Args:
            returns: Dictionary of asset names to return arrays
            target_return: Optional target return (otherwise maximize Sharpe)
            
        Returns:
            PortfolioWeights object
        """
        assets = list(returns.keys())
        n_assets = len(assets)
        
        if n_assets < 2:
            # Single asset: 100% weight
            arr = list(returns.values())[0]
            return PortfolioWeights(
                assets=assets,
                weights=np.array([1.0]),
                expected_return=np.mean(arr),
                volatility=np.std(arr),
                sharpe_ratio=(np.mean(arr) - self.risk_free_rate) / (np.std(arr) + 1e-10),
                method='single_asset'
            )
        
        # Create returns matrix
        ret_matrix = np.column_stack([returns[a] for a in assets])
        
        # Expected returns and covariance
        mu = np.mean(ret_matrix, axis=0)
        cov = np.cov(ret_matrix.T)
        
        # Add small regularization for numerical stability
        cov += np.eye(n_assets) * 1e-8
        
        if target_return is None:
            # Maximum Sharpe Ratio portfolio
            weights = self._max_sharpe(mu, cov)
        else:
            # Minimum variance for target return
            weights = self._min_variance_target(mu, cov, target_return)
        
        # Calculate portfolio metrics
        port_return = np.sum(weights * mu)
        port_vol = np.sqrt(weights @ cov @ weights)
        sharpe = (port_return - self.risk_free_rate) / (port_vol + 1e-10)
        
        return PortfolioWeights(
            assets=assets,
            weights=weights,
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            method='markowitz'
        )
    
    def _max_sharpe(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Find maximum Sharpe ratio weights"""
        n = len(mu)
        
        # Analytical solution for max Sharpe (no constraints)
        cov_inv = np.linalg.inv(cov)
        excess_returns = mu - self.risk_free_rate
        
        # w ∝ Σ^(-1) * (μ - r_f)
        raw_weights = cov_inv @ excess_returns
        
        # Normalize to sum to 1
        total = np.sum(raw_weights)
        if abs(total) > 1e-10:
            weights = raw_weights / total
        else:
            weights = np.ones(n) / n
        
        # Apply constraints (long only, max 50% per asset)
        weights = np.clip(weights, 0.0, 0.5)
        weights /= np.sum(weights) + 1e-10
        
        return weights
    
    def _min_variance_target(self, mu: np.ndarray, cov: np.ndarray, 
                              target: float) -> np.ndarray:
        """Find minimum variance portfolio for target return"""
        n = len(mu)
        cov_inv = np.linalg.inv(cov)
        
        # Lagrange multiplier solution
        ones = np.ones(n)
        
        A = ones @ cov_inv @ ones
        B = ones @ cov_inv @ mu
        C = mu @ cov_inv @ mu
        det = A * C - B * B
        
        if abs(det) < 1e-10:
            return np.ones(n) / n
        
        # Weights for target return
        lambda1 = (C - B * target) / det
        lambda2 = (A * target - B) / det
        
        weights = cov_inv @ (lambda1 * ones + lambda2 * mu)
        
        # Apply constraints
        weights = np.clip(weights, 0.0, 0.5)
        weights /= np.sum(weights) + 1e-10
        
        return weights


class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing.
    Maximizes logarithmic utility of wealth.
    """
    
    def __init__(self, max_kelly_fraction: float = 0.25, 
                 fractional_kelly: float = 0.5):
        self.max_kelly = max_kelly_fraction
        self.fractional = fractional_kelly  # Use fraction of Kelly (risk management)
    
    def calculate_kelly(self, win_rate: float, win_loss_ratio: float) -> float:
        """
        Calculate Kelly fraction for a strategy.
        
        Args:
            win_rate: Probability of winning (0 to 1)
            win_loss_ratio: Average win / Average loss
            
        Returns:
            Kelly fraction (position size as fraction of capital)
        """
        if win_loss_ratio <= 0:
            return 0.0
        
        # Kelly formula: f* = p - q/b where p=win_prob, q=loss_prob, b=win/loss ratio
        p = win_rate
        q = 1 - win_rate
        b = win_loss_ratio
        
        kelly = p - q / b
        
        # Apply fractional Kelly and cap
        kelly = kelly * self.fractional
        kelly = max(0, min(kelly, self.max_kelly))
        
        return kelly
    
    def calculate_multi_asset_kelly(self, returns: Dict[str, np.ndarray],
                                     cov: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Approximate Kelly allocation for multiple assets.
        Uses continuous-time approximation: w* = Σ^(-1) * μ
        """
        assets = list(returns.keys())
        ret_matrix = np.column_stack([returns[a] for a in assets])
        
        mu = np.mean(ret_matrix, axis=0) * 252  # Annualize
        if cov is None:
            cov = np.cov(ret_matrix.T) * 252
        
        # Regularize
        cov += np.eye(len(assets)) * 1e-8
        
        try:
            cov_inv = np.linalg.inv(cov)
            raw_weights = cov_inv @ mu
            
            # Apply fractional Kelly and caps
            weights = {}
            total_abs = np.sum(np.abs(raw_weights))
            for i, asset in enumerate(assets):
                w = raw_weights[i] * self.fractional
                w = max(-self.max_kelly, min(w, self.max_kelly))
                weights[asset] = w
            
            return weights
        except np.linalg.LinAlgError:
            # Fallback to equal weight
            return {a: 1.0 / len(assets) for a in assets}


class RiskParity:
    """
    Risk Parity allocation - equal risk contribution from each asset.
    """
    
    def allocate(self, returns: Dict[str, np.ndarray]) -> PortfolioWeights:
        """Calculate risk parity weights"""
        assets = list(returns.keys())
        ret_matrix = np.column_stack([returns[a] for a in assets])
        
        cov = np.cov(ret_matrix.T)
        n = len(assets)
        
        # Simple approximation: inverse volatility weighting
        volatilities = np.sqrt(np.diag(cov))
        inv_vol = 1.0 / (volatilities + 1e-10)
        weights = inv_vol / np.sum(inv_vol)
        
        # Refine with iterative risk budgeting (simplified)
        for _ in range(10):
            port_vol = np.sqrt(weights @ cov @ weights)
            marginal_risk = cov @ weights / port_vol
            risk_contrib = weights * marginal_risk
            
            # Adjust weights toward equal risk contribution
            target_risk = port_vol / n
            adjustment = target_risk / (risk_contrib + 1e-10)
            weights *= adjustment
            weights /= np.sum(weights)
        
        # Calculate metrics
        mu = np.mean(ret_matrix, axis=0)
        port_return = np.sum(weights * mu)
        port_vol = np.sqrt(weights @ cov @ weights)
        
        return PortfolioWeights(
            assets=assets,
            weights=weights,
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=port_return / (port_vol + 1e-10),
            method='risk_parity'
        )


class HiddenMarkovRegimeDetector:
    """
    Simplified HMM for market regime detection.
    In production, use hmmlearn or pyhmm for full HMM implementation.
    """
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.returns_buffer = deque(maxlen=252)
        
        # Regime characteristics (to be learned)
        self.regime_means = np.zeros(n_regimes)
        self.regime_stds = np.ones(n_regimes)
        self.regime_probs = np.ones(n_regimes) / n_regimes
        
        # Initialized flag
        self.calibrated = False
    
    def add_return(self, return_val: float) -> Optional[RegimeState]:
        """Add return and detect current regime"""
        self.returns_buffer.append(return_val)
        
        if len(self.returns_buffer) < 60:
            return None
        
        if not self.calibrated:
            self._calibrate()
        
        # Calculate recent statistics
        recent = np.array(list(self.returns_buffer)[-20:])
        recent_mean = np.mean(recent)
        recent_std = np.std(recent)
        
        # Find most likely regime
        likelihoods = []
        for k in range(self.n_regimes):
            # Gaussian likelihood
            diff = recent_mean - self.regime_means[k]
            likelihood = np.exp(-0.5 * diff**2 / (self.regime_stds[k]**2 + 1e-10))
            likelihood *= self.regime_probs[k]
            likelihoods.append(likelihood)
        
        regime = np.argmax(likelihoods)
        prob = likelihoods[regime] / (sum(likelihoods) + 1e-10)
        
        # Determine recommended leverage based on regime
        leverage = self._get_recommended_leverage(regime, recent_std)
        
        return RegimeState(
            regime=regime,
            probability=prob,
            recommended_leverage=leverage
        )
    
    def _calibrate(self):
        """Calibrate regime parameters from historical data"""
        if len(self.returns_buffer) < 100:
            return
        
        returns = np.array(self.returns_buffer)
        
        # Simple clustering-based initialization
        # Sort returns and divide into quantiles
        sorted_ret = np.sort(returns)
        n = len(sorted_ret)
        
        for k in range(self.n_regimes):
            start = int(k * n / self.n_regimes)
            end = int((k + 1) * n / self.n_regimes)
            segment = sorted_ret[start:end]
            
            self.regime_means[k] = np.mean(segment)
            self.regime_stds[k] = max(np.std(segment), 0.001)
            self.regime_probs[k] = 1.0 / self.n_regimes
        
        self.calibrated = True
    
    def _get_recommended_leverage(self, regime: int, current_vol: float) -> float:
        """Get leverage recommendation based on regime"""
        # Regime interpretation:
        # 0: Low vol, positive return (bull) -> High leverage
        # 1: Low vol, negative return (bear) -> Low leverage
        # 2: High vol, positive return (volatile bull) -> Medium leverage
        # 3: High vol, negative return (crash) -> Very low leverage
        
        base_leverage = 1.0
        
        if regime == 0:  # Bull
            return min(2.0, base_leverage * 1.5)
        elif regime == 1:  # Bear
            return base_leverage * 0.5
        elif regime == 2:  # Volatile bull
            return base_leverage * 1.0
        else:  # Crash/high vol bear
            return base_leverage * 0.25


class PortfolioOptimizer:
    """
    Main portfolio optimization engine combining all methods.
    """
    
    def __init__(self):
        self.markowitz = MarkowitzOptimizer()
        self.kelly = KellyCriterion()
        self.risk_parity = RiskParity()
        self.regime_detector = HiddenMarkovRegimeDetector()
        
        # Track returns for each asset
        self.asset_returns: Dict[str, deque] = {}
        self.max_history = 500
    
    def add_asset(self, symbol: str):
        """Add new asset to tracking"""
        self.asset_returns[symbol] = deque(maxlen=self.max_history)
    
    def update_prices(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        Update prices and return optimal allocations.
        
        Returns:
            Dictionary of asset -> optimal weight
        """
        # Calculate returns
        for symbol, price in prices.items():
            if symbol not in self.asset_returns:
                self.add_asset(symbol)
            
            if len(self.asset_returns[symbol]) > 0:
                prev_price = list(self.asset_returns[symbol])[-1]
                ret = (price - prev_price) / prev_price
                self.asset_returns[symbol].append(ret)
            else:
                self.asset_returns[symbol].append(0.0)
        
        # Get regime
        avg_return = np.mean([list(r)[-1] if len(r) > 0 else 0 for r in self.asset_returns.values()])
        regime_state = self.regime_detector.add_return(avg_return)
        
        # Prepare returns dict for optimization
        returns_dict = {s: np.array(list(r)) for s, r in self.asset_returns.items() if len(r) >= 50}
        
        if len(returns_dict) < 2:
            return {s: 1.0 / len(prices) for s in prices}
        
        # Run optimizations
        markowitz_weights = self.markowitz.optimize(returns_dict)
        risk_parity_weights = self.risk_parity.allocate(returns_dict)
        kelly_weights = self.kelly.calculate_multi_asset_kelly(returns_dict)
        
        # Blend approaches based on regime
        if regime_state and regime_state.regime in [0, 2]:  # Favorable regimes
            # More aggressive: Kelly-focused
            final_weights = {}
            for i, asset in enumerate(markowitz_weights.assets):
                kelly_w = kelly_weights.get(asset, 0)
                mark_w = markowitz_weights.weights[i]
                final_weights[asset] = 0.6 * kelly_w + 0.4 * mark_w
        else:
            # Conservative: Risk parity focused
            final_weights = dict(zip(risk_parity_weights.assets, risk_parity_weights.weights))
        
        # Normalize
        total = sum(abs(w) for w in final_weights.values())
        if total > 0:
            final_weights = {k: v / total for k, v in final_weights.items()}
        
        return final_weights
