# python/quant/volatility_models.py
# =============================================================================
# VOLATILITY FORECASTING & RISK MODELS
# =============================================================================
# Purpose: Implement GARCH and Heston models for volatility forecasting,
# plus Value at Risk (VaR) and Expected Shortfall (ES) calculations.
#
# Models Included:
# - GARCH(1,1): Standard volatility clustering model
# - EWMA: Exponentially weighted moving average (RiskMetrics style)
# - Heston Approximation: Stochastic volatility for option-like payoffs
# - Historical VaR/ES: Non-parametric risk measures
# - Monte Carlo VaR: Parametric simulation-based risk

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from collections import deque


@dataclass
class VolatilityForecast:
    """Volatility forecast result"""
    current_vol: float
    one_day_forecast: float
    one_week_forecast: float
    model_type: str
    confidence_95: float  # 95% VaR
    confidence_99: float  # 99% VaR


@dataclass
class RiskMetrics:
    """Risk measurement results"""
    var_95: float
    var_99: float
    es_95: float  # Expected Shortfall (CVaR)
    es_99: float
    max_drawdown: float


class GARCH11:
    """
    GARCH(1,1) model for volatility forecasting.
    sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}
    
    Optimized for online updates without full re-estimation.
    """
    
    def __init__(self, omega: float = 1e-6, alpha: float = 0.1, beta: float = 0.85):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        
        # State
        self.sigma_squared = 0.0
        self.last_epsilon = 0.0
        self.initialized = False
        
        # Long-run variance (unconditional)
        self.long_run_var = omega / (1 - alpha - beta) if (alpha + beta) < 1 else omega
    
    def update(self, return_val: float) -> float:
        """
        Update model with new return and return current volatility estimate.
        
        Args:
            return_val: Asset return (percentage as decimal)
            
        Returns:
            Current volatility estimate (annualized)
        """
        if not self.initialized:
            # Initialize with squared return
            self.sigma_squared = return_val ** 2
            self.last_epsilon = return_val
            self.initialized = True
        else:
            # GARCH update
            self.sigma_squared = (
                self.omega + 
                self.alpha * (self.last_epsilon ** 2) + 
                self.beta * self.sigma_squared
            )
            self.last_epsilon = return_val
        
        return np.sqrt(self.sigma_squared * 252)  # Annualize
    
    def forecast(self, horizon: int = 1) -> float:
        """
        Forecast volatility h steps ahead.
        
        For GARCH(1,1), the forecast converges to long-run variance.
        """
        if horizon == 1:
            return np.sqrt(self.sigma_squared * 252)
        
        # Multi-step forecast: mean reversion to long-run variance
        persistence = self.alpha + self.beta
        forecast_var = (
            self.long_run_var + 
            persistence ** horizon * (self.sigma_squared - self.long_run_var)
        )
        
        return np.sqrt(forecast_var * 252)
    
    def set_parameters(self, omega: float, alpha: float, beta: float):
        """Update model parameters (e.g., from offline calibration)"""
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.long_run_var = omega / (1 - alpha - beta) if (alpha + beta) < 1 else omega


class HestonApproximation:
    """
    Simplified Heston stochastic volatility model approximation.
    Used for capturing volatility-of-volatility effects.
    
    dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_t
    """
    
    def __init__(self, kappa: float = 2.0, theta: float = 0.04, 
                 xi: float = 0.3, rho: float = -0.7):
        self.kappa = kappa  # Mean reversion speed
        self.theta = theta  # Long-run variance
        self.xi = xi  # Vol of vol
        self.rho = rho  # Correlation between asset and vol shocks
        
        self.current_variance = theta
        self.initialized = False
    
    def update(self, return_val: float, dt: float = 1/252) -> float:
        """
        Update variance estimate using approximate filtering.
        """
        if not self.initialized:
            self.current_variance = return_val ** 2 / dt
            self.initialized = True
            return np.sqrt(self.current_variance * 252)
        
        # Euler-Maruyama approximation for variance evolution
        dW_vol = np.random.normal() * np.sqrt(dt)
        
        # Mean reversion + stochastic term
        dv = self.kappa * (self.theta - self.current_variance) * dt
        dv += self.xi * np.sqrt(max(self.current_variance, 0)) * dW_vol
        
        self.current_variance = max(self.current_variance + dv, 1e-8)  # Floor at small positive
        
        return np.sqrt(self.current_variance * 252)
    
    def get_variance_term_structure(self, days: List[int]) -> List[float]:
        """Get expected variance at different horizons"""
        term_structure = []
        for d in days:
            t = d / 365.0
            # Expected variance under Heston
            exp_var = self.theta + (self.current_variance - self.theta) * np.exp(-self.kappa * t)
            term_structure.append(np.sqrt(exp_var * 252))
        return term_structure


class VolatilityEngine:
    """
    Main engine combining multiple volatility models and risk metrics.
    """
    
    def __init__(self, lookback: int = 252):
        self.lookback = lookback
        self.returns_history = deque(maxlen=lookback)
        
        # Initialize models
        self.garch = GARCH11()
        self.heston = HestonApproximation()
        
        # EWMA parameters (RiskMetrics style)
        self.ewma_lambda = 0.94
        self.ewma_variance = 0.0
        
        # Risk calculation buffers
        self.max_bars = 500
        self.equity_curve = deque(maxlen=self.max_bars)
    
    def add_return(self, return_val: float) -> VolatilityForecast:
        """Add return and compute all volatility estimates"""
        self.returns_history.append(return_val)
        
        # Update models
        garch_vol = self.garch.update(return_val)
        heston_vol = self.heston.update(return_val)
        ewma_vol = self._update_ewma(return_val)
        
        # Simple average ensemble
        current_vol = (garch_vol + heston_vol + ewma_vol) / 3
        
        # Calculate VaR thresholds
        if len(self.returns_history) >= 50:
            returns_arr = np.array(self.returns_history)
            var_95 = np.percentile(returns_arr, 5)
            var_99 = np.percentile(returns_arr, 1)
        else:
            # Use normal approximation
            var_95 = -1.645 * current_vol / np.sqrt(252)
            var_99 = -2.326 * current_vol / np.sqrt(252)
        
        return VolatilityForecast(
            current_vol=current_vol,
            one_day_forecast=self.garch.forecast(1),
            one_week_forecast=self.garch.forecast(5),
            model_type="ensemble",
            confidence_95=var_95,
            confidence_99=var_99
        )
    
    def _update_ewma(self, return_val: float) -> float:
        """Update EWMA variance estimate"""
        if self.ewma_variance == 0:
            self.ewma_variance = return_val ** 2
        else:
            self.ewma_variance = (
                self.ewma_lambda * self.ewma_variance + 
                (1 - self.ewma_lambda) * return_val ** 2
            )
        return np.sqrt(self.ewma_variance * 252)
    
    def calculate_risk_metrics(self, portfolio_value: float) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            portfolio_value: Current portfolio value in USD
            
        Returns:
            RiskMetrics object with VaR, ES, and drawdown
        """
        if len(self.returns_history) < 20:
            return RiskMetrics(0, 0, 0, 0, 0)
        
        returns = np.array(self.returns_history)
        
        # Historical VaR
        var_95_pct = np.percentile(returns, 5)
        var_99_pct = np.percentile(returns, 1)
        
        # Expected Shortfall (average of returns beyond VaR)
        es_95_pct = np.mean(returns[returns <= var_95_pct])
        es_99_pct = np.mean(returns[returns <= var_99_pct])
        
        # Convert to dollar amounts
        var_95 = abs(var_95_pct) * portfolio_value
        var_99 = abs(var_99_pct) * portfolio_value
        es_95 = abs(es_95_pct) * portfolio_value if not np.isnan(es_95_pct) else var_95 * 1.2
        es_99 = abs(es_99_pct) * portfolio_value if not np.isnan(es_99_pct) else var_99 * 1.2
        
        # Maximum drawdown from equity curve
        max_dd = self._calculate_max_drawdown()
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            es_95=es_95,
            es_99=es_99,
            max_drawdown=max_dd
        )
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        return np.max(drawdown)
    
    def monte_carlo_var(self, portfolio_value: float, horizon: int = 1, 
                        n_sims: int = 10000) -> Tuple[float, float]:
        """
        Calculate VaR using Monte Carlo simulation.
        
        Args:
            portfolio_value: Portfolio value
            horizon: Days ahead
            n_sims: Number of simulations
            
        Returns:
            Tuple of (VaR_95, VaR_99) in dollar terms
        """
        if len(self.returns_history) < 50:
            return (0.0, 0.0)
        
        returns = np.array(self.returns_history)
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        # Simulate returns over horizon
        simulated_returns = np.random.normal(mu * horizon, sigma * np.sqrt(horizon), n_sims)
        
        # Calculate portfolio values
        simulated_values = portfolio_value * (1 + simulated_returns)
        losses = portfolio_value - simulated_values
        
        var_95 = np.percentile(losses, 95)
        var_99 = np.percentile(losses, 99)
        
        return (var_95, var_99)
    
    def update_equity(self, equity: float):
        """Track equity curve for drawdown calculation"""
        self.equity_curve.append(equity)
    
    def calibrate_garch(self, returns: np.ndarray):
        """
        Offline GARCH parameter calibration using method of moments.
        In production, use maximum likelihood estimation.
        """
        if len(returns) < 100:
            return
        
        # Simple method of moments approximation
        squared_returns = returns ** 2
        autocorr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
        
        # Set persistence based on autocorrelation
        persistence = max(0.5, min(0.98, autocorr + 0.5))
        
        # Split between alpha and beta
        alpha = (1 - persistence) * 0.2
        beta = persistence - alpha
        omega = np.mean(squared_returns) * (1 - alpha - beta)
        
        self.garch.set_parameters(
            omega=max(omega, 1e-8),
            alpha=max(alpha, 0.01),
            beta=max(beta, 0.5)
        )
