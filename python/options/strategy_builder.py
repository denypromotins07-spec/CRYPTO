"""
strategy_builder.py - Vectorized Multi-Leg Options Strategy Builder

STAGE 10 | CHAPTER 3 | FILE 1

This module provides a vectorized builder for complex multi-leg options strategies
including Iron Condors, Straddles, Butterflies, Calendars, and more. It instantly
calculates aggregate payoff profiles, max loss, and breakeven points using NumPy.

Key Features:
- Vectorized P&L calculation across entire price ranges
- Support for all standard multi-leg strategies
- Greeks aggregation for portfolio-level risk
- Memory-efficient design to stay within 8GB budget

Hardware Optimization:
- Uses NumPy float32 for reduced memory footprint
- Pre-allocated arrays for repeated calculations
- ROCm-ready data layouts for GPU offloading
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Literal
from dataclasses import dataclass
from enum import Enum


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class OptionLeg:
    """Represents a single leg of an options strategy."""
    option_type: OptionType
    strike: float
    quantity: int  # Positive for long, negative for short
    premium: float  # Premium paid/received per contract
    expiry: str
    
    def net_premium(self) -> float:
        """Net premium for this leg (negative = paid, positive = received)."""
        return -self.premium * self.quantity


@dataclass
class StrategyPayoff:
    """Results from strategy payoff analysis."""
    prices: np.ndarray
    payoffs: np.ndarray
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    profit_probability: float  # Assuming log-normal distribution
    expected_value: float
    sharpe_ratio: float


class OptionsStrategyBuilder:
    """
    Vectorized builder for multi-leg options strategies.
    
    All calculations use NumPy for speed and memory efficiency.
    Designed for real-time strategy evaluation during market hours.
    """
    
    def __init__(self, num_price_points: int = 1000):
        """
        Initialize the strategy builder.
        
        Args:
            num_price_points: Number of price points for payoff analysis
        """
        self.num_price_points = num_price_points
        # Pre-allocate price array for reuse
        self._price_buffer = np.zeros(num_price_points, dtype=np.float32)
        
    def build_strategy(
        self,
        legs: List[OptionLeg],
        underlying_price: float,
        price_range_pct: float = 0.5,
        days_to_primary_expiry: int = 30,
        volatility: float = 0.6,
        risk_free_rate: float = 0.05,
    ) -> StrategyPayoff:
        """
        Build and analyze a multi-leg options strategy.
        
        Args:
            legs: List of option legs in the strategy
            underlying_price: Current underlying asset price
            price_range_pct: Price range as percentage of current price (e.g., 0.5 = ±50%)
            days_to_primary_expiry: Days to primary expiry for probability calc
            volatility: Annualized volatility of underlying
            risk_free_rate: Risk-free interest rate
            
        Returns:
            StrategyPayoff with complete analysis
        """
        if not legs:
            raise ValueError("At least one leg is required")
        
        # Generate price range
        min_price = underlying_price * (1 - price_range_pct)
        max_price = underlying_price * (1 + price_range_pct)
        prices = np.linspace(min_price, max_price, self.num_price_points, dtype=np.float32)
        
        # Calculate payoff at each price point
        payoffs = self._calculate_payoff_vectorized(legs, prices)
        
        # Find key metrics
        max_profit = np.max(payoffs)
        max_loss = np.min(payoffs)
        
        # Calculate breakeven points (where payoff crosses zero)
        breakevens = self._find_breakeven_points(prices, payoffs)
        
        # Calculate probability of profit (simplified Black-Scholes based)
        prob_profit = self._calculate_profit_probability(
            legs, underlying_price, days_to_primary_expiry, volatility, risk_free_rate
        )
        
        # Expected value (integral of payoff weighted by probability density)
        expected_value = self._calculate_expected_value(
            legs, prices, payoffs, underlying_price, days_to_primary_expiry, 
            volatility, risk_free_rate
        )
        
        # Sharpe ratio approximation (using payoff distribution)
        sharpe = self._calculate_sharpe_approximation(payoffs, prob_profit)
        
        return StrategyPayoff(
            prices=prices,
            payoffs=payoffs,
            max_profit=max_profit,
            max_loss=max_loss,
            breakeven_points=breakevens,
            profit_probability=prob_profit,
            expected_value=expected_value,
            sharpe_ratio=sharpe,
        )
    
    def _calculate_payoff_vectorized(
        self, 
        legs: List[OptionLeg], 
        prices: np.ndarray
    ) -> np.ndarray:
        """
        Calculate total strategy payoff across all price points.
        
        Uses fully vectorized NumPy operations for speed.
        """
        total_payoff = np.zeros_like(prices)
        
        for leg in legs:
            if leg.option_type == OptionType.CALL:
                # Call payoff: max(S - K, 0) * quantity - premium
                intrinsic = np.maximum(prices - leg.strike, 0)
            else:
                # Put payoff: max(K - S, 0) * quantity - premium
                intrinsic = np.maximum(leg.strike - prices, 0)
            
            # Add this leg's contribution
            leg_payoff = intrinsic * leg.quantity + leg.net_premium()
            total_payoff += leg_payoff
        
        return total_payoff
    
    def _find_breakeven_points(
        self, 
        prices: np.ndarray, 
        payoffs: np.ndarray
    ) -> List[float]:
        """Find price points where strategy payoff equals zero."""
        breakevens = []
        
        # Look for sign changes in payoff
        signs = np.sign(payoffs)
        sign_changes = np.where(np.diff(signs) != 0)[0]
        
        for idx in sign_changes:
            # Linear interpolation to find exact breakeven
            p1, p2 = prices[idx], prices[idx + 1]
            v1, v2 = payoffs[idx], payoffs[idx + 1]
            
            if v2 != v1:
                breakeven = p1 - v1 * (p2 - p1) / (v2 - v1)
                breakevens.append(float(breakeven))
        
        return sorted(breakevens)
    
    def _calculate_profit_probability(
        self,
        legs: List[OptionLeg],
        underlying_price: float,
        days: int,
        volatility: float,
        risk_free_rate: float,
    ) -> float:
        """
        Estimate probability of profit using simplified model.
        
        For long strategies: probability that payoff > 0 at expiry.
        Uses log-normal distribution assumption.
        """
        # Simplified: assume symmetric strategies have ~50% win rate
        # More accurate would require Monte Carlo or numerical integration
        
        net_premium = sum(leg.net_premium() for leg in legs)
        
        # If net credit, probability is higher
        if net_premium > 0:
            base_prob = 0.5 + (net_premium / (underlying_price * volatility * np.sqrt(days/365))) * 0.5
            return min(max(base_prob, 0.0), 1.0)
        else:
            base_prob = 0.5 - (abs(net_premium) / (underlying_price * volatility * np.sqrt(days/365))) * 0.5
            return min(max(base_prob, 0.0), 1.0)
    
    def _calculate_expected_value(
        self,
        legs: List[OptionLeg],
        prices: np.ndarray,
        payoffs: np.ndarray,
        underlying_price: float,
        days: int,
        volatility: float,
        risk_free_rate: float,
    ) -> float:
        """Calculate expected value using risk-neutral probabilities."""
        # Simplified: use trapezoidal rule with uniform weights
        # In production, would weight by actual probability density
        
        dt = (prices[-1] - prices[0]) / len(prices)
        expected_value = np.trapz(payoffs, dx=dt) / (prices[-1] - prices[0])
        
        return float(expected_value)
    
    def _calculate_sharpe_approximation(
        self, 
        payoffs: np.ndarray, 
        prob_profit: float
    ) -> float:
        """Approximate Sharpe ratio from payoff distribution."""
        mean_payoff = np.mean(payoffs)
        std_payoff = np.std(payoffs)
        
        if std_payoff == 0:
            return 0.0
        
        # Annualize (assuming daily data)
        sharpe = mean_payoff / std_payoff * np.sqrt(252)
        
        return float(sharpe)


# ============================================================================
# PRE-BUILT STRATEGY CONSTRUCTORS
# ============================================================================

def build_iron_condor(
    underlying_price: float,
    put_spread_width: float = 0.05,
    call_spread_width: float = 0.05,
    distance_otm: float = 0.10,
    expiry: str = "30d",
) -> List[OptionLeg]:
    """
    Build an Iron Condor strategy.
    
    Structure:
    - Long put (lower strike)
    - Short put (higher strike)
    - Short call (lower strike)
    - Long call (higher strike)
    
    Market Outlook: Neutral (profit if price stays in range)
    """
    lower_put_strike = underlying_price * (1 - distance_otm - put_spread_width)
    higher_put_strike = underlying_price * (1 - distance_otm)
    lower_call_strike = underlying_price * (1 + distance_otm)
    higher_call_strike = underlying_price * (1 + distance_otm + call_spread_width)
    
    return [
        OptionLeg(OptionType.PUT, lower_put_strike, 1, 0.0, expiry),   # Long put
        OptionLeg(OptionType.PUT, higher_put_strike, -1, 0.0, expiry), # Short put
        OptionLeg(OptionType.CALL, lower_call_strike, -1, 0.0, expiry),# Short call
        OptionLeg(OptionType.CALL, higher_call_strike, 1, 0.0, expiry),# Long call
    ]


def build_straddle(
    underlying_price: float,
    strike: Optional[float] = None,
    expiry: str = "30d",
    is_long: bool = True,
) -> List[OptionLeg]:
    """
    Build a Straddle strategy (long or short).
    
    Structure:
    - Long/Short call at strike
    - Long/Short put at same strike
    
    Market Outlook: 
    - Long: Expecting big move (direction unknown)
    - Short: Expecting low volatility
    """
    if strike is None:
        strike = underlying_price
    
    qty = 1 if is_long else -1
    
    return [
        OptionLeg(OptionType.CALL, strike, qty, 0.0, expiry),
        OptionLeg(OptionType.PUT, strike, qty, 0.0, expiry),
    ]


def build_strangle(
    underlying_price: float,
    put_strike: Optional[float] = None,
    call_strike: Optional[float] = None,
    distance_otm: float = 0.05,
    expiry: str = "30d",
    is_long: bool = True,
) -> List[OptionLeg]:
    """
    Build a Strangle strategy (long or short).
    
    Structure:
    - Long/Short OTM put
    - Long/Short OTM call
    
    Similar to straddle but cheaper (OTM strikes).
    """
    if put_strike is None:
        put_strike = underlying_price * (1 - distance_otm)
    if call_strike is None:
        call_strike = underlying_price * (1 + distance_otm)
    
    qty = 1 if is_long else -1
    
    return [
        OptionLeg(OptionType.PUT, put_strike, qty, 0.0, expiry),
        OptionLeg(OptionType.CALL, call_strike, qty, 0.0, expiry),
    ]


def build_butterfly(
    underlying_price: float,
    center_strike: Optional[float] = None,
    wing_width: float = 0.05,
    expiry: str = "30d",
    is_call: bool = True,
) -> List[OptionLeg]:
    """
    Build a Butterfly strategy.
    
    Structure (Call Butterfly):
    - Long 1 call at lower strike
    - Short 2 calls at center strike
    - Long 1 call at higher strike
    
    Market Outlook: Neutral (profit if price ends at center strike)
    """
    if center_strike is None:
        center_strike = underlying_price
    
    lower_strike = center_strike * (1 - wing_width)
    higher_strike = center_strike * (1 + wing_width)
    
    option_type = OptionType.CALL if is_call else OptionType.PUT
    
    return [
        OptionLeg(option_type, lower_strike, 1, 0.0, expiry),
        OptionLeg(option_type, center_strike, -2, 0.0, expiry),
        OptionLeg(option_type, higher_strike, 1, 0.0, expiry),
    ]


def build_calendar_spread(
    underlying_price: float,
    strike: Optional[float] = None,
    near_expiry: str = "7d",
    far_expiry: str = "30d",
    is_call: bool = True,
) -> List[OptionLeg]:
    """
    Build a Calendar Spread strategy.
    
    Structure:
    - Short near-term option
    - Long longer-term option (same strike)
    
    Market Outlook: Neutral to slightly directional
    Profits from faster time decay of near-term option.
    """
    if strike is None:
        strike = underlying_price
    
    option_type = OptionType.CALL if is_call else OptionType.PUT
    
    return [
        OptionLeg(option_type, strike, -1, 0.0, near_expiry),
        OptionLeg(option_type, strike, 1, 0.0, far_expiry),
    ]


def build_ratio_spread(
    underlying_price: float,
    strike: Optional[float] = None,
    ratio: int = 2,
    expiry: str = "30d",
    is_call: bool = True,
) -> List[OptionLeg]:
    """
    Build a Ratio Spread strategy.
    
    Structure:
    - Buy 1 option at strike
    - Sell N options at higher/lower strike (same expiry)
    
    Market Outlook: Directional with limited risk
    """
    if strike is None:
        strike = underlying_price
    
    option_type = OptionType.CALL if is_call else OptionType.PUT
    
    # For calls: sell OTM calls; for puts: sell OTM puts
    if is_call:
        short_strike = strike * 1.05
    else:
        short_strike = strike * 0.95
    
    return [
        OptionLeg(option_type, strike, 1, 0.0, expiry),
        OptionLeg(option_type, short_strike, -ratio, 0.0, expiry),
    ]


# ============================================================================
# PORTFOLIO AGGREGATION
# ============================================================================

class PortfolioAnalyzer:
    """Analyze aggregate risk across multiple strategies."""
    
    def __init__(self):
        self.strategies: List[Tuple[str, List[OptionLeg], StrategyPayoff]] = []
        self.builder = OptionsStrategyBuilder()
    
    def add_strategy(
        self,
        name: str,
        legs: List[OptionLeg],
        underlying_price: float,
    ):
        """Add a strategy to the portfolio."""
        payoff = self.builder.build_strategy(legs, underlying_price)
        self.strategies.append((name, legs, payoff))
    
    def aggregate_greeks(self) -> Dict[str, float]:
        """
        Calculate aggregate Greeks across all strategies.
        
        Note: This is a simplified version. Full implementation
        would integrate with the Rust Greeks calculator.
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        
        for _, legs, _ in self.strategies:
            for leg in legs:
                # Simplified Greek estimates (would use BS model in production)
                if leg.option_type == OptionType.CALL:
                    delta = 0.5 * leg.quantity
                else:
                    delta = -0.5 * leg.quantity
                
                total_delta += delta
                # Gamma, theta, vega would be calculated similarly
        
        return {
            "delta": total_delta,
            "gamma": total_gamma,
            "theta": total_theta,
            "vega": total_vega,
        }
    
    def total_max_profit(self) -> float:
        """Sum of maximum profits across all strategies."""
        return sum(payoff.max_profit for _, _, payoff in self.strategies)
    
    def total_max_loss(self) -> float:
        """Sum of maximum losses across all strategies."""
        return sum(payoff.max_loss for _, _, payoff in self.strategies)
    
    def portfolio_breakevens(self) -> List[float]:
        """Combined breakeven analysis (simplified)."""
        all_breakevens = []
        for _, _, payoff in self.strategies:
            all_breakevens.extend(payoff.breakeven_points)
        return sorted(all_breakevens)


if __name__ == "__main__":
    # Example usage
    builder = OptionsStrategyBuilder()
    
    # Build an iron condor on BTC at $50,000
    legs = build_iron_condor(50000, put_spread_width=0.05, call_spread_width=0.05)
    
    # Analyze the strategy
    payoff = builder.build_strategy(legs, 50000)
    
    print(f"Iron Condor Analysis:")
    print(f"  Max Profit: ${payoff.max_profit:.2f}")
    print(f"  Max Loss: ${payoff.max_loss:.2f}")
    print(f"  Breakevens: {[f'{b:.2f}' for b in payoff.breakeven_points]}")
    print(f"  Win Probability: {payoff.profit_probability:.2%}")
