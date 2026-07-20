"""
Monte Carlo Simulation Engine Module

Monte Carlo simulation engine for backtesting. Generates thousands of randomized
equity curves (shuffling trade sequences) to calculate robust Value at Risk (VaR),
Expected Shortfall, and the probability of ruin.

Key features:
- Multiple randomization methods (shuffle, block bootstrap, parametric)
- Path generation with realistic constraints
- Risk metrics calculation (VaR, ES, max drawdown distribution)
- Probability of ruin estimation
- Confidence interval computation
- Memory-bounded operation for 8GB systems

Target: 10,000+ simulations per minute
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import warnings


@dataclass
class Trade:
    """Single trade record"""
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    quantity: float
    side: int  # 1 = long, -1 = short
    pnl: float
    return_pct: float
    duration_bars: int
    max_drawdown: float
    max_profit: float


@dataclass
class EquityCurve:
    """Equity curve from a simulation path"""
    values: np.ndarray
    returns: np.ndarray
    peak_values: np.ndarray
    drawdowns: np.ndarray
    max_drawdown: float
    final_value: float
    total_return: float
    
    def __len__(self):
        return len(self.values)


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation"""
    # Input parameters
    num_simulations: int
    initial_capital: float
    
    # Equity curves
    final_values: np.ndarray
    max_drawdowns: np.ndarray
    total_returns: np.ndarray
    
    # Risk metrics
    var_95: float
    var_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    
    # Statistics
    mean_final_value: float
    median_final_value: float
    std_final_value: float
    
    # Ruin statistics
    ruin_probability: float
    ruin_threshold: float
    
    # Confidence intervals
    ci_95_lower: float
    ci_95_upper: float
    ci_99_lower: float
    ci_99_upper: float
    
    # Percentiles
    percentiles: Dict[int, float]
    
    # Distribution stats
    skewness: float
    kurtosis: float
    
    # Best/worst paths
    best_path_idx: int
    worst_path_idx: int
    best_return: float
    worst_return: float


class RandomizationMethod:
    """Base class for randomization methods"""
    
    def randomize(self, trades: List[Trade], rng: np.random.Generator) -> List[Trade]:
        raise NotImplementedError


class ShuffleRandomization(RandomizationMethod):
    """Simple shuffle of trade order"""
    
    def randomize(self, trades: List[Trade], rng: np.random.Generator) -> List[Trade]:
        indices = rng.permutation(len(trades))
        return [trades[i] for i in indices]


class BlockBootstrap(RandomizationMethod):
    """Block bootstrap preserving some autocorrelation"""
    
    def __init__(self, block_size: int = 5):
        self.block_size = block_size
    
    def randomize(self, trades: List[Trade], rng: np.random.Generator) -> List[Trade]:
        n = len(trades)
        if n <= self.block_size:
            return trades.copy()
        
        # Create blocks
        num_blocks = (n + self.block_size - 1) // self.block_size
        blocks = []
        
        for i in range(0, n, self.block_size):
            block = trades[i:min(i + self.block_size, n)]
            blocks.append(block)
        
        # Resample blocks with replacement
        resampled = []
        block_indices = rng.choice(len(blocks), size=num_blocks, replace=True)
        
        for idx in block_indices:
            resampled.extend(blocks[idx])
        
        # Trim to original length
        return resampled[:n]


class ParametricRandomization(RandomizationMethod):
    """Parametric sampling based on trade statistics"""
    
    def __init__(self, fit_distribution: str = 'normal'):
        self.fit_distribution = fit_distribution
        self.mean_return: float = 0.0
        self.std_return: float = 1.0
        self.win_rate: float = 0.5
        self.avg_win: float = 1.0
        self.avg_loss: float = -1.0
    
    def fit(self, trades: List[Trade]) -> None:
        """Fit distribution parameters to historical trades"""
        if not trades:
            return
        
        returns = np.array([t.return_pct for t in trades])
        
        self.mean_return = np.mean(returns)
        self.std_return = np.std(returns)
        
        wins = [t.return_pct for t in trades if t.pnl > 0]
        losses = [t.return_pct for t in trades if t.pnl <= 0]
        
        self.win_rate = len(wins) / len(trades) if trades else 0.5
        self.avg_win = np.mean(wins) if wins else 1.0
        self.avg_loss = np.mean(losses) if losses else -1.0
    
    def randomize(self, trades: List[Trade], rng: np.random.Generator) -> List[Trade]:
        """Generate synthetic trades from fitted distribution"""
        n = len(trades)
        
        new_trades = []
        for i, orig_trade in enumerate(trades):
            is_win = rng.random() < self.win_rate
            
            if is_win:
                if self.fit_distribution == 'normal':
                    ret = rng.normal(self.avg_win, self.std_return)
                elif self.fit_distribution == 'lognormal':
                    ret = rng.lognormal(np.log(max(0.001, self.avg_win)), 0.5)
                else:
                    ret = rng.normal(self.avg_win, self.std_return)
            else:
                if self.fit_distribution == 'normal':
                    ret = rng.normal(self.avg_loss, self.std_return)
                else:
                    ret = rng.normal(self.avg_loss, self.std_return)
            
            # Create synthetic trade
            new_trade = Trade(
                entry_time=i,
                exit_time=i + 1,
                entry_price=100.0,
                exit_price=100.0 * (1 + ret),
                quantity=orig_trade.quantity,
                side=orig_trade.side,
                pnl=ret * orig_trade.quantity * 100,
                return_pct=ret,
                duration_bars=1,
                max_drawdown=min(0, ret),
                max_profit=max(0, ret),
            )
            new_trades.append(new_trade)
        
        return new_trades


class MonteCarloSimulator:
    """
    Main Monte Carlo simulation engine for backtesting analysis.
    
    Designed for memory efficiency and speed on systems with 8GB RAM limit.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_per_trade: float = 1.0,
        slippage_bps: float = 5.0,
        max_positions: int = 1,
        position_sizing: str = 'fixed',
        risk_per_trade: float = 0.02,
    ):
        self.initial_capital = initial_capital
        self.commission_per_trade = commission_per_trade
        self.slippage_bps = slippage_bps / 10000.0
        self.max_positions = max_positions
        self.position_sizing = position_sizing
        self.risk_per_trade = risk_per_trade
        
        # Results storage (memory-bounded)
        self.equity_curves: deque = deque(maxlen=100)  # Keep only subset
        self.final_values: List[float] = []
        self.max_drawdowns: List[float] = []
        
        # Random generator
        self.rng = np.random.default_rng()
    
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility"""
        self.rng = np.random.default_rng(seed)
    
    def _apply_costs(self, pnl: float, quantity: float, price: float) -> float:
        """Apply transaction costs to PnL"""
        # Commission
        cost = self.commission_per_trade * 2  # Entry + exit
        
        # Slippage
        slippage = self.slippage_bps * price * quantity * 2
        
        return pnl - cost - slippage
    
    def _calculate_position_size(self, capital: float, trade: Trade) -> float:
        """Calculate position size based on sizing method"""
        if self.position_sizing == 'fixed':
            return trade.quantity
        elif self.position_sizing == 'percent_risk':
            # Size based on risk percentage
            risk_amount = capital * self.risk_per_trade
            if trade.max_drawdown != 0:
                size = abs(risk_amount / (trade.max_drawdown * trade.entry_price))
            else:
                size = trade.quantity
            return min(size, capital / trade.entry_price)
        elif self.position_sizing == 'kelly':
            # Simplified Kelly criterion
            win_prob = 0.5  # Would need historical estimate
            avg_win = 1.0
            avg_loss = 1.0
            kelly_frac = win_prob - (1 - win_prob) / (avg_win / avg_loss)
            kelly_frac = max(0, min(kelly_frac, 0.25))  # Cap at quarter Kelly
            return capital * kelly_frac / trade.entry_price
        else:
            return trade.quantity
    
    def simulate_path(
        self,
        trades: List[Trade],
        apply_compounding: bool = True
    ) -> EquityCurve:
        """
        Simulate a single equity path from a sequence of trades.
        
        Args:
            trades: Ordered list of trades
            apply_compounding: Whether to compound returns
            
        Returns:
            EquityCurve object
        """
        capital = self.initial_capital
        values = [capital]
        peak = capital
        peaks = [peak]
        drawdowns = [0.0]
        returns = [0.0]
        
        for trade in trades:
            # Calculate position size
            qty = self._calculate_position_size(capital, trade)
            
            # Apply costs
            net_pnl = self._apply_costs(trade.pnl, qty, trade.entry_price)
            
            # Update capital
            if apply_compounding:
                capital += net_pnl
            else:
                capital = self.initial_capital + sum(
                    self._apply_costs(t.pnl, t.quantity, t.entry_price)
                    for t in trades[:trades.index(trade) + 1]
                )
            
            capital = max(0, capital)  # No negative capital
            
            # Record values
            values.append(capital)
            peak = max(peak, capital)
            peaks.append(peak)
            
            dd = (peak - capital) / peak if peak > 0 else 0
            drawdowns.append(dd)
            
            ret = (values[-1] - values[-2]) / values[-2] if values[-2] > 0 else 0
            returns.append(ret)
        
        return EquityCurve(
            values=np.array(values),
            returns=np.array(returns),
            peak_values=np.array(peaks),
            drawdowns=np.array(drawdowns),
            max_drawdown=max(drawdowns),
            final_value=values[-1],
            total_return=(values[-1] - self.initial_capital) / self.initial_capital,
        )
    
    def run_simulation(
        self,
        trades: List[Trade],
        num_simulations: int = 10000,
        randomization_method: RandomizationMethod = None,
        show_progress: bool = False,
    ) -> SimulationResult:
        """
        Run full Monte Carlo simulation.
        
        Args:
            trades: Historical trades to randomize
            num_simulations: Number of simulation paths
            randomization_method: Method for randomizing trades
            show_progress: Print progress updates
            
        Returns:
            SimulationResult with all metrics
        """
        if not trades:
            raise ValueError("No trades provided")
        
        if randomization_method is None:
            randomization_method = ShuffleRandomization()
        
        # Fit parametric method if needed
        if isinstance(randomization_method, ParametricRandomization):
            randomization_method.fit(trades)
        
        # Storage for results
        self.final_values = []
        self.max_drawdowns = []
        
        # Run simulations
        for i in range(num_simulations):
            # Randomize trade order
            randomized_trades = randomization_method.randomize(trades, self.rng)
            
            # Simulate path
            curve = self.simulate_path(randomized_trades)
            
            # Store results
            self.final_values.append(curve.final_value)
            self.max_drawdowns.append(curve.max_drawdown)
            
            # Keep subset of equity curves
            if i < self.equity_curves.maxlen:
                self.equity_curves.append(curve)
            
            if show_progress and (i + 1) % 1000 == 0:
                print(f"  Progress: {i + 1}/{num_simulations}")
        
        # Convert to arrays
        final_values_arr = np.array(self.final_values)
        max_drawdowns_arr = np.array(self.max_drawdowns)
        total_returns_arr = (final_values_arr - self.initial_capital) / self.initial_capital
        
        # Calculate risk metrics
        var_95 = np.percentile(total_returns_arr, 5)
        var_99 = np.percentile(total_returns_arr, 1)
        
        # Expected shortfall (average of worst cases)
        es_95_mask = total_returns_arr <= var_95
        es_99_mask = total_returns_arr <= var_99
        
        es_95 = np.mean(total_returns_arr[es_95_mask]) if np.any(es_95_mask) else var_95
        es_99 = np.mean(total_returns_arr[es_99_mask]) if np.any(es_99_mask) else var_99
        
        # Ruin probability (capital drops below threshold)
        ruin_threshold = self.initial_capital * 0.5  # 50% loss
        ruin_count = sum(1 for fv in final_values_arr if fv < ruin_threshold)
        ruin_probability = ruin_count / num_simulations
        
        # Confidence intervals
        ci_95 = np.percentile(final_values_arr, [2.5, 97.5])
        ci_99 = np.percentile(final_values_arr, [0.5, 99.5])
        
        # Percentiles
        percentiles = {
            p: float(np.percentile(final_values_arr, p))
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }
        
        # Distribution statistics
        skewness = float(self._calculate_skewness(total_returns_arr))
        kurtosis = float(self._calculate_kurtosis(total_returns_arr))
        
        # Best/worst paths
        best_idx = int(np.argmax(total_returns_arr))
        worst_idx = int(np.argmin(total_returns_arr))
        
        return SimulationResult(
            num_simulations=num_simulations,
            initial_capital=self.initial_capital,
            final_values=final_values_arr,
            max_drawdowns=max_drawdowns_arr,
            total_returns=total_returns_arr,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=es_95,
            expected_shortfall_99=es_99,
            mean_final_value=float(np.mean(final_values_arr)),
            median_final_value=float(np.median(final_values_arr)),
            std_final_value=float(np.std(final_values_arr)),
            ruin_probability=ruin_probability,
            ruin_threshold=ruin_threshold,
            ci_95_lower=ci_95[0],
            ci_95_upper=ci_95[1],
            ci_99_lower=ci_99[0],
            ci_99_upper=ci_99[1],
            percentiles=percentiles,
            skewness=skewness,
            kurtosis=kurtosis,
            best_path_idx=best_idx,
            worst_path_idx=worst_idx,
            best_return=float(total_returns_arr[best_idx]),
            worst_return=float(total_returns_arr[worst_idx]),
        )
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate sample skewness"""
        n = len(data)
        if n < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0.0
        
        skew = np.sum(((data - mean) / std) ** 3) * n / ((n - 1) * (n - 2))
        return float(skew)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis"""
        n = len(data)
        if n < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        if std == 0:
            return 0.0
        
        kurt = (np.sum(((data - mean) / std) ** 4) / n) - 3
        return float(kurt)
    
    def get_equity_curve_percentiles(self, percentiles: List[int] = [5, 25, 50, 75, 95]) -> Dict[int, np.ndarray]:
        """Get equity curve values at different percentiles over time"""
        if not self.equity_curves:
            return {}
        
        # Align all curves to same length
        max_len = max(len(ec.values) for ec in self.equity_curves)
        aligned = []
        
        for ec in self.equity_curves:
            if len(ec.values) < max_len:
                # Pad with last value
                padded = np.pad(ec.values, (0, max_len - len(ec.values)), mode='edge')
            else:
                padded = ec.values[:max_len]
            aligned.append(padded)
        
        aligned = np.array(aligned)
        
        result = {}
        for p in percentiles:
            result[p] = np.percentile(aligned, p, axis=0)
        
        return result
    
    def plot_results(self, result: SimulationResult, show_individual: int = 50) -> None:
        """
        Plot simulation results (requires matplotlib).
        
        Args:
            result: Simulation result to plot
            show_individual: Number of individual paths to show
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histogram of final values
        axes[0, 0].hist(result.final_values, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(result.mean_final_value, color='r', linestyle='--', label=f'Mean: ${result.mean_final_value:,.0f}')
        axes[0, 0].axvline(result.median_final_value, color='g', linestyle='--', label=f'Median: ${result.median_final_value:,.0f}')
        axes[0, 0].set_xlabel('Final Capital')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Final Values')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Max drawdown distribution
        axes[0, 1].hist(result.max_drawdowns, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].axvline(np.mean(result.max_drawdowns), color='r', linestyle='--', label=f'Mean: {np.mean(result.max_drawdowns):.2%}')
        axes[0, 1].set_xlabel('Maximum Drawdown')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Maximum Drawdowns')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sample equity curves
        num_to_show = min(show_individual, len(self.equity_curves))
        for i, ec in enumerate(list(self.equity_curves)[:num_to_show]):
            axes[1, 0].plot(ec.values, alpha=0.3, linewidth=0.5)
        
        axes[1, 0].set_xlabel('Trade Number')
        axes[1, 0].set_ylabel('Capital')
        axes[1, 0].set_title(f'Sample Equity Curves (n={num_to_show})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_returns = np.sort(result.total_returns)
        cum_prob = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
        axes[1, 1].plot(sorted_returns, cum_prob)
        axes[1, 1].axvline(result.var_95, color='r', linestyle='--', label=f'VaR 95%: {result.var_95:.2%}')
        axes[1, 1].axvline(0, color='k', linestyle='-', alpha=0.5)
        axes[1, 1].set_xlabel('Total Return')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Distribution of Returns')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Example usage demonstration
    print("Monte Carlo Simulation Engine")
    print("=" * 50)
    
    # Generate sample trades
    np.random.seed(42)
    n_trades = 500
    
    trades = []
    for i in range(n_trades):
        is_win = np.random.random() < 0.55  # 55% win rate
        if is_win:
            ret = np.random.exponential(0.02)  # Avg 2% win
        else:
            ret = -np.random.exponential(0.015)  # Avg 1.5% loss
        
        trade = Trade(
            entry_time=i,
            exit_time=i + 1,
            entry_price=100.0,
            exit_price=100.0 * (1 + ret),
            quantity=10,
            side=1,
            pnl=ret * 10 * 100,
            return_pct=ret,
            duration_bars=1,
            max_drawdown=min(0, ret),
            max_profit=max(0, ret),
        )
        trades.append(trade)
    
    print(f"\nGenerated {len(trades)} sample trades")
    
    # Run simulation
    simulator = MonteCarloSimulator(
        initial_capital=100000,
        commission_per_trade=1.0,
        slippage_bps=5.0,
        position_sizing='fixed',
    )
    
    simulator.set_seed(123)
    
    print("\nRunning Monte Carlo simulation (10,000 paths)...")
    result = simulator.run_simulation(
        trades,
        num_simulations=10000,
        randomization_method=ShuffleRandomization(),
        show_progress=True,
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("SIMULATION RESULTS")
    print("=" * 50)
    
    print(f"\nNumber of simulations: {result.num_simulations:,}")
    print(f"Initial capital: ${result.initial_capital:,.0f}")
    
    print(f"\nFinal Value Statistics:")
    print(f"  Mean:   ${result.mean_final_value:,.2f}")
    print(f"  Median: ${result.median_final_value:,.2f}")
    print(f"  Std Dev: ${result.std_final_value:,.2f}")
    
    print(f"\nRisk Metrics:")
    print(f"  VaR 95%:  {result.var_95:.2%}")
    print(f"  VaR 99%:  {result.var_99:.2%}")
    print(f"  ES 95%:   {result.expected_shortfall_95:.2%}")
    print(f"  ES 99%:   {result.expected_shortfall_99:.2%}")
    
    print(f"\nDrawdown Analysis:")
    print(f"  Mean Max DD:  {np.mean(result.max_drawdowns):.2%}")
    print(f"  Median Max DD: {np.median(result.max_drawdowns):.2%}")
    
    print(f"\nRuin Analysis:")
    print(f"  Ruin Probability: {result.ruin_probability:.2%}")
    print(f"  Ruin Threshold: ${result.ruin_threshold:,.0f}")
    
    print(f"\nConfidence Intervals:")
    print(f"  95% CI: [${result.ci_95_lower:,.2f}, ${result.ci_95_upper:,.2f}]")
    print(f"  99% CI: [${result.ci_99_lower:,.2f}, ${result.ci_99_upper:,.2f}]")
    
    print(f"\nDistribution Shape:")
    print(f"  Skewness:  {result.skewness:.3f}")
    print(f"  Kurtosis:  {result.kurtosis:.3f}")
    
    print(f"\nBest/Worst Paths:")
    print(f"  Best Return:  {result.best_return:.2%}")
    print(f"  Worst Return: {result.worst_return:.2%}")
    
    print("\n" + "=" * 50)
    print("Monte Carlo simulation complete!")
