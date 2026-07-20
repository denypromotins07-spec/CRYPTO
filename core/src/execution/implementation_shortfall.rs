/// Implementation Shortfall (IS) Execution Algorithm
/// ===================================================
/// 
/// Minimizes the total cost of execution by balancing:
/// 1. Market impact cost (trading too fast moves the price against you)
/// 2. Timing risk/alpha decay (trading too slow exposes you to adverse price moves)
/// 
/// Uses the Almgren-Chriss framework with real-time volatility adaptation.
/// Optimized for crypto markets with high volatility and 24/7 trading.
/// 
/// Key features:
/// - Dynamic trade-off between impact and timing risk
/// - Real-time volatility estimation using Parkinson estimator
/// - Alpha decay modeling for momentum signals
/// - Risk aversion parameter tuning
/// 
/// Designed for AMD Ryzen AI 5 with microsecond latency.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use std::f64::consts::SQRT_2;

use crate::execution::order_types::{OrderSide, OrderType, ChildOrder, TimeInForce};
use crate::market_data::order_book::OrderBookSnapshot;
use crate::microstructure::price_impact_model::PriceImpactModel;

/// Configuration for Implementation Shortfall execution
#[derive(Clone, Debug)]
pub struct ISConfig {
    /// Risk aversion parameter (lambda)
    /// Higher = more aggressive (prioritize speed over impact)
    /// Lower = more passive (prioritize impact over speed)
    pub risk_aversion: f64,
    
    /// Alpha decay rate (per millisecond)
    /// Expected price drift if signal is correct
    pub alpha_decay_rate: f64,
    
    /// Maximum execution time (milliseconds)
    pub max_execution_time_ms: u64,
    
    /// Minimum child order size
    pub min_child_size: f64,
    
    /// Volatility lookback window (milliseconds)
    pub vol_window_ms: u64,
    
    /// Price impact model parameters
    pub impact_model: ImpactParams,
    
    /// Urgency multiplier (for time-sensitive orders)
    pub urgency: f64,
}

impl Default for ISConfig {
    fn default() -> Self {
        Self {
            risk_aversion: 0.5,           // Moderate risk aversion
            alpha_decay_rate: 0.0001,     // 0.01% per ms decay
            max_execution_time_ms: 60000, // 1 minute max
            min_child_size: 0.001,
            vol_window_ms: 5000,          // 5 second vol window
            impact_model: ImpactParams::default(),
            urgency: 1.0,
        }
    }
}

/// Parameters for the price impact model
#[derive(Clone, Debug)]
pub struct ImpactParams {
    /// Temporary impact coefficient (eta)
    pub temporary_impact: f64,
    
    /// Permanent impact coefficient (gamma)
    pub permanent_impact: f64,
    
    /// Decay rate of temporary impact
    pub impact_decay: f64,
}

impl Default for ImpactParams {
    fn default() -> Self {
        Self {
            temporary_impact: 0.0001,    // 1 bp per unit volume
            permanent_impact: 0.00001,   // 0.1 bp permanent
            impact_decay: 0.1,           // Decay rate
        }
    }
}

/// Real-time volatility estimator using Parkinson method
/// More efficient than standard deviation for high-frequency data
pub struct VolatilityEstimator {
    /// High prices in window
    highs: Vec<f64>,
    /// Low prices in window
    lows: Vec<f64>,
    /// Window size
    window_size: usize,
    /// Current index
    current_idx: usize,
    /// Last estimated volatility (annualized)
    last_vol: f64,
    /// Scale factor for annualization
    annualization_factor: f64,
}

impl VolatilityEstimator {
    pub fn new(window_size: usize, sample_interval_ms: u64) -> Self {
        // Annualization: sqrt(trading_periods_per_year)
        // For crypto: 365 * 24 * 60 * 60 * 1000 / sample_interval_ms
        let periods_per_year = 365.0 * 24.0 * 3600.0 * 1e6 / sample_interval_ms as f64;
        let annualization_factor = periods_per_year.sqrt();
        
        Self {
            highs: Vec::with_capacity(window_size),
            lows: Vec::with_capacity(window_size),
            window_size,
            current_idx: 0,
            last_vol: 0.0,
            annualization_factor,
        }
    }
    
    /// Update with new high/low for current period
    #[inline]
    pub fn update_period(&mut self, high: f64, low: f64) {
        if self.highs.len() < self.window_size {
            self.highs.push(high);
            self.lows.push(low);
        } else {
            self.highs[self.current_idx] = high;
            self.lows[self.current_idx] = low;
            self.current_idx = (self.current_idx + 1) % self.window_size;
        }
        
        self.last_vol = self.calculate_parkinson_vol();
    }
    
    /// Calculate Parkinson volatility estimator
    /// σ² = (1 / (4 * ln(2) * N)) * Σ(ln(H_i/L_i))²
    fn calculate_parkinson_vol(&self) -> f64 {
        if self.highs.len() < 5 {
            return 0.0;
        }
        
        let n = self.highs.len() as f64;
        let mut sum_sq = 0.0;
        
        for i in 0..self.highs.len() {
            let h = self.highs[i];
            let l = self.lows[i];
            if h > 0.0 && l > 0.0 {
                let log_ratio = (h / l).ln();
                sum_sq += log_ratio * log_ratio;
            }
        }
        
        let variance = sum_sq / (4.0 * 2.0_f64.ln() * n);
        let vol = variance.sqrt() * self.annualization_factor;
        
        vol
    }
    
    /// Get current volatility estimate
    #[inline]
    pub fn get_volatility(&self) -> f64 {
        self.last_vol
    }
    
    /// Reset estimator
    pub fn reset(&mut self) {
        self.highs.clear();
        self.lows.clear();
        self.current_idx = 0;
        self.last_vol = 0.0;
    }
}

/// Optimal trading trajectory calculator (Almgren-Chriss)
pub struct TradingTrajectory {
    /// Total quantity to execute
    total_qty: f64,
    /// Number of intervals
    n_intervals: usize,
    /// Quantity per interval
    quantities: Vec<f64>,
    /// Current interval index
    current_interval: AtomicUsize,
}

impl TradingTrajectory {
    /// Calculate optimal trading schedule using Almgren-Chriss formula
    /// 
    /// The optimal strategy trades:
    /// - More aggressively when volatility is high
    /// - More aggressively when risk aversion is high
    /// - Less aggressively when market impact is high
    pub fn calculate(config: &ISConfig, total_qty: f64, 
                     volatility: f64, n_intervals: usize,
                     execution_time_ms: u64) -> Self {
        let dt = execution_time_ms as f64 / n_intervals as f64; // Time per interval (ms)
        
        // Almgren-Chriss optimal trading rate
        // κ = sqrt((λ * σ²) / η)
        // where λ = risk aversion, σ = volatility, η = temporary impact
        
        let lambda = config.risk_aversion * config.urgency;
        let sigma = volatility;
        let eta = config.impact_model.temporary_impact;
        
        // Handle edge cases
        if eta <= 0.0 || sigma <= 0.0 {
            // Equal distribution if we can't calculate
            let equal_qty = total_qty / n_intervals as f64;
            return Self {
                total_qty,
                n_intervals,
                quantities: vec![equal_qty; n_intervals],
                current_interval: AtomicUsize::new(0),
            };
        }
        
        let kappa = ((lambda * sigma * sigma) / eta).sqrt();
        
        // Generate trading schedule
        // q(t) = Q * (κ * cosh(κ(T-t))) / sinh(κT)
        let mut quantities = Vec::with_capacity(n_intervals);
        let total_time = execution_time_ms as f64;
        
        // Precompute constants
        let kappa_total = kappa * total_time;
        let sinh_kappa_total = kappa_total.sinh();
        
        if sinh_kappa_total.abs() < 1e-10 {
            // Degenerate case, use equal distribution
            let equal_qty = total_qty / n_intervals as f64;
            return Self {
                total_qty,
                n_intervals,
                quantities: vec![equal_qty; n_intervals],
                current_interval: AtomicUsize::new(0),
            };
        }
        
        let mut remaining = total_qty;
        for i in 0..n_intervals {
            let t = i as f64 * dt;
            let time_remaining = total_time - t;
            
            // Optimal quantity for this interval
            let kappa_remaining = kappa * time_remaining;
            let qty = total_qty * (kappa * kappa_remaining.cosh()) / sinh_kappa_total * dt / total_time;
            
            let qty = qty.min(remaining); // Don't exceed remaining
            quantities.push(qty);
            remaining -= qty;
        }
        
        // Add any remaining to last interval
        if remaining > 0.0 && !quantities.is_empty() {
            *quantities.last_mut().unwrap() += remaining;
        }
        
        Self {
            total_qty,
            n_intervals,
            quantities,
            current_interval: AtomicUsize::new(0),
        }
    }
    
    /// Get next quantity to trade
    pub fn next_quantity(&self) -> Option<f64> {
        let idx = self.current_interval.load(Ordering::Relaxed);
        if idx >= self.n_intervals {
            return None;
        }
        
        Some(self.quantities[idx])
    }
    
    /// Advance to next interval
    pub fn advance(&self) {
        self.current_interval.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get remaining quantity
    pub fn remaining_quantity(&self) -> f64 {
        let idx = self.current_interval.load(Ordering::Relaxed);
        if idx >= self.n_intervals {
            return 0.0;
        }
        
        self.quantities[idx..].iter().sum()
    }
    
    /// Reset trajectory
    pub fn reset(&self) {
        self.current_interval.store(0, Ordering::Relaxed);
    }
}

/// Implementation Shortfall Execution Engine
pub struct ISExecutor {
    /// Executor configuration
    config: ISConfig,
    
    /// Volatility estimator
    vol_estimator: VolatilityEstimator,
    
    /// Current trading trajectory
    trajectory: Option<TradingTrajectory>,
    
    /// Total quantity to execute
    total_qty: AtomicU64,
    
    /// Executed quantity
    executed_qty: AtomicU64,
    
    /// Average execution price
    avg_price: AtomicU64, // Stored as fixed point (price * 1e8)
    
    /// Execution active flag
    is_active: AtomicBool,
    
    /// Start time
    start_time: Instant,
    
    /// Symbol
    symbol: String,
    
    /// Side
    side: OrderSide,
    
    /// Initial mid price (for shortfall calculation)
    initial_mid_price: f64,
    
    /// Current unrealized shortfall
    current_shortfall: f64,
    
    /// Price impact model
    impact_model: PriceImpactModel,
}

impl ISExecutor {
    /// Create new IS executor
    pub fn new(config: ISConfig, symbol: &str, side: OrderSide, 
               total_qty: f64, initial_price: f64) -> Self {
        Self {
            config: config.clone(),
            vol_estimator: VolatilityEstimator::new(50, 100), // 50 samples, 100ms each
            trajectory: None,
            total_qty: AtomicU64::new((total_qty * 1e8) as u64),
            executed_qty: AtomicU64::new(0),
            avg_price: AtomicU64::new((initial_price * 1e8) as u64),
            is_active: AtomicBool::new(true),
            start_time: Instant::now(),
            symbol: symbol.to_string(),
            side,
            initial_mid_price: initial_price,
            current_shortfall: 0.0,
            impact_model: PriceImpactModel::new(
                config.impact_model.temporary_impact,
                config.impact_model.permanent_impact,
            ),
        }
    }
    
    /// Process market data and determine order action
    pub fn on_market_data(&mut self, snapshot: &OrderBookSnapshot,
                          timestamp: u64) -> Option<ChildOrder> {
        if !self.is_active.load(Ordering::Relaxed) {
            return None;
        }
        
        // Update volatility estimator
        let high = snapshot.best_ask();
        let low = snapshot.best_bid();
        self.vol_estimator.update_period(high, low);
        
        // Recalculate trajectory periodically based on current volatility
        let elapsed_ms = self.start_time.elapsed().as_millis() as u64;
        if elapsed_ms % 100 == 0 || self.trajectory.is_none() {
            self.recalculate_trajectory(elapsed_ms);
        }
        
        // Get next quantity from trajectory
        if let Some(ref traj) = self.trajectory {
            if let Some(qty) = traj.next_quantity() {
                if qty >= self.config.min_child_size {
                    // Determine optimal price
                    let price = self.determine_optimal_price(snapshot, qty);
                    
                    let child = ChildOrder {
                        symbol: self.symbol.clone(),
                        side: self.side,
                        order_type: OrderType::Limit,
                        quantity: qty,
                        price,
                        time_in_force: TimeInForce::IOC,
                        parent_id: None,
                    };
                    
                    // Advance trajectory after sending order
                    traj.advance();
                    
                    return Some(child);
                }
            }
        }
        
        None
    }
    
    /// Recalculate trading trajectory based on current conditions
    fn recalculate_trajectory(&mut self, elapsed_ms: u64) {
        let remaining_time = self.config.max_execution_time_ms.saturating_sub(elapsed_ms);
        if remaining_time < 100 {
            return; // Too little time left
        }
        
        let remaining_qty = self.total_qty.load(Ordering::Relaxed) as f64 / 1e8 
            - self.executed_qty.load(Ordering::Relaxed) as f64 / 1e8;
        
        if remaining_qty < self.config.min_child_size {
            return;
        }
        
        let volatility = self.vol_estimator.get_volatility();
        let n_intervals = (remaining_time / 100).max(5) as usize; // At least 5 intervals
        
        self.trajectory = Some(TradingTrajectory::calculate(
            &self.config,
            remaining_qty,
            volatility,
            n_intervals,
            remaining_time,
        ));
    }
    
    /// Determine optimal limit price for order
    fn determine_optimal_price(&self, snapshot: &OrderBookSnapshot, qty: f64) -> f64 {
        let mid = snapshot.mid_price();
        
        // Calculate expected impact
        let expected_impact = self.impact_model.estimate_immediate_impact(qty, &snapshot);
        
        match self.side {
            OrderSide::Buy => {
                // For buys: willing to pay up to mid + impact
                // But try to save half the spread
                let spread = snapshot.spread();
                let aggressive_price = mid + expected_impact;
                let passive_price = snapshot.best_bid() + spread * 0.3;
                
                // Blend based on urgency
                passive_price + (aggressive_price - passive_price) * self.config.urgency.min(1.0)
            },
            OrderSide::Sell => {
                // For sells: willing to accept down to mid - impact
                let spread = snapshot.spread();
                let aggressive_price = mid - expected_impact;
                let passive_price = snapshot.best_ask() - spread * 0.3;
                
                passive_price - (passive_price - aggressive_price) * self.config.urgency.min(1.0)
            },
        }
    }
    
    /// Notify of fill
    pub fn on_fill(&self, filled_qty: f64, fill_price: f64) {
        let filled_u64 = (filled_qty * 1e8) as u64;
        let price_u64 = (fill_price * 1e8) as u64;
        
        // Update executed quantity
        self.executed_qty.fetch_add(filled_u64, Ordering::Relaxed);
        
        // Update average price (atomic not ideal for this, but acceptable for monitoring)
        let executed = self.executed_qty.load(Ordering::Relaxed) as f64 / 1e8;
        let avg = self.avg_price.load(Ordering::Relaxed) as f64 / 1e8;
        let new_avg = (avg * (executed - filled_qty) + fill_price * filled_qty) / executed;
        self.avg_price.store((new_avg * 1e8) as u64, Ordering::Relaxed);
        
        // Check completion
        let remaining = self.total_qty.load(Ordering::Relaxed) as f64 / 1e8 - executed;
        if remaining <= 0.0 {
            self.is_active.store(false, Ordering::Relaxed);
        }
        
        // Update shortfall
        self.update_shortfall(fill_price);
    }
    
    /// Update implementation shortfall tracking
    fn update_shortfall(&self, fill_price: f64) {
        let executed = self.executed_qty.load(Ordering::Relaxed) as f64 / 1e8;
        let avg_price = self.avg_price.load(Ordering::Relaxed) as f64 / 1e8;
        
        // Shortfall = (avg_exec_price - decision_price) * qty for buys
        // Shortfall = (decision_price - avg_exec_price) * qty for sells
        self.current_shortfall = match self.side {
            OrderSide::Buy => (avg_price - self.initial_mid_price) * executed,
            OrderSide::Sell => (self.initial_mid_price - avg_price) * executed,
        };
    }
    
    /// Get current implementation shortfall (in base currency)
    pub fn get_shortfall(&self) -> f64 {
        self.current_shortfall
    }
    
    /// Get shortfall as basis points of notional
    pub fn get_shortfall_bps(&self) -> f64 {
        let executed = self.executed_qty.load(Ordering::Relaxed) as f64 / 1e8;
        let notional = executed * self.initial_mid_price;
        
        if notional > 0.0 {
            (self.current_shortfall / notional) * 10000.0
        } else {
            0.0
        }
    }
    
    /// Cancel execution
    pub fn cancel(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }
    
    /// Get execution progress (0.0 to 1.0)
    pub fn get_progress(&self) -> f64 {
        let executed = self.executed_qty.load(Ordering::Relaxed) as f64;
        let total = self.total_qty.load(Ordering::Relaxed) as f64;
        if total > 0.0 {
            executed / total
        } else {
            1.0
        }
    }
    
    /// Check if execution is complete
    pub fn is_complete(&self) -> bool {
        !self.is_active.load(Ordering::Relaxed)
    }
    
    /// Get current volatility estimate
    pub fn current_volatility(&self) -> f64 {
        self.vol_estimator.get_volatility()
    }
    
    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    /// Get remaining quantity
    pub fn remaining_qty(&self) -> f64 {
        let total = self.total_qty.load(Ordering::Relaxed) as f64 / 1e8;
        let executed = self.executed_qty.load(Ordering::Relaxed) as f64 / 1e8;
        total - executed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_volatility_estimator() {
        let mut vol_est = VolatilityEstimator::new(20, 100);
        
        // Simulate some price ranges
        for i in 0..20 {
            let base = 100.0 + (i as f64 * 0.1);
            vol_est.update_period(base + 0.5, base - 0.5);
        }
        
        let vol = vol_est.get_volatility();
        assert!(vol > 0.0);
    }
    
    #[test]
    fn test_trajectory_calculation() {
        let config = ISConfig::default();
        let traj = TradingTrajectory::calculate(&config, 1.0, 0.5, 10, 60000);
        
        let total: f64 = traj.quantities.iter().sum();
        assert!(total > 0.9); // Should be close to total quantity
    }
}
