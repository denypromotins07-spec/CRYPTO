//! Price Impact Model Module
//! 
//! Implements Kyle's Lambda and the Almgren-Chriss price impact model
//! to predict the exact price movement caused by incoming market orders
//! based on current book depth and historical trade aggressiveness.
//! 
//! Key models:
//! - Kyle's Lambda (linear price impact)
//! - Almgren-Chriss (temporary + permanent impact)
//! - Square-root impact law
//! - Order book decay dynamics
//! 
//! Target latency: < 2 microseconds per prediction

use std::sync::atomic::{AtomicU64, Ordering};

/// Parameters for Kyle's Lambda model
#[derive(Debug, Clone)]
pub struct KyleLambdaParams {
    /// Initial lambda estimate
    pub lambda_initial: f64,
    /// Learning rate for online lambda updates
    pub learning_rate: f64,
    /// Minimum lambda value (market resilience floor)
    pub lambda_min: f64,
    /// Maximum lambda value
    pub lambda_max: f64,
    /// Decay factor for old observations
    pub decay_factor: f64,
}

impl Default for KyleLambdaParams {
    fn default() -> Self {
        Self {
            lambda_initial: 1e-6,
            learning_rate: 0.001,
            lambda_min: 1e-8,
            lambda_max: 1e-4,
            decay_factor: 0.99,
        }
    }
}

/// Parameters for Almgren-Chriss model
#[derive(Debug, Clone)]
pub struct AlmgrenChrissParams {
    /// Temporary impact coefficient (epsilon)
    pub epsilon: f64,
    /// Permanent impact coefficient (gamma)
    pub gamma: f64,
    /// Market volatility (sigma)
    pub sigma: f64,
    /// Trading horizon in seconds
    pub trading_horizon: f64,
    /// Risk aversion parameter
    pub risk_aversion: f64,
}

impl Default for AlmgrenChrissParams {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            gamma: 0.05,
            sigma: 0.02,
            trading_horizon: 3600.0, // 1 hour
            risk_aversion: 0.5,
        }
    }
}

/// Represents a single trade for impact analysis
#[derive(Clone, Debug)]
pub struct TradeRecord {
    pub timestamp_ns: u64,
    pub price: i64,
    pub quantity: u64,
    pub side: TradeSide,
    pub was_aggressive: bool, // True if taker order
    pub mid_price_before: f64,
    pub mid_price_after: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Result of a price impact prediction
#[derive(Debug, Clone)]
pub struct PriceImpactPrediction {
    /// Expected immediate price impact in basis points
    pub immediate_impact_bps: f64,
    /// Expected permanent price impact in basis points
    pub permanent_impact_bps: f64,
    /// Expected temporary impact (reverts over time)
    pub temporary_impact_bps: f64,
    /// Time to half-reversion in milliseconds
    pub reversion_half_life_ms: f64,
    /// Confidence in the prediction (0.0 to 1.0)
    pub confidence: f64,
    /// Optimal execution quantity for minimal impact
    pub optimal_quantity: u64,
    /// Estimated slippage cost in quote currency
    pub estimated_slippage_cost: f64,
    /// Current Kyle's Lambda value
    pub current_lambda: f64,
}

impl Default for PriceImpactPrediction {
    fn default() -> Self {
        Self {
            immediate_impact_bps: 0.0,
            permanent_impact_bps: 0.0,
            temporary_impact_bps: 0.0,
            reversion_half_life_ms: 1000.0,
            confidence: 0.0,
            optimal_quantity: 0,
            estimated_slippage_cost: 0.0,
            current_lambda: 0.0,
        }
    }
}

/// Kyle's Lambda estimator for linear price impact
/// 
/// Model: ΔP = λ * Q + ε
/// Where λ is the price impact per unit quantity
pub struct KyleLambdaEstimator {
    /// Current lambda estimate
    lambda: f64,
    /// Running sum of quantity squared (for normalization)
    sum_q_squared: f64,
    /// Running sum of price_change * quantity
    sum_dp_q: f64,
    /// Number of observations
    observation_count: u64,
    /// Parameters
    params: KyleLambdaParams,
    /// Recent trades for rolling calculation
    recent_trades: Vec<TradeRecord>,
    max_recent_trades: usize,
}

impl KyleLambdaEstimator {
    pub fn new(params: KyleLambdaParams, max_recent_trades: usize) -> Self {
        Self {
            lambda: params.lambda_initial,
            sum_q_squared: 0.0,
            sum_dp_q: 0.0,
            observation_count: 0,
            params,
            recent_trades: Vec::with_capacity(max_recent_trades),
            max_recent_trades,
        }
    }

    /// Update lambda with a new trade observation
    #[inline]
    pub fn update(&mut self, trade: TradeRecord) {
        let dp = trade.mid_price_after - trade.mid_price_before;
        let signed_q = match trade.side {
            TradeSide::Buy => trade.quantity as f64,
            TradeSide::Sell => -(trade.quantity as f64),
        };

        // Apply decay to old observations
        if self.observation_count > 0 {
            self.sum_q_squared *= self.params.decay_factor;
            self.sum_dp_q *= self.params.decay_factor;
        }

        // Add new observation
        self.sum_q_squared += signed_q * signed_q;
        self.sum_dp_q += dp * signed_q;
        self.observation_count += 1;

        // Update lambda estimate
        if self.sum_q_squared > 0.0 {
            let new_lambda = self.sum_dp_q / self.sum_q_squared;
            
            // Online learning update with smoothing
            self.lambda = (1.0 - self.params.learning_rate) * self.lambda
                + self.params.learning_rate * new_lambda;
            
            // Clamp to reasonable bounds
            self.lambda = self.lambda.clamp(self.params.lambda_min, self.params.lambda_max);
        }

        // Store recent trade
        self.recent_trades.push(trade);
        if self.recent_trades.len() > self.max_recent_trades {
            self.recent_trades.remove(0);
        }
    }

    /// Predict price impact for a given quantity
    #[inline]
    pub fn predict_impact(&self, quantity: u64, side: TradeSide) -> f64 {
        let signed_q = match side {
            TradeSide::Buy => quantity as f64,
            TradeSide::Sell => -(quantity as f64),
        };
        
        self.lambda * signed_q
    }

    /// Get current lambda value
    #[inline]
    pub fn current_lambda(&self) -> f64 {
        self.lambda
    }

    /// Reset the estimator
    pub fn reset(&mut self) {
        self.lambda = self.params.lambda_initial;
        self.sum_q_squared = 0.0;
        self.sum_dp_q = 0.0;
        self.observation_count = 0;
        self.recent_trades.clear();
    }
}

/// Almgren-Chriss optimal execution model
/// 
/// Models both temporary and permanent price impact
/// Temporary impact: affects execution price but reverts
/// Permanent impact: shifts the market price permanently
pub struct AlmgrenChrissModel {
    params: AlmgrenChrissParams,
    /// Current market state
    current_mid_price: f64,
    /// Recent volatility estimate
    recent_volatility: f64,
    /// Volume clock
    volume_clock: f64,
}

impl AlmgrenChrissModel {
    pub fn new(params: AlmgrenChrissParams) -> Self {
        Self {
            params,
            current_mid_price: 0.0,
            recent_volatility: params.sigma,
            volume_clock: 0.0,
        }
    }

    /// Update current market state
    #[inline]
    pub fn update_state(&mut self, mid_price: f64, volume: f64) {
        self.current_mid_price = mid_price;
        self.volume_clock += volume;
    }

    /// Calculate optimal execution trajectory
    /// 
    /// Returns the optimal quantity to execute at each time step
    /// to minimize total cost (impact + risk)
    pub fn calculate_optimal_trajectory(
        &self,
        total_quantity: u64,
        num_steps: usize,
    ) -> Vec<u64> {
        let q_total = total_quantity as f64;
        let tau = self.params.trading_horizon / num_steps as f64;
        
        // Almgren-Chriss optimal trajectory formula
        // x(t) = X * (sinh(k*(T-t)) / sinh(k*T))
        // where k = sqrt(gamma * lambda / (2 * epsilon))
        
        let k = ((self.params.gamma * self.params.risk_aversion) 
            / (2.0 * self.params.epsilon)).sqrt();
        let sinh_kt = (k * self.params.trading_horizon).sinh();
        
        let mut trajectory = Vec::with_capacity(num_steps);
        let mut remaining = q_total;
        
        for i in 0..num_steps {
            let t = i as f64 * tau;
            let remaining_time = self.params.trading_horizon - t;
            
            if sinh_kt > 0.0 {
                let optimal_remaining = q_total * (k * remaining_time).sinh() / sinh_kt;
                let execute = (remaining - optimal_remaining).max(0.0);
                trajectory.push(execute as u64);
                remaining -= execute;
            } else {
                // Uniform execution if sinh is zero
                trajectory.push((q_total / num_steps as f64) as u64);
            }
        }
        
        // Execute any remaining quantity
        if remaining > 0.0 && !trajectory.is_empty() {
            *trajectory.last_mut().unwrap() += remaining as u64;
        }
        
        trajectory
    }

    /// Calculate total expected cost for executing a quantity
    pub fn calculate_expected_cost(
        &self,
        quantity: u64,
        execution_time_seconds: f64,
    ) -> f64 {
        let q = quantity as f64;
        let v = self.volume_clock / self.params.trading_horizon; // Average volume rate
        
        if v == 0.0 {
            return f64::INFINITY;
        }

        // Temporary impact cost
        let temp_impact = self.params.epsilon * (q / v).powi(2);
        
        // Permanent impact cost
        let perm_impact = self.params.gamma * q;
        
        // Risk cost (variance of execution price)
        let risk_cost = self.params.risk_aversion * 
            self.recent_volatility.powi(2) * execution_time_seconds * q.powi(2);
        
        temp_impact + perm_impact + risk_cost
    }
}

/// Square-root impact model (empirically observed in many markets)
/// 
/// Model: ΔP/P = σ * (Q/V)^δ
/// Where δ ≈ 0.5 (square root), σ is a coefficient, V is daily volume
pub struct SquareRootImpactModel {
    /// Impact coefficient
    sigma: f64,
    /// Exponent (typically close to 0.5)
    delta: f64,
    /// Reference daily volume
    daily_volume: f64,
    /// Current price
    current_price: f64,
}

impl SquareRootImpactModel {
    pub fn new(sigma: f64, delta: f64, daily_volume: f64, current_price: f64) -> Self {
        Self {
            sigma,
            delta,
            daily_volume,
            current_price,
        }
    }

    /// Update model parameters
    #[inline]
    pub fn update(&mut self, daily_volume: f64, current_price: f64) {
        self.daily_volume = daily_volume;
        self.current_price = current_price;
    }

    /// Predict price impact using square-root law
    #[inline]
    pub fn predict_impact_bps(&self, quantity: u64) -> f64 {
        if self.daily_volume == 0.0 || self.current_price == 0.0 {
            return 0.0;
        }

        let q_ratio = quantity as f64 / self.daily_volume;
        let impact_fraction = self.sigma * q_ratio.powf(self.delta);
        
        // Convert to basis points
        impact_fraction * 10000.0
    }

    /// Calibrate sigma from observed impacts
    pub fn calibrate_sigma(&mut self, observed_impacts: &[(u64, f64)]) {
        if observed_impacts.is_empty() {
            return;
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;

        for &(qty, impact_bps) in observed_impacts {
            let x = (qty as f64 / self.daily_volume).powf(self.delta);
            let y = impact_bps / 10000.0;
            sum_x += x;
            sum_y += y;
        }

        if sum_x > 0.0 {
            self.sigma = sum_y / sum_x;
        }
    }
}

/// Combined price impact predictor using multiple models
pub struct PriceImpactPredictor {
    kyle_estimator: KyleLambdaEstimator,
    ac_model: AlmgrenChrissModel,
    sr_model: SquareRootImpactModel,
    /// Weights for model combination
    kyle_weight: f64,
    ac_weight: f64,
    sr_weight: f64,
    /// Statistics
    pub predictions_made: AtomicU64,
    pub avg_prediction_error: f64,
    /// Order book depth cache for context
    best_bid_depth: f64,
    best_ask_depth: f64,
}

impl PriceImpactPredictor {
    pub fn new(
        kyle_params: KyleLambdaParams,
        ac_params: AlmgrenChrissParams,
        daily_volume: f64,
        current_price: f64,
    ) -> Self {
        Self {
            kyle_estimator: KyleLambdaEstimator::new(kyle_params.clone(), 1000),
            ac_model: AlmgrenChrissModel::new(ac_params),
            sr_model: SquareRootImpactModel::new(0.1, 0.5, daily_volume, current_price),
            kyle_weight: 0.4,
            ac_weight: 0.3,
            sr_weight: 0.3,
            predictions_made: AtomicU64::new(0),
            avg_prediction_error: 0.0,
            best_bid_depth: 0.0,
            best_ask_depth: 0.0,
        }
    }

    /// Record a trade for model learning
    #[inline]
    pub fn record_trade(
        &mut self,
        timestamp_ns: u64,
        price: i64,
        quantity: u64,
        side: TradeSide,
        mid_price_before: f64,
        mid_price_after: f64,
        was_aggressive: bool,
    ) {
        let trade = TradeRecord {
            timestamp_ns,
            price,
            quantity,
            side: side.clone(),
            was_aggressive,
            mid_price_before,
            mid_price_after,
        };

        if was_aggressive {
            self.kyle_estimator.update(trade.clone());
        }

        self.ac_model.update_state(mid_price_after, quantity as f64);
    }

    /// Update order book depth context
    #[inline]
    pub fn update_book_context(&mut self, bid_depth: f64, ask_depth: f64) {
        self.best_bid_depth = bid_depth;
        self.best_ask_depth = ask_depth;
    }

    /// Predict comprehensive price impact for a market order
    #[inline]
    pub fn predict_full_impact(
        &mut self,
        quantity: u64,
        side: TradeSide,
        current_price: f64,
    ) -> PriceImpactPrediction {
        let mut prediction = PriceImpactPrediction::default();
        prediction.current_lambda = self.kyle_estimator.current_lambda();

        // Kyle's Lambda prediction (immediate linear impact)
        let kyle_impact = self.kyle_estimator.predict_impact(quantity, side.clone());
        let kyle_impact_bps = (kyle_impact / current_price) * 10000.0;

        // Square-root model prediction
        let sr_impact_bps = self.sr_model.predict_impact_bps(quantity);

        // Adjust based on current book depth
        let depth_factor = match side {
            TradeSide::Buy => {
                if self.best_ask_depth > 0.0 {
                    (quantity as f64 / self.best_ask_depth).min(1.0)
                } else {
                    1.0
                }
            }
            TradeSide::Sell => {
                if self.best_bid_depth > 0.0 {
                    (quantity as f64 / self.best_bid_depth).min(1.0)
                } else {
                    1.0
                }
            }
        };

        // Combine predictions
        let base_impact = self.kyle_weight * kyle_impact_bps.abs()
            + self.sr_weight * sr_impact_bps;
        
        prediction.immediate_impact_bps = base_impact * depth_factor;
        
        // Permanent impact (smaller than temporary)
        prediction.permanent_impact_bps = base_impact * 0.3 * depth_factor;
        
        // Temporary impact (reverts)
        prediction.temporary_impact_bps = prediction.immediate_impact_bps 
            - prediction.permanent_impact_bps;

        // Reversion half-life (empirical: larger orders take longer to revert)
        prediction.reversion_half_life_ms = 500.0 + (quantity as f64 * 0.1).min(5000.0);

        // Confidence based on recent prediction accuracy and data quality
        let data_confidence = (self.kyle_estimator.observation_count as f64 / 100.0).min(1.0);
        prediction.confidence = data_confidence * (1.0 - self.avg_prediction_error.min(1.0));

        // Optimal quantity (split into smaller orders if too large)
        let optimal_single_order = (self.best_ask_depth * 0.1) as u64;
        prediction.optimal_quantity = optimal_single_order.max(1);

        // Estimated slippage cost
        prediction.estimated_slippage_cost = 
            (prediction.immediate_impact_bps / 10000.0) * current_price * quantity as f64;

        self.predictions_made.fetch_add(1, Ordering::Relaxed);

        prediction
    }

    /// Update model weights based on recent performance
    pub fn adapt_weights(&mut self, actual_impacts: &[f64], predicted_impacts: &[f64]) {
        if actual_impacts.len() != predicted_impacts.len() || actual_impacts.is_empty() {
            return;
        }

        // Calculate errors for each model component
        let mut errors = [0.0; 3];
        let n = actual_impacts.len() as f64;

        for i in 0..actual_impacts.len() {
            errors[0] += (actual_impacts[i] - predicted_impacts[i]).abs();
        }

        // Simple adaptive weighting: reduce weight on high-error components
        let total_error: f64 = errors.iter().sum();
        if total_error > 0.0 {
            // Inverse error weighting
            let inv_errors: Vec<f64> = errors.iter()
                .map(|&e| if e < 0.001 { 1000.0 } else { 1.0 / e })
                .collect();
            
            let sum_inv: f64 = inv_errors.iter().sum();
            if sum_inv > 0.0 {
                self.kyle_weight = inv_errors[0] / sum_inv;
                self.sr_weight = inv_errors[1] / sum_inv;
                // Normalize
                let total = self.kyle_weight + self.sr_weight;
                self.kyle_weight /= total;
                self.sr_weight /= total;
                self.ac_weight = 1.0 - self.kyle_weight - self.sr_weight;
            }
        }

        // Update average error
        self.avg_prediction_error = total_error / n;
    }

    /// Get recommended execution strategy for large orders
    pub fn recommend_execution_strategy(
        &self,
        total_quantity: u64,
        urgency: f64, // 0.0 (patient) to 1.0 (urgent)
    ) -> ExecutionStrategy {
        let price = self.sr_model.current_price;
        let full_impact = self.kyle_estimator.current_lambda() * total_quantity as f64;
        
        // If impact is small, execute immediately
        if full_impact.abs() < price * 0.001 {
            return ExecutionStrategy::Immediate(total_quantity);
        }

        // Otherwise, split based on urgency
        let num_slices = match urgency {
            u if u >= 0.8 => 1,
            u if u >= 0.5 => 5,
            u if u >= 0.2 => 10,
            _ => 20,
        };

        let slice_quantity = total_quantity / num_slices as u64;
        let interval_ms = (self.ac_model.params.trading_horizon * 1000.0 / num_slices as f64) as u64;

        ExecutionStrategy::Sliced {
            slice_quantity,
            num_slices,
            interval_ms,
        }
    }
}

/// Execution strategy recommendation
#[derive(Debug, Clone)]
pub enum ExecutionStrategy {
    /// Execute entire quantity immediately
    Immediate(u64),
    /// Split into slices with intervals
    Sliced {
        slice_quantity: u64,
        num_slices: usize,
        interval_ms: u64,
    },
    /// Use VWAP-style execution
    Vwap {
        total_quantity: u64,
        duration_minutes: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kyle_lambda_estimation() {
        let params = KyleLambdaParams::default();
        let mut estimator = KyleLambdaEstimator::new(params, 100);

        // Simulate trades with known impact
        for i in 0..100 {
            let qty = 1000 + (i % 10) as u64 * 100;
            let mid_before = 50000.0;
            let mid_after = mid_before + 0.01 * qty as f64; // Known lambda = 0.01

            estimator.update(TradeRecord {
                timestamp_ns: i * 1000000,
                price: mid_after as i64,
                quantity: qty,
                side: TradeSide::Buy,
                was_aggressive: true,
                mid_price_before: mid_before,
                mid_price_after: mid_after,
            });
        }

        // Lambda should converge near 0.01
        assert!(estimator.current_lambda() > 0.005);
        assert!(estimator.current_lambda() < 0.02);
    }

    #[test]
    fn test_square_root_impact() {
        let model = SquareRootImpactModel::new(0.1, 0.5, 1000000.0, 50000.0);
        
        let impact_small = model.predict_impact_bps(1000);
        let impact_large = model.predict_impact_bps(10000);
        
        // Larger orders should have more impact
        assert!(impact_large > impact_small);
        
        // Impact should scale roughly with square root
        let ratio = impact_large / impact_small;
        let expected_ratio = (10.0).sqrt();
        assert!((ratio - expected_ratio).abs() < 0.5);
    }
}
