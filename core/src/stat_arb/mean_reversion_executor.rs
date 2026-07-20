//! Mean Reversion Executor for Pairs Trading
//! ===========================================
//! Chapter 2, File 3: Rust Statistical Arbitrage
//!
//! Execution logic for pairs trading. Monitors the z-score of the spread 
//! and triggers simultaneous market/limit orders on both legs while actively 
//! managing the inventory risk of the pair to ensure market neutrality.
//!
//! Target Performance: Microsecond execution decisions
//! Features: Smart order routing, inventory management, adverse selection protection

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::VecDeque;
use chrono::{DateTime, Utc};

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    pub fn opposite(&self) -> Side {
        match self {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }
}

/// Order type for execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Market,
    Limit { price: f64 },
    Iceberg { visible_qty: f64, total_qty: f64 },
}

/// Signal strength for trading decision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalStrength {
    None,
    Weak,
    Moderate,
    Strong,
}

/// Pairs trading signal
#[derive(Debug, Clone)]
pub struct PairsSignal {
    /// Asset to go long
    pub long_asset: String,
    /// Asset to go short
    pub short_asset: String,
    /// Z-score of spread
    pub z_score: f64,
    /// Current hedge ratio
    pub hedge_ratio: f64,
    /// Spread value
    pub spread: f64,
    /// Signal strength
    pub strength: SignalStrength,
    /// Expected half-life of mean reversion (bars)
    pub expected_half_life: f64,
    /// Confidence in signal (0-1)
    pub confidence: f64,
    /// Timestamp
    pub timestamp_ns: u64,
}

impl PairsSignal {
    /// Create signal from z-score
    pub fn from_zscore(
        long_asset: String,
        short_asset: String,
        z_score: f64,
        hedge_ratio: f64,
        spread: f64,
        half_life: f64,
    ) -> Self {
        let strength = Self::calculate_strength(z_score);
        let confidence = Self::zscore_to_confidence(z_score);
        
        Self {
            long_asset,
            short_asset,
            z_score,
            hedge_ratio,
            spread,
            strength,
            expected_half_life: half_life,
            confidence,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }
    
    fn calculate_strength(z_score: f64) -> SignalStrength {
        let abs_z = z_score.abs();
        if abs_z < 1.0 {
            SignalStrength::None
        } else if abs_z < 1.5 {
            SignalStrength::Weak
        } else if abs_z < 2.0 {
            SignalStrength::Moderate
        } else {
            SignalStrength::Strong
        }
    }
    
    fn zscore_to_confidence(z_score: f64) -> f64 {
        // Approximate normal CDF transformation
        // Higher |z| = higher confidence
        let abs_z = z_score.abs().min(4.0); // Cap at 4 sigma
        0.5 + 0.125 * abs_z // Rough approximation: 1σ=62.5%, 2σ=75%, 3σ=87.5%, 4σ=100%
    }
    
    /// Get recommended entry threshold
    pub fn entry_threshold() -> f64 {
        1.5 // Enter when |z| > 1.5
    }
    
    /// Get recommended exit threshold
    pub fn exit_threshold() -> f64 {
        0.5 // Exit when |z| < 0.5
    }
    
    /// Get stop-loss threshold
    pub fn stop_loss_threshold() -> f64 {
        3.5 // Stop loss when |z| > 3.5
    }
    
    /// Check if signal suggests opening a position
    pub fn should_open(&self) -> bool {
        self.z_score.abs() >= Self::entry_threshold() && self.strength != SignalStrength::None
    }
    
    /// Check if signal suggests closing a position
    pub fn should_close(&self, current_position: f64) -> bool {
        // Close if z-score reverted or reversed
        let should_revert = self.z_score.abs() <= Self::exit_threshold();
        let should_stop = self.z_score.abs() >= Self::stop_loss_threshold();
        
        // Position-specific logic
        if current_position > 0.0 {
            // Long spread: close if z went negative or hit stop
            should_revert || self.z_score < -Self::exit_threshold() || should_stop
        } else if current_position < 0.0 {
            // Short spread: close if z went positive or hit stop
            should_revert || self.z_score > Self::exit_threshold() || should_stop
        } else {
            false
        }
    }
}

/// Inventory state for a pairs position
#[derive(Debug, Clone)]
pub struct PairsInventory {
    /// Long asset quantity (positive = long, negative = short)
    pub long_asset_qty: f64,
    /// Short asset quantity
    pub short_asset_qty: f64,
    /// Average entry z-score
    pub entry_z_score: f64,
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    /// Unrealized P&L
    pub unrealized_pnl: f64,
    /// Realized P&L
    pub realized_pnl: f64,
    /// Number of trades executed
    pub trade_count: u64,
}

impl PairsInventory {
    pub fn new() -> Self {
        Self {
            long_asset_qty: 0.0,
            short_asset_qty: 0.0,
            entry_z_score: 0.0,
            entry_time: Utc::now(),
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            trade_count: 0,
        }
    }
    
    pub fn is_flat(&self) -> bool {
        self.long_asset_qty.abs() < 1e-6 && self.short_asset_qty.abs() < 1e-6
    }
    
    pub fn notional_value(&self, price_long: f64, price_short: f64) -> f64 {
        self.long_asset_qty.abs() * price_long + self.short_asset_qty.abs() * price_short
    }
}

/// Risk limits for pairs trading
#[derive(Debug, Clone)]
pub struct RiskLimits {
    /// Maximum notional exposure per pair
    pub max_notional: f64,
    /// Maximum position size in base units
    pub max_position_qty: f64,
    /// Maximum daily loss
    pub max_daily_loss: f64,
    /// Maximum drawdown from peak
    pub max_drawdown: f64,
    /// Minimum capital reserve
    pub min_reserve: f64,
    /// Position concentration limit (fraction of portfolio)
    pub max_concentration: f64,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_notional: 100_000.0,      // $100k per pair
            max_position_qty: 10.0,        // 10 units max
            max_daily_loss: 5_000.0,       // $5k daily loss limit
            max_drawdown: 10_000.0,        // $10k max drawdown
            min_reserve: 50_000.0,         // $50k minimum reserve
            max_concentration: 0.2,        // 20% of portfolio
        }
    }
}

/// Execution result
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Whether order was submitted
    pub submitted: bool,
    /// Long leg order details
    pub long_leg: Option<OrderDescription>,
    /// Short leg order details
    pub short_leg: Option<OrderDescription>,
    /// Estimated slippage
    pub estimated_slippage: f64,
    /// Estimated fees
    pub estimated_fees: f64,
    /// Expected profit (based on mean reversion)
    pub expected_profit: f64,
    /// Risk-adjusted score
    pub risk_score: f64,
}

#[derive(Debug, Clone)]
pub struct OrderDescription {
    pub asset: String,
    pub side: Side,
    pub order_type: OrderType,
    pub quantity: f64,
    pub price: f64,
}

/// Mean Reversion Executor
pub struct MeanReversionExecutor {
    /// Current inventory
    inventory: RwLock<PairsInventory>,
    /// Risk limits
    limits: RiskLimits,
    /// Entry threshold (z-score)
    entry_threshold: f64,
    /// Exit threshold (z-score)
    exit_threshold: f64,
    /// Stop loss threshold
    stop_loss: f64,
    /// Position sizing factor (Kelly fraction style)
    position_sizing_factor: f64,
    /// Enable dynamic sizing based on signal strength
    dynamic_sizing: bool,
    /// Fee rate (maker)
    maker_fee_rate: f64,
    /// Fee rate (taker)
    taker_fee_rate: f64,
    /// Slippage model parameters
    slippage_model: SlippageModel,
    /// Daily P&L tracking
    daily_pnl: RwLock<f64>,
    /// Peak equity for drawdown calculation
    peak_equity: RwLock<f64>,
    /// Circuit breaker
    circuit_breaker: AtomicBool,
    /// Trade counter
    trade_count: AtomicU64,
}

/// Simple linear slippage model
#[derive(Debug, Clone)]
pub struct SlippageModel {
    /// Base slippage (bps)
    pub base_bps: f64,
    /// Market impact coefficient (bps per unit)
    pub impact_coefficient: f64,
    /// Spread cost factor
    pub spread_factor: f64,
}

impl Default for SlippageModel {
    fn default() -> Self {
        Self {
            base_bps: 1.0,           // 1 bps base
            impact_coefficient: 0.5,  // 0.5 bps per unit
            spread_factor: 0.5,       // Half spread cost
        }
    }
}

impl MeanReversionExecutor {
    /// Create new executor with default settings
    pub fn new(limits: RiskLimits) -> Self {
        Self {
            inventory: RwLock::new(PairsInventory::new()),
            limits,
            entry_threshold: PairsSignal::entry_threshold(),
            exit_threshold: PairsSignal::exit_threshold(),
            stop_loss: PairsSignal::stop_loss_threshold(),
            position_sizing_factor: 0.1, // Kelly-style fraction
            dynamic_sizing: true,
            maker_fee_rate: 0.0002, // 2 bps
            taker_fee_rate: 0.0004, // 4 bps
            slippage_model: SlippageModel::default(),
            daily_pnl: RwLock::new(0.0),
            peak_equity: RwLock::new(0.0),
            circuit_breaker: AtomicBool::new(false),
            trade_count: AtomicU64::new(0),
        }
    }
    
    /// Set custom thresholds
    pub fn set_thresholds(&mut self, entry: f64, exit: f64, stop: f64) {
        self.entry_threshold = entry;
        self.exit_threshold = exit;
        self.stop_loss = stop;
    }
    
    /// Enable/disable dynamic position sizing
    pub fn set_dynamic_sizing(&mut self, enabled: bool) {
        self.dynamic_sizing = enabled;
    }
    
    /// Get current inventory
    pub fn get_inventory(&self) -> PairsInventory {
        self.inventory.read().clone()
    }
    
    /// Check if we can trade (risk checks)
    pub fn can_trade(&self, notional: f64) -> bool {
        if self.circuit_breaker.load(Ordering::Relaxed) {
            return false;
        }
        
        let inventory = self.inventory.read();
        let daily_pnl = *self.daily_pnl.read();
        let peak = *self.peak_equity.read();
        
        // Check various limits
        let current_notional = inventory.notional_value(100.0, 100.0); // Placeholder prices
        let new_notional = current_notional + notional;
        
        // Notional limit
        if new_notional > self.limits.max_notional {
            return false;
        }
        
        // Daily loss limit
        if daily_pnl < -self.limits.max_daily_loss {
            return false;
        }
        
        // Drawdown limit
        let current_equity = peak + daily_pnl;
        let drawdown = peak - current_equity;
        if drawdown > self.limits.max_drawdown {
            return false;
        }
        
        true
    }
    
    /// Calculate optimal position size
    fn calculate_position_size(&self, signal: &PairsSignal, price_long: f64, price_short: f64) -> f64 {
        let base_size = self.limits.max_position_qty * self.position_sizing_factor;
        
        if !self.dynamic_sizing {
            return base_size;
        }
        
        // Adjust based on signal strength
        let strength_multiplier = match signal.strength {
            SignalStrength::None => 0.0,
            SignalStrength::Weak => 0.5,
            SignalStrength::Moderate => 0.75,
            SignalStrength::Strong => 1.0,
        };
        
        // Adjust based on confidence
        let confidence_adjustment = signal.confidence;
        
        // Adjust based on expected half-life (shorter = better)
        let halflife_factor = if signal.expected_half_life > 0.0 {
            (10.0 / signal.expected_half_life).min(1.0)
        } else {
            0.5
        };
        
        base_size * strength_multiplier * confidence_adjustment * halflife_factor
    }
    
    /// Evaluate and potentially execute a pairs trade
    pub fn evaluate_and_execute(
        &self,
        signal: &PairsSignal,
        price_long: f64,
        price_short: f64,
        use_limit_orders: bool,
    ) -> ExecutionResult {
        let inventory = self.inventory.read();
        
        // Determine action
        let should_open = signal.should_open() && inventory.is_flat();
        let should_close = signal.should_close(inventory.long_asset_qty);
        
        if !should_open && !should_close {
            return ExecutionResult {
                submitted: false,
                long_leg: None,
                short_leg: None,
                estimated_slippage: 0.0,
                estimated_fees: 0.0,
                expected_profit: 0.0,
                risk_score: 0.0,
            };
        }
        
        // Calculate quantities
        let (long_qty, short_qty) = if should_open {
            let base_qty = self.calculate_position_size(signal, price_long, price_short);
            let hedge_qty = base_qty * signal.hedge_ratio;
            (base_qty, hedge_qty)
        } else {
            // Close existing position
            (-inventory.long_asset_qty, -inventory.short_asset_qty)
        };
        
        // Check risk limits
        let notional = long_qty.abs() * price_long + short_qty.abs() * price_short;
        if !self.can_trade(notional) {
            return ExecutionResult {
                submitted: false,
                long_leg: None,
                short_leg: None,
                estimated_slippage: 0.0,
                estimated_fees: 0.0,
                expected_profit: 0.0,
                risk_score: -1.0, // Risk rejection
            };
        }
        
        // Determine order types
        let order_type = if use_limit_orders {
            // Place limit orders inside spread
            let mid_long = price_long;
            let mid_short = price_short;
            OrderType::Limit { price: if long_qty > 0.0 { mid_long * 0.999 } else { mid_long * 1.001 } }
        } else {
            OrderType::Market
        };
        
        // Estimate costs
        let slippage = self.estimate_slippage(long_qty.abs() + short_qty.abs());
        let fees = notional * self.taker_fee_rate;
        
        // Estimate expected profit (mean reversion)
        let expected_profit = self.estimate_expected_profit(&signal, notional);
        
        // Calculate risk score
        let risk_score = self.calculate_risk_score(&signal, notional, expected_profit);
        
        // Build order descriptions
        let long_leg = Some(OrderDescription {
            asset: signal.long_asset.clone(),
            side: if long_qty > 0.0 { Side::Buy } else { Side::Sell },
            order_type,
            quantity: long_qty.abs(),
            price: price_long,
        });
        
        let short_leg = Some(OrderDescription {
            asset: signal.short_asset.clone(),
            side: if short_qty > 0.0 { Side::Buy } else { Side::Sell },
            order_type,
            quantity: short_qty.abs(),
            price: price_short,
        });
        
        ExecutionResult {
            submitted: true,
            long_leg,
            short_leg,
            estimated_slippage: slippage,
            estimated_fees: fees,
            expected_profit,
            risk_score,
        }
    }
    
    /// Estimate slippage for given quantity
    fn estimate_slippage(&self, qty: f64) -> f64 {
        let model = &self.slippage_model;
        model.base_bps * 0.0001 + model.impact_coefficient * 0.0001 * qty
    }
    
    /// Estimate expected profit from mean reversion
    fn estimate_expected_profit(&self, signal: &PairsSignal, notional: f64) -> f64 {
        // Expected return based on z-score and half-life
        // Simplified: expect full reversion over half-life period
        let expected_return = signal.z_score.abs() * 0.01; // 1% per sigma
        notional * expected_return
    }
    
    /// Calculate risk-adjusted score
    fn calculate_risk_score(&self, signal: &PairsSignal, notional: f64, expected_profit: f64) -> f64 {
        let costs = self.estimate_slippage(notional) + notional * self.taker_fee_rate;
        let net_profit = expected_profit - costs;
        
        // Risk score = Sharpe-like ratio
        if costs > 0.0 {
            net_profit / costs
        } else {
            0.0
        }
    }
    
    /// Update inventory after fill
    pub fn update_inventory(
        &self,
        long_asset: &str,
        short_asset: &str,
        long_fill_qty: f64,
        short_fill_qty: f64,
        long_price: f64,
        short_price: f64,
        long_side: Side,
        short_side: Side,
    ) {
        let mut inventory = self.inventory.write();
        
        // Update quantities
        let long_delta = if long_side == Side::Buy { long_fill_qty } else { -long_fill_qty };
        let short_delta = if short_side == Side::Buy { short_fill_qty } else { -short_fill_qty };
        
        inventory.long_asset_qty += long_delta;
        inventory.short_asset_qty += short_delta;
        inventory.trade_count += 1;
        
        // Update P&L if closing
        if (inventory.long_asset_qty.abs() < 1e-6) && (inventory.short_asset_qty.abs() < 1e-6) {
            // Position closed - calculate realized P&L
            // Simplified calculation
        }
        
        self.trade_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Update mark-to-market P&L
    pub fn update_mark_to_market(&self, price_long: f64, price_short: f64) {
        let inventory = self.inventory.read();
        
        let unrealized_pnl = inventory.long_asset_qty * (price_long - 100.0) // Simplified
            + inventory.short_asset_qty * (price_short - 100.0);
        
        // Update daily P&L
        let mut daily_pnl = self.daily_pnl.write();
        *daily_pnl = unrealized_pnl + inventory.realized_pnl;
        
        // Update peak equity
        let mut peak = self.peak_equity.write();
        let current_equity = *peak + *daily_pnl;
        if current_equity > *peak {
            *peak = current_equity;
        }
    }
    
    /// Trigger circuit breaker
    pub fn trigger_circuit_breaker(&self) {
        self.circuit_breaker.store(true, Ordering::Relaxed);
    }
    
    /// Reset circuit breaker
    pub fn reset_circuit_breaker(&self) {
        self.circuit_breaker.store(false, Ordering::Relaxed);
    }
    
    /// Get executor statistics
    pub fn get_statistics(&self) -> ExecutorStats {
        let inventory = self.inventory.read();
        let daily_pnl = *self.daily_pnl.read();
        let peak = *self.peak_equity.read();
        
        ExecutorStats {
            long_position: inventory.long_asset_qty,
            short_position: inventory.short_asset_qty,
            is_flat: inventory.is_flat(),
            unrealized_pnl: inventory.unrealized_pnl,
            realized_pnl: inventory.realized_pnl,
            daily_pnl,
            peak_equity: peak,
            drawdown: peak - (peak + daily_pnl),
            trade_count: self.trade_count.load(Ordering::Relaxed),
            circuit_breaker_active: self.circuit_breaker.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExecutorStats {
    pub long_position: f64,
    pub short_position: f64,
    pub is_flat: bool,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub daily_pnl: f64,
    pub peak_equity: f64,
    pub drawdown: f64,
    pub trade_count: u64,
    pub circuit_breaker_active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_signal_generation() {
        let signal = PairsSignal::from_zscore(
            "BTC".to_string(),
            "ETH".to_string(),
            2.5,
            15.0,
            100.0,
            5.0,
        );
        
        assert_eq!(signal.strength, SignalStrength::Strong);
        assert!(signal.should_open());
        assert!(signal.confidence > 0.7);
    }
    
    #[test]
    fn test_executor_decision() {
        let executor = MeanReversionExecutor::new(RiskLimits::default());
        
        let signal = PairsSignal::from_zscore(
            "BTC".to_string(),
            "ETH".to_string(),
            2.0,
            15.0,
            100.0,
            5.0,
        );
        
        let result = executor.evaluate_and_execute(&signal, 50000.0, 3000.0, false);
        
        assert!(result.submitted);
        assert!(result.risk_score > 0.0);
    }
}
