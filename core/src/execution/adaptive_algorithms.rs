/// Adaptive Execution Algorithm
/// ==============================
/// 
/// Dynamically switches between aggressive (market order) and passive (limit order)
/// execution tactics based on real-time market conditions:
/// - Order book toxicity (VPIN)
/// - Spread dynamics
/// - Short-term alpha signals
/// - Volatility regime
/// 
/// Uses a state machine approach with fuzzy logic for smooth transitions.
/// Optimized for microsecond decision-making on AMD Ryzen AI 5.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicU8, Ordering};
use std::time::{Duration, Instant};

use crate::execution::order_types::{OrderSide, OrderType, ChildOrder, TimeInForce};
use crate::market_data::order_book::OrderBookSnapshot;
use crate::microstructure::cancellation_rates::CancellationTracker;

/// Execution mode/tactic
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ExecutionMode {
    /// Passive: Post limit orders, earn spread
    Passive = 0,
    
    /// Neutral: Mix of limit and market orders
    Neutral = 1,
    
    /// Aggressive: Use market orders, prioritize fill certainty
    Aggressive = 2,
    
    /// Sniper: Wait for specific conditions, then hit hard
    Sniper = 3,
}

impl ExecutionMode {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => ExecutionMode::Passive,
            1 => ExecutionMode::Neutral,
            2 => ExecutionMode::Aggressive,
            3 => ExecutionMode::Sniper,
            _ => ExecutionMode::Neutral,
        }
    }
}

/// Market condition scores (0.0 to 1.0)
#[derive(Clone, Debug, Default)]
pub struct MarketConditionScores {
    /// Toxicity score (high = toxic, avoid aggressive trading)
    pub toxicity: f64,
    
    /// Spread score (high = wide spread, favor passive)
    pub spread_wide: f64,
    
    /// Volatility score (high = high vol, be cautious)
    pub volatility: f64,
    
    /// Momentum score (positive = favorable momentum)
    pub momentum: f64,
    
    /// Liquidity score (high = deep book, can trade larger)
    pub liquidity: f64,
    
    /// Urgency score (high = need to trade now)
    pub urgency: f64,
}

/// Configuration for adaptive execution
#[derive(Clone, Debug)]
pub struct AdaptiveConfig {
    /// Minimum time between mode changes (milliseconds)
    pub min_mode_change_ms: u64,
    
    /// Hysteresis threshold for mode switching (prevents chattering)
    pub hysteresis: f64,
    
    /// Base aggression level (0.0 to 1.0)
    pub base_aggression: f64,
    
    /// Maximum order size as fraction of top-of-book
    pub max_order_tob_fraction: f64,
    
    /// Minimum size for passive orders
    pub min_passive_size: f64,
    
    /// Alpha signal decay (milliseconds)
    pub alpha_decay_ms: u64,
    
    /// VPIN toxicity threshold
    pub toxicity_threshold: f64,
    
    /// Spread threshold (as fraction of mid price)
    pub spread_threshold: f64,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            min_mode_change_ms: 50,       // 50ms minimum between changes
            hysteresis: 0.15,             // 15% hysteresis
            base_aggression: 0.5,         // Start neutral
            max_order_tob_fraction: 0.3,  // Max 30% of TOB
            min_passive_size: 0.001,
            alpha_decay_ms: 500,          // 500ms alpha half-life
            toxicity_threshold: 0.6,
            spread_threshold: 0.0005,     // 5 bps
        }
    }
}

/// Fuzzy logic state machine for mode selection
pub struct ModeStateMachine {
    /// Current mode
    current_mode: AtomicU8,
    
    /// Mode scores (accumulated evidence for each mode)
    passive_score: f64,
    neutral_score: f64,
    aggressive_score: f64,
    sniper_score: f64,
    
    /// Last mode change time
    last_change_time: Instant,
    
    /// Configuration
    config: AdaptiveConfig,
}

impl ModeStateMachine {
    pub fn new(config: AdaptiveConfig, initial_mode: ExecutionMode) -> Self {
        Self {
            current_mode: AtomicU8::new(initial_mode as u8),
            passive_score: 0.33,
            neutral_score: 0.34,
            aggressive_score: 0.33,
            sniper_score: 0.0,
            last_change_time: Instant::now(),
            config,
        }
    }
    
    /// Update scores based on market conditions and potentially switch mode
    pub fn update(&mut self, conditions: &MarketConditionScores) -> ExecutionMode {
        // Calculate mode scores using fuzzy logic rules
        
        // === PASSIVE MODE CONDITIONS ===
        // Favor passive when:
        // - Spread is wide (can earn more)
        // - Toxicity is high (avoid being picked off)
        // - Liquidity is good (orders will fill eventually)
        let passive_evidence = 
            conditions.spread_wide * 0.35 +
            conditions.toxicity * 0.30 +
            conditions.liquidity * 0.20 +
            (1.0 - conditions.urgency) * 0.15;
        
        // === AGGRESSIVE MODE CONDITIONS ===
        // Favor aggressive when:
        // - Spread is tight (less cost to cross)
        // - Toxicity is low (safe to trade)
        // - Momentum is favorable (alpha decaying)
        // - Urgency is high
        let aggressive_evidence =
            (1.0 - conditions.spread_wide) * 0.25 +
            (1.0 - conditions.toxicity) * 0.25 +
            conditions.momentum.abs() * 0.25 +
            conditions.urgency * 0.25;
        
        // === NEUTRAL MODE ===
        // Default when no strong signals
        let neutral_evidence = 0.5 - (passive_evidence + aggressive_evidence).abs() / 2.0;
        
        // === SNIPER MODE CONDITIONS ===
        // Activate when:
        // - Very low toxicity
        // - Strong momentum signal
        // - Good liquidity to absorb
        let sniper_evidence = if conditions.toxicity < 0.3 && 
                                 conditions.momentum.abs() > 0.7 &&
                                 conditions.liquidity > 0.6 {
            0.8
        } else {
            0.1
        };
        
        // Apply exponential smoothing to scores
        let alpha = 0.3; // Smoothing factor
        self.passive_score = (1.0 - alpha) * self.passive_score + alpha * passive_evidence;
        self.neutral_score = (1.0 - alpha) * self.neutral_score + alpha * neutral_evidence;
        self.aggressive_score = (1.0 - alpha) * self.aggressive_score + alpha * aggressive_evidence;
        self.sniper_score = (1.0 - alpha) * self.sniper_score + alpha * sniper_evidence;
        
        // Normalize scores
        let total = self.passive_score + self.neutral_score + self.aggressive_score + self.sniper_score;
        if total > 0.0 {
            self.passive_score /= total;
            self.neutral_score /= total;
            self.aggressive_score /= total;
            self.sniper_score /= total;
        }
        
        // Determine best mode
        let best_mode = self.find_best_mode();
        
        // Check if we should switch (with hysteresis)
        let current = ExecutionMode::from_u8(self.current_mode.load(Ordering::Relaxed));
        if self.should_switch(current, best_mode) {
            self.current_mode.store(best_mode as u8, Ordering::Relaxed);
            self.last_change_time = Instant::now();
        }
        
        best_mode
    }
    
    fn find_best_mode(&self) -> ExecutionMode {
        let mut best = ExecutionMode::Neutral;
        let mut best_score = self.neutral_score;
        
        if self.passive_score > best_score {
            best = ExecutionMode::Passive;
            best_score = self.passive_score;
        }
        
        if self.aggressive_score > best_score {
            best = ExecutionMode::Aggressive;
            best_score = self.aggressive_score;
        }
        
        if self.sniper_score > best_score {
            best = ExecutionMode::Sniper;
        }
        
        best
    }
    
    fn should_switch(&self, current: ExecutionMode, proposed: ExecutionMode) -> bool {
        // Don't switch too frequently
        if self.last_change_time.elapsed().as_millis() as u64 < self.config.min_mode_change_ms {
            return false;
        }
        
        // Same mode, no switch needed
        if current == proposed {
            return false;
        }
        
        // Get score difference with hysteresis
        let current_score = self.get_mode_score(current);
        let proposed_score = self.get_mode_score(proposed);
        
        // Need clear advantage to switch
        proposed_score > current_score + self.config.hysteresis
    }
    
    fn get_mode_score(&self, mode: ExecutionMode) -> f64 {
        match mode {
            ExecutionMode::Passive => self.passive_score,
            ExecutionMode::Neutral => self.neutral_score,
            ExecutionMode::Aggressive => self.aggressive_score,
            ExecutionMode::Sniper => self.sniper_score,
        }
    }
    
    /// Get current mode
    pub fn current_mode(&self) -> ExecutionMode {
        ExecutionMode::from_u8(self.current_mode.load(Ordering::Relaxed))
    }
    
    /// Get all scores for monitoring
    pub fn get_scores(&self) -> (f64, f64, f64, f64) {
        (self.passive_score, self.neutral_score, self.aggressive_score, self.sniper_score)
    }
    
    /// Force a specific mode (for emergency situations)
    pub fn force_mode(&self, mode: ExecutionMode) {
        self.current_mode.store(mode as u8, Ordering::Relaxed);
    }
}

/// Adaptive Execution Engine
pub struct AdaptiveExecutor {
    /// Executor configuration
    config: AdaptiveConfig,
    
    /// Mode state machine
    mode_machine: ModeStateMachine,
    
    /// Cancellation/spoofing tracker
    cancellation_tracker: CancellationTracker,
    
    /// Remaining quantity to execute
    remaining_qty: AtomicU64,
    
    /// Executed quantity
    executed_qty: AtomicU64,
    
    /// Execution active flag
    is_active: AtomicBool,
    
    /// Symbol
    symbol: String,
    
    /// Side
    side: OrderSide,
    
    /// Start time
    start_time: Instant,
    
    /// Orders sent counter
    orders_sent: u64,
    
    /// Last alpha signal value
    last_alpha: f64,
    
    /// Alpha signal timestamp
    alpha_timestamp: Instant,
}

impl AdaptiveExecutor {
    /// Create new adaptive executor
    pub fn new(config: AdaptiveConfig, symbol: &str, side: OrderSide, total_qty: f64) -> Self {
        Self {
            config: config.clone(),
            mode_machine: ModeStateMachine::new(config, ExecutionMode::Neutral),
            cancellation_tracker: CancellationTracker::new(100, 50),
            remaining_qty: AtomicU64::new((total_qty * 1e8) as u64),
            executed_qty: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
            start_time: Instant::now(),
            symbol: symbol.to_string(),
            side,
            orders_sent: 0,
            last_alpha: 0.0,
            alpha_timestamp: Instant::now(),
        }
    }
    
    /// Process market data and determine action
    pub fn on_market_data(&mut self, snapshot: &OrderBookSnapshot,
                          timestamp: u64) -> Option<ChildOrder> {
        if !self.is_active.load(Ordering::Relaxed) {
            return None;
        }
        
        // Update cancellation tracker
        if let Some(cancels) = snapshot.recent_cancellations {
            for cancel in cancels {
                self.cancellation_tracker.record_cancellation(
                    cancel.price, 
                    cancel.volume,
                    cancel.side,
                );
            }
        }
        
        // Build market condition scores
        let conditions = self.build_condition_scores(snapshot);
        
        // Update mode based on conditions
        let mode = self.mode_machine.update(&conditions);
        
        // Update alpha decay
        let alpha = self.get_current_alpha();
        
        // Determine order based on mode
        match mode {
            ExecutionMode::Passive => self.create_passive_order(snapshot, &conditions),
            ExecutionMode::Neutral => self.create_neutral_order(snapshot, &conditions),
            ExecutionMode::Aggressive => self.create_aggressive_order(snapshot, &conditions),
            ExecutionMode::Sniper => self.create_sniper_order(snapshot, &conditions, alpha),
        }
    }
    
    /// Build market condition scores from order book
    fn build_condition_scores(&self, snapshot: &OrderBookSnapshot) -> MarketConditionScores {
        let mid = snapshot.mid_price();
        let spread = snapshot.spread();
        
        // Toxicity from VPIN/cancellation analysis
        let toxicity = self.cancellation_tracker.get_spoofing_ratio();
        
        // Spread score (normalized)
        let spread_ratio = spread / mid;
        let spread_wide = (spread_ratio / self.config.spread_threshold).min(1.0);
        
        // Volatility from recent price movement
        let volatility = snapshot.volatility_1s().min(1.0);
        
        // Momentum from order flow imbalance
        let momentum = snapshot.order_flow_imbalance();
        
        // Liquidity from book depth
        let liquidity = (snapshot.total_visible_volume() / snapshot.adtv()).min(1.0);
        
        // Urgency based on remaining quantity and time
        let elapsed = self.start_time.elapsed().as_secs() as f64;
        let remaining = self.remaining_qty.load(Ordering::Relaxed) as f64 / 1e8;
        let urgency = if elapsed > 0.0 {
            (remaining / (elapsed + 1.0)).min(1.0)
        } else {
            0.5
        };
        
        MarketConditionScores {
            toxicity,
            spread_wide,
            volatility,
            momentum,
            liquidity,
            urgency,
        }
    }
    
    /// Get current alpha signal (decaying over time)
    fn get_current_alpha(&mut self) -> f64 {
        let elapsed_ms = self.alpha_timestamp.elapsed().as_millis() as f64;
        let decay_factor = (-elapsed_ms / self.config.alpha_decay_ms as f64).exp();
        self.last_alpha * decay_factor
    }
    
    /// Set new alpha signal
    pub fn set_alpha_signal(&mut self, alpha: f64) {
        self.last_alpha = alpha.clamp(-1.0, 1.0);
        self.alpha_timestamp = Instant::now();
    }
    
    /// Create passive (limit) order
    fn create_passive_order(&mut self, snapshot: &OrderBookSnapshot,
                           _conditions: &MarketConditionScores) -> Option<ChildOrder> {
        let remaining = self.remaining_qty.load(Ordering::Relaxed) as f64 / 1e8;
        if remaining < self.config.min_passive_size {
            return None;
        }
        
        // Place limit order inside the spread
        let mid = snapshot.mid_price();
        let spread = snapshot.spread();
        
        let price = match self.side {
            OrderSide::Buy => {
                // Bid above best bid but below mid
                let best_bid = snapshot.best_bid();
                best_bid + spread * 0.3
            },
            OrderSide::Sell => {
                // Ask below best ask but above mid
                let best_ask = snapshot.best_ask();
                best_ask - spread * 0.3
            },
        };
        
        // Size based on book depth
        let available = match self.side {
            OrderSide::Buy => snapshot.ask_volume_at_best(),
            OrderSide::Sell => snapshot.bid_volume_at_best(),
        };
        
        let qty = (available * 0.5).min(remaining).max(self.config.min_passive_size);
        
        self.orders_sent += 1;
        
        Some(ChildOrder {
            symbol: self.symbol.clone(),
            side: self.side,
            order_type: OrderType::Limit,
            quantity: qty,
            price,
            time_in_force: TimeInForce::GTC,
            parent_id: None,
        })
    }
    
    /// Create neutral (mixed) order
    fn create_neutral_order(&mut self, snapshot: &OrderBookSnapshot,
                           _conditions: &MarketConditionScores) -> Option<ChildOrder> {
        let remaining = self.remaining_qty.load(Ordering::Relaxed) as f64 / 1e8;
        if remaining < self.config.min_passive_size {
            return None;
        }
        
        // Use limit order at mid or slightly better
        let price = match self.side {
            OrderSide::Buy => snapshot.best_bid() + snapshot.spread() * 0.4,
            OrderSide::Sell => snapshot.best_ask() - snapshot.spread() * 0.4,
        };
        
        let qty = remaining * 0.2; // Trade 20% of remaining
        
        self.orders_sent += 1;
        
        Some(ChildOrder {
            symbol: self.symbol.clone(),
            side: self.side,
            order_type: OrderType::Limit,
            quantity: qty,
            price,
            time_in_force: TimeInForce::IOC,
            parent_id: None,
        })
    }
    
    /// Create aggressive (market) order
    fn create_aggressive_order(&mut self, snapshot: &OrderBookSnapshot,
                              _conditions: &MarketConditionScores) -> Option<ChildOrder> {
        let remaining = self.remaining_qty.load(Ordering::Relaxed) as f64 / 1e8;
        if remaining < self.config.min_passive_size {
            return None;
        }
        
        // Use market order for immediate fill
        let available = match self.side {
            OrderSide::Buy => snapshot.ask_volume_at_best(),
            OrderSide::Sell => snapshot.bid_volume_at_best(),
        };
        
        // Don't take more than available or configured max
        let qty = available.min(remaining * 0.3).min(
            snapshot.adtv() * self.config.max_order_tob_fraction
        );
        
        if qty < self.config.min_passive_size {
            return None;
        }
        
        self.orders_sent += 1;
        
        Some(ChildOrder {
            symbol: self.symbol.clone(),
            side: self.side,
            order_type: OrderType::Market,
            quantity: qty,
            price: 0.0, // Market order
            time_in_force: TimeInForce::IOC,
            parent_id: None,
        })
    }
    
    /// Create sniper order (wait for opportunity, then strike)
    fn create_sniper_order(&mut self, snapshot: &OrderBookSnapshot,
                          conditions: &MarketConditionScores,
                          alpha: f64) -> Option<ChildOrder> {
        // Only fire if alpha is strong and in our favor
        let alpha_threshold = 0.5;
        let favorable_alpha = match self.side {
            OrderSide::Buy => alpha > alpha_threshold,
            OrderSide::Sell => alpha < -alpha_threshold,
        };
        
        if !favorable_alpha || conditions.toxicity > 0.4 {
            return None; // Wait for better opportunity
        }
        
        let remaining = self.remaining_qty.load(Ordering::Relaxed) as f64 / 1e8;
        if remaining < self.config.min_passive_size {
            return None;
        }
        
        // Aggressive limit order that will likely fill immediately
        let price = match self.side {
            OrderSide::Buy => snapshot.best_ask(), // Cross spread
            OrderSide::Sell => snapshot.best_bid(),
        };
        
        let qty = remaining * 0.5; // Hit hard with 50%
        
        self.orders_sent += 1;
        
        Some(ChildOrder {
            symbol: self.symbol.clone(),
            side: self.side,
            order_type: OrderType::Limit,
            quantity: qty,
            price,
            time_in_force: TimeInForce::IOC,
            parent_id: None,
        })
    }
    
    /// Notify of fill
    pub fn on_fill(&self, filled_qty: f64) {
        let filled_u64 = (filled_qty * 1e8) as u64;
        self.executed_qty.fetch_add(filled_u64, Ordering::Relaxed);
        self.remaining_qty.fetch_sub(filled_u64, Ordering::Relaxed);
        
        let remaining = self.remaining_qty.load(Ordering::Relaxed);
        if remaining == 0 {
            self.is_active.store(false, Ordering::Relaxed);
        }
    }
    
    /// Cancel execution
    pub fn cancel(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }
    
    /// Get current mode
    pub fn current_mode(&self) -> ExecutionMode {
        self.mode_machine.current_mode()
    }
    
    /// Get execution progress
    pub fn get_progress(&self) -> f64 {
        let executed = self.executed_qty.load(Ordering::Relaxed) as f64;
        let total = executed + self.remaining_qty.load(Ordering::Relaxed) as f64;
        if total > 0.0 {
            executed / total
        } else {
            1.0
        }
    }
    
    /// Check if complete
    pub fn is_complete(&self) -> bool {
        !self.is_active.load(Ordering::Relaxed)
    }
    
    /// Get mode scores for monitoring
    pub fn get_mode_scores(&self) -> (f64, f64, f64, f64) {
        self.mode_machine.get_scores()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mode_transitions() {
        let config = AdaptiveConfig::default();
        let mut machine = ModeStateMachine::new(config.clone(), ExecutionMode::Neutral);
        
        // Test transition to passive with high toxicity
        let conditions = MarketConditionScores {
            toxicity: 0.8,
            spread_wide: 0.6,
            ..Default::default()
        };
        
        let mode = machine.update(&conditions);
        assert!(mode == ExecutionMode::Passive || mode == ExecutionMode::Neutral);
    }
}
