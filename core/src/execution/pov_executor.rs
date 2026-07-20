/// POV (Percentage of Volume) Execution Algorithm
/// ================================================
/// 
/// Implements Percentage of Volume execution strategy that dynamically slices
/// parent orders to participate at a fixed percentage of real-time market volume.
/// 
/// Key features:
/// - Adaptive participation rate based on market conditions
/// - Toxic flow detection to avoid adverse selection
/// - Minimum market impact through careful pacing
/// - Microsecond-level order management
/// 
/// Optimized for AMD Ryzen AI 5 with lock-free data structures.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};
use crate::execution::order_types::{OrderSide, OrderType, ParentOrder, ChildOrder};
use crate::market_data::order_book::OrderBookSnapshot;
use crate::risk::position_tracker::PositionState;

/// Configuration for POV execution
#[derive(Clone, Debug)]
pub struct POVConfig {
    /// Target participation rate (0.0 to 1.0)
    pub target_participation: f64,
    
    /// Minimum participation rate during low liquidity
    pub min_participation: f64,
    
    /// Maximum participation rate to avoid detection
    pub max_participation: f64,
    
    /// Lookback window for volume calculation (milliseconds)
    pub volume_window_ms: u64,
    
    /// Aggressiveness multiplier (higher = more aggressive)
    pub aggressiveness: f64,
    
    /// Maximum order size as fraction of ADTV
    pub max_order_adtv_fraction: f64,
    
    /// Minimum child order size (in base currency)
    pub min_child_size: f64,
    
    /// Toxic flow threshold (VPIN level to reduce participation)
    pub toxic_flow_threshold: f64,
}

impl Default for POVConfig {
    fn default() -> Self {
        Self {
            target_participation: 0.10,      // 10% of market volume
            min_participation: 0.02,         // 2% minimum
            max_participation: 0.25,         // 25% maximum
            volume_window_ms: 1000,          // 1 second lookback
            aggressiveness: 1.0,
            max_order_adtv_fraction: 0.01,   // 1% of ADTV max
            min_child_size: 0.001,           // Minimum order size
            toxic_flow_threshold: 0.7,       // VPIN threshold
        }
    }
}

/// Real-time volume tracker with bounded memory
pub struct VolumeTracker {
    /// Rolling volume buffer (circular)
    volumes: Vec<f64>,
    /// Timestamps for each volume bucket
    timestamps: Vec<u64>,
    /// Current index in circular buffer
    current_idx: usize,
    /// Total volume in window
    total_volume: f64,
    /// Window duration in milliseconds
    window_ms: u64,
    /// Last update timestamp
    last_update: u64,
}

impl VolumeTracker {
    pub fn new(window_ms: u64, buckets: usize) -> Self {
        Self {
            volumes: vec![0.0; buckets],
            timestamps: vec![0; buckets],
            current_idx: 0,
            total_volume: 0.0,
            window_ms,
            last_update: 0,
        }
    }
    
    /// Update with new trade volume
    #[inline]
    pub fn update(&mut self, timestamp: u64, volume: f64) {
        let bucket_duration = self.window_ms / self.volumes.len() as u64;
        let current_bucket = (timestamp / bucket_duration) as usize % self.volumes.len();
        
        // Check if we've moved to a new bucket
        if current_bucket != self.current_idx {
            // Subtract old volume before overwriting
            self.total_volume -= self.volumes[current_bucket];
            self.volumes[current_bucket] = 0.0;
            self.timestamps[current_bucket] = timestamp;
            self.current_idx = current_bucket;
        }
        
        self.volumes[current_bucket] += volume;
        self.total_volume += volume;
        self.last_update = timestamp;
    }
    
    /// Get total volume in the lookback window
    #[inline]
    pub fn get_window_volume(&self) -> f64 {
        // Expire old buckets
        let cutoff = self.last_update.saturating_sub(self.window_ms);
        let mut active_volume = 0.0;
        
        for i in 0..self.volumes.len() {
            if self.timestamps[i] >= cutoff {
                active_volume += self.volumes[i];
            }
        }
        
        active_volume
    }
    
    /// Get instantaneous volume rate (volume per second)
    #[inline]
    pub fn get_volume_rate(&self) -> f64 {
        let volume = self.get_window_volume();
        volume * 1000.0 / self.window_ms as f64
    }
    
    /// Reset tracker
    pub fn reset(&mut self) {
        self.volumes.fill(0.0);
        self.timestamps.fill(0);
        self.total_volume = 0.0;
        self.current_idx = 0;
    }
}

/// VPIN (Volume-Synchronized Probability of Informed Trading) calculator
/// Detects toxic flow to adjust participation rate
pub struct VPINCalculator {
    /// Buy volume buckets
    buy_volumes: Vec<f64>,
    /// Sell volume buckets
    sell_volumes: Vec<f64>,
    /// Current bucket index
    current_bucket: usize,
    /// Number of buckets
    n_buckets: usize,
    /// Current buy volume in active bucket
    current_buy: f64,
    /// Current sell volume in active bucket
    current_sell: f64,
    /// Bucket size (target volume per bucket)
    bucket_size: f64,
}

impl VPINCalculator {
    pub fn new(n_buckets: usize, bucket_size: f64) -> Self {
        Self {
            buy_volumes: vec![0.0; n_buckets],
            sell_volumes: vec![0.0; n_buckets],
            current_bucket: 0,
            n_buckets,
            current_buy: 0.0,
            current_sell: 0.0,
            bucket_size,
        }
    }
    
    /// Update with new trade
    #[inline]
    pub fn update(&mut self, volume: f64, is_buy: bool) {
        if is_buy {
            self.current_buy += volume;
        } else {
            self.current_sell += volume;
        }
        
        // Check if bucket is full
        let total_in_bucket = self.current_buy + self.current_sell;
        if total_in_bucket >= self.bucket_size {
            // Store completed bucket
            self.buy_volumes[self.current_bucket] = self.current_buy;
            self.sell_volumes[self.current_bucket] = self.current_sell;
            
            // Move to next bucket
            self.current_bucket = (self.current_bucket + 1) % self.n_buckets;
            self.current_buy = 0.0;
            self.current_sell = 0.0;
        }
    }
    
    /// Calculate current VPIN value
    #[inline]
    pub fn calculate_vpin(&self) -> f64 {
        let mut sum_abs_diff = 0.0;
        let mut sum_total = 0.0;
        
        for i in 0..self.n_buckets {
            let buy_vol = self.buy_volumes[i];
            let sell_vol = self.sell_volumes[i];
            let total = buy_vol + sell_vol;
            
            if total > 0.0 {
                sum_abs_diff += (buy_vol - sell_vol).abs();
                sum_total += total;
            }
        }
        
        if sum_total > 0.0 {
            sum_abs_diff / sum_total
        } else {
            0.0
        }
    }
}

/// POV Execution Engine
pub struct POVExecutor {
    /// Executor configuration
    config: POVConfig,
    
    /// Volume tracker for participation calculation
    volume_tracker: VolumeTracker,
    
    /// VPIN calculator for toxic flow detection
    vpin_calc: VPINCalculator,
    
    /// Remaining quantity to execute
    remaining_qty: AtomicU64,
    
    /// Executed quantity
    executed_qty: AtomicU64,
    
    /// Execution active flag
    is_active: AtomicBool,
    
    /// Start time of execution
    start_time: Instant,
    
    /// Symbol being traded
    symbol: String,
    
    /// Side of the order
    side: OrderSide,
    
    /// Last participation rate used
    last_participation: f64,
    
    /// Orders sent counter (for tracking)
    orders_sent: u64,
}

impl POVExecutor {
    /// Create new POV executor
    pub fn new(config: POVConfig, symbol: &str, side: OrderSide, total_qty: f64) -> Self {
        Self {
            config: config.clone(),
            volume_tracker: VolumeTracker::new(config.volume_window_ms, 10),
            vpin_calc: VPINCalculator::new(30, total_qty * 0.01),
            remaining_qty: AtomicU64::new((total_qty * 1e8) as u64),
            executed_qty: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
            start_time: Instant::now(),
            symbol: symbol.to_string(),
            side,
            last_participation: config.target_participation,
            orders_sent: 0,
        }
    }
    
    /// Process market data update and determine order action
    /// Returns optional child order to submit
    pub fn on_market_data(&mut self, snapshot: &OrderBookSnapshot, 
                          timestamp: u64) -> Option<ChildOrder> {
        if !self.is_active.load(Ordering::Relaxed) {
            return None;
        }
        
        // Update volume tracker with recent trades
        if let Some(recent_trades) = snapshot.recent_trades {
            for trade in recent_trades {
                self.volume_tracker.update(trade.timestamp, trade.volume);
                
                // Determine if buyer or seller initiated
                let is_buy = trade.price >= snapshot.mid_price();
                self.vpin_calc.update(trade.volume, is_buy);
            }
        }
        
        // Calculate current VPIN
        let vpin = self.vpin_calc.calculate_vpin();
        
        // Adjust participation based on toxic flow
        let adjusted_participation = self.calculate_participation(vpin, snapshot);
        self.last_participation = adjusted_participation;
        
        // Check if we should send an order
        if let Some(order_qty) = self.determine_order_size(adjusted_participation, snapshot) {
            self.orders_sent += 1;
            
            // Determine order price based on side and spread
            let price = self.determine_order_price(snapshot);
            
            let child = ChildOrder {
                symbol: self.symbol.clone(),
                side: self.side,
                order_type: OrderType::Limit,
                quantity: order_qty,
                price,
                time_in_force: crate::execution::order_types::TimeInForce::IOC,
                parent_id: None,
            };
            
            return Some(child);
        }
        
        None
    }
    
    /// Calculate adaptive participation rate
    fn calculate_participation(&self, vpin: f64, snapshot: &OrderBookSnapshot) -> f64 {
        let mut participation = self.config.target_participation;
        
        // Reduce participation when VPIN indicates toxic flow
        if vpin > self.config.toxic_flow_threshold {
            let reduction_factor = 1.0 - (vpin - self.config.toxic_flow_threshold);
            participation *= reduction_factor.max(self.config.min_participation / self.config.target_participation);
        }
        
        // Adjust based on spread (wider spread = lower participation)
        let spread_ratio = snapshot.spread() / snapshot.mid_price();
        if spread_ratio > 0.001 {
            participation *= 0.8;
        }
        
        // Apply aggressiveness multiplier
        participation *= self.config.aggressiveness;
        
        // Clamp to bounds
        participation.clamp(self.config.min_participation, self.config.max_participation)
    }
    
    /// Determine order size based on participation rate
    fn determine_order_size(&self, participation: f64, snapshot: &OrderBookSnapshot) -> Option<f64> {
        let remaining = self.remaining_qty.load(Ordering::Relaxed) as f64 / 1e8;
        
        if remaining < self.config.min_child_size {
            return None;
        }
        
        // Estimate available volume at best prices
        let available_volume = match self.side {
            OrderSide::Buy => snapshot.ask_volume_at_best(),
            OrderSide::Sell => snapshot.bid_volume_at_best(),
        };
        
        // Target volume based on participation
        let target_volume = available_volume * participation;
        
        if target_volume < self.config.min_child_size {
            return None;
        }
        
        // Don't exceed remaining quantity
        let order_qty = target_volume.min(remaining);
        
        // Check against ADTV limit
        let adtv_limit = snapshot.adtv() * self.config.max_order_adtv_fraction;
        let order_qty = order_qty.min(adtv_limit);
        
        if order_qty >= self.config.min_child_size {
            Some(order_qty)
        } else {
            None
        }
    }
    
    /// Determine optimal order price
    fn determine_order_price(&self, snapshot: &OrderBookSnapshot) -> f64 {
        match self.side {
            OrderSide::Buy => {
                // Try to cross half the spread for better fill probability
                let mid = snapshot.mid_price();
                let best_ask = snapshot.best_ask();
                // Price between mid and best ask
                mid + (best_ask - mid) * 0.5
            },
            OrderSide::Sell => {
                let mid = snapshot.mid_price();
                let best_bid = snapshot.best_bid();
                // Price between mid and best bid
                mid - (mid - best_bid) * 0.5
            },
        }
    }
    
    /// Notify of partial fill
    pub fn on_partial_fill(&self, filled_qty: f64) {
        let filled_u64 = (filled_qty * 1e8) as u64;
        self.executed_qty.fetch_add(filled_u64, Ordering::Relaxed);
        self.remaining_qty.fetch_sub(filled_u64, Ordering::Relaxed);
    }
    
    /// Notify of full fill
    pub fn on_fill(&self, filled_qty: f64) {
        self.on_partial_fill(filled_qty);
        
        let remaining = self.remaining_qty.load(Ordering::Relaxed);
        if remaining == 0 {
            self.is_active.store(false, Ordering::Relaxed);
        }
    }
    
    /// Cancel execution
    pub fn cancel(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }
    
    /// Get execution progress
    pub fn get_progress(&self) -> f64 {
        let executed = self.executed_qty.load(Ordering::Relaxed) as f64;
        let total = executed + self.remaining_qty.load(Ordering::Relaxed) as f64;
        if total > 0.0 {
            executed / total
        } else {
            0.0
        }
    }
    
    /// Get current participation rate
    pub fn current_participation(&self) -> f64 {
        self.last_participation
    }
    
    /// Get current VPIN
    pub fn current_vpin(&self) -> f64 {
        self.vpin_calc.calculate_vpin()
    }
    
    /// Check if execution is complete
    pub fn is_complete(&self) -> bool {
        !self.is_active.load(Ordering::Relaxed) || 
        self.remaining_qty.load(Ordering::Relaxed) == 0
    }
    
    /// Get elapsed time
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_volume_tracker() {
        let mut tracker = VolumeTracker::new(1000, 10);
        
        // Add some volume
        tracker.update(100, 50.0);
        tracker.update(200, 30.0);
        tracker.update(300, 20.0);
        
        assert!(tracker.get_window_volume() > 0.0);
    }
    
    #[test]
    fn test_vpin_calculation() {
        let mut vpin_calc = VPINCalculator::new(10, 100.0);
        
        // Add balanced buy/sell volume
        vpin_calc.update(50.0, true);
        vpin_calc.update(50.0, false);
        
        // Fill bucket
        vpin_calc.update(50.0, true);
        vpin_calc.update(50.0, false);
        
        let vpin = vpin_calc.calculate_vpin();
        assert!(vpin >= 0.0 && vpin <= 1.0);
    }
}
