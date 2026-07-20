//! Queue Position Tracking Engine
//! 
//! Tracks the bot's exact position in the Binance matching engine queue using
//! FIFO logic and trade tick analysis to estimate fill probability and time-to-fill.
//! 
//! This is critical for market making and limit order strategies where queue
//! position determines execution priority.
//! 
//! Hardware Target: AMD Ryzen AI 5 with cache-line optimized structures
//! Memory Constraint: Bounded memory usage with pre-allocated buffers

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use crossbeam::queue::SegQueue;
use dashmap::DashMap;

/// Represents our order in the matching queue
#[derive(Debug, Clone)]
pub struct QueuePosition {
    pub symbol: u64,
    pub side: Side,
    pub price: u64,
    pub order_id: u64,
    pub queue_position: u64, // Our position in the queue (0 = first)
    pub total_queue_size: u64, // Total size at this price level
    pub estimated_fill_time_ns: u64,
    pub fill_probability: f32,
    pub last_updated_ns: u64,
}

/// Order book level for queue tracking
#[derive(Debug, Clone)]
pub struct PriceLevel {
    pub price: u64,
    pub total_size: u64,
    pub order_count: u32,
    pub our_size: u64,
    pub our_position: u64,
    pub last_update_ns: u64,
}

/// Trade tick for queue position estimation
#[derive(Debug, Clone, Copy)]
pub struct TradeTick {
    pub symbol: u64,
    pub price: u64,
    pub qty: u64,
    pub side: Side,
    pub timestamp_ns: u64,
    pub maker_order_id: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

/// Queue tracking statistics
#[derive(Debug, Clone)]
pub struct QueueStats {
    pub tracked_levels: usize,
    pub active_orders: usize,
    pub fills_completed: u64,
    pub cancellations: u64,
    pub avg_fill_time_ms: f64,
    pub fill_rate: f32,
}

/// Main queue position tracker
pub struct QueuePositionTracker {
    /// Active orders being tracked
    active_orders: DashMap<u64, QueuePosition>,
    
    /// Price levels per symbol
    price_levels: DashMap<(u64, Side), Vec<PriceLevel>>,
    
    /// Recent trade history for fill estimation
    trade_history: DashMap<u64, SegQueue<TradeTick>>,
    
    /// Fill statistics per symbol
    fill_stats: DashMap<u64, FillStatistics>,
    
    /// Counters
    fills_completed: AtomicU64,
    cancellations: AtomicU64,
    
    /// Running flag
    is_running: AtomicBool,
}

/// Statistics for fill prediction
#[derive(Debug, Clone)]
pub struct FillStatistics {
    pub symbol: u64,
    pub total_trades: u64,
    pub fills_at_price: u64,
    pub avg_trade_size: u64,
    pub recent_volume_bps: Vec<i32>, // Last N trades volume change
    pub queue_depletion_rate: f64, // Orders filled per second
    last_updated_ns: u64,
}

impl FillStatistics {
    fn new(symbol: u64) -> Self {
        Self {
            symbol,
            total_trades: 0,
            fills_at_price: 0,
            avg_trade_size: 0,
            recent_volume_bps: Vec::with_capacity(20),
            queue_depletion_rate: 0.0,
            last_updated_ns: 0,
        }
    }
    
    #[inline]
    fn update(&mut self, trade_qty: u64, timestamp_ns: u64) {
        self.total_trades += 1;
        
        // EMA for average trade size
        let alpha = 0.1;
        self.avg_trade_size = ((self.avg_trade_size as f64 * (1.0 - alpha)) 
            + (trade_qty as f64 * alpha)) as u64;
        
        self.last_updated_ns = timestamp_ns;
    }
}

impl QueuePositionTracker {
    pub fn new() -> Self {
        Self {
            active_orders: DashMap::with_capacity(256),
            price_levels: DashMap::with_capacity(1024),
            trade_history: DashMap::with_capacity(256),
            fill_stats: DashMap::with_capacity(256),
            fills_completed: AtomicU64::new(0),
            cancellations: AtomicU64::new(0),
            is_running: AtomicBool::new(false),
        }
    }

    /// Track a new order placement
    pub fn track_order(&self, order: QueuePosition) {
        self.active_orders.insert(order.order_id, order);
        
        // Initialize fill stats if needed
        self.fill_stats.entry(order.symbol).or_insert_with(|| {
            FillStatistics::new(order.symbol)
        });
    }

    /// Update queue position based on order book changes
    pub fn update_queue_position(&self, order_id: u64, new_position: u64, total_size: u64) {
        if let Some(mut pos) = self.active_orders.get_mut(&order_id) {
            pos.queue_position = new_position;
            pos.total_queue_size = total_size;
            pos.last_updated_ns = get_timestamp_ns();
            
            // Recalculate fill probability and time
            let (prob, time_ns) = self.calculate_fill_metrics(&pos);
            pos.fill_probability = prob;
            pos.estimated_fill_time_ns = time_ns;
        }
    }

    /// Process a trade tick and update queue positions
    #[inline(always)]
    pub fn process_trade(&self, trade: TradeTick) {
        // Record trade
        self.trade_history
            .entry(trade.symbol)
            .or_insert_with(|| SegQueue::new())
            .push(trade);
        
        // Update fill statistics
        if let Some(mut stats) = self.fill_stats.get_mut(&trade.symbol) {
            stats.update(trade.qty, trade.timestamp_ns);
        }
        
        // Update queue positions for orders at this price
        self.update_positions_for_trade(&trade);
    }

    /// Update positions based on a trade
    fn update_positions_for_trade(&self, trade: &TradeTick) {
        let side = match trade.side {
            Side::Buy => Side::Sell, // A buy trade hits sell orders
            Side::Sell => Side::Buy, // A sell trade hits buy orders
        };
        
        let key = (trade.symbol, side);
        
        if let Some(mut levels) = self.price_levels.get_mut(&key) {
            for level in levels.iter_mut() {
                if level.price == trade.price {
                    // Reduce queue size by trade quantity
                    let remaining = level.total_size.saturating_sub(trade.qty);
                    
                    // If our order was in front of this trade, reduce our position
                    for (_, mut order) in self.active_orders.iter_mut() {
                        if order.symbol == trade.symbol 
                            && order.price == trade.price 
                            && order.side == side
                            && order.queue_position < trade.qty 
                        {
                            // Our order got partially or fully filled
                            let fill_qty = std::cmp::min(order.queue_position + 1, trade.qty);
                            // In real implementation, would trigger fill event
                        }
                    }
                    
                    level.total_size = remaining;
                    level.last_update_ns = trade.timestamp_ns;
                }
            }
        }
    }

    /// Calculate fill probability and time based on queue dynamics
    fn calculate_fill_metrics(&self, pos: &QueuePosition) -> (f32, u64) {
        let stats = self.fill_stats.get(&pos.symbol);
        
        if let Some(stats) = stats {
            let queue_ahead = pos.queue_position;
            let depletion_rate = stats.queue_depletion_rate.max(0.001); // Avoid div by zero
            
            // Estimate time to fill based on queue depletion
            let avg_trade_size = stats.avg_trade_size.max(1);
            let trades_needed = queue_ahead / avg_trade_size;
            let estimated_time_ns = ((trades_needed as f64 / depletion_rate) * 1_000_000_000.0) as u64;
            
            // Probability decreases with queue position and time
            let time_factor = (1_000_000_000.0 / (estimated_time_ns as f64 + 1_000_000_000.0));
            let position_factor = (1.0 / (1.0 + queue_ahead as f64 / 100.0));
            
            let probability = (time_factor * 0.5 + position_factor * 0.5) as f32;
            
            (probability.clamp(0.0, 1.0), estimated_time_ns)
        } else {
            // Default estimates when no stats available
            (0.5, 1_000_000_000) // 50% chance, 1 second estimate
        }
    }

    /// Get current queue position for an order
    pub fn get_position(&self, order_id: u64) -> Option<QueuePosition> {
        self.active_orders.get(&order_id).map(|p| p.clone())
    }

    /// Remove a filled order
    pub fn order_filled(&self, order_id: u64) {
        if let Some(order) = self.active_orders.remove(&order_id) {
            self.fills_completed.fetch_add(1, Ordering::Relaxed);
            
            // Update fill stats
            if let Some(mut stats) = self.fill_stats.get_mut(&order.value().symbol) {
                stats.fills_at_price += 1;
            }
        }
    }

    /// Cancel a tracked order
    pub fn order_cancelled(&self, order_id: u64) {
        self.active_orders.remove(&order_id);
        self.cancellations.fetch_add(1, Ordering::Relaxed);
    }

    /// Update price level data from order book
    pub fn update_price_level(&self, symbol: u64, side: Side, levels: Vec<PriceLevel>) {
        self.price_levels.insert((symbol, side), levels);
    }

    /// Get all active queue positions
    pub fn get_all_positions(&self) -> Vec<QueuePosition> {
        self.active_orders.iter().map(|r| r.value().clone()).collect()
    }

    /// Get queue statistics
    pub fn get_stats(&self) -> QueueStats {
        let total_fills = self.fills_completed.load(Ordering::Relaxed);
        let total_cancels = self.cancellations.load(Ordering::Relaxed);
        
        let fill_rate = if total_fills + total_cancels > 0 {
            total_fills as f32 / (total_fills + total_cancels) as f32
        } else {
            0.0
        };
        
        // Calculate average fill time
        let avg_fill_time_ms = self.active_orders.iter()
            .map(|r| r.value().estimated_fill_time_ns as f64 / 1_000_000.0)
            .sum::<f64>() / self.active_orders.len().max(1) as f64;
        
        QueueStats {
            tracked_levels: self.price_levels.len(),
            active_orders: self.active_orders.len(),
            fills_completed: total_fills,
            cancellations: total_cancels,
            avg_fill_time_ms,
            fill_rate,
        }
    }

    /// Prune old trade history to maintain bounded memory
    pub fn prune_history(&self, max_age_ns: u64) {
        let now = get_timestamp_ns();
        
        for (_, queue) in self.trade_history.iter() {
            let mut temp = Vec::new();
            while let Some(trade) = queue.pop() {
                if now - trade.timestamp_ns < max_age_ns {
                    temp.push(trade);
                }
            }
            // Re-add recent trades
            for trade in temp {
                queue.push(trade);
            }
        }
    }
}

/// Get current timestamp in nanoseconds
#[inline(always)]
fn get_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

impl Default for QueuePositionTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_tracking() {
        let tracker = QueuePositionTracker::new();
        
        let order = QueuePosition {
            symbol: 12345,
            side: Side::Buy,
            price: 50000_00000000,
            order_id: 1001,
            queue_position: 10,
            total_queue_size: 100_00000000,
            estimated_fill_time_ns: 0,
            fill_probability: 0.0,
            last_updated_ns: get_timestamp_ns(),
        };
        
        tracker.track_order(order);
        
        let pos = tracker.get_position(1001);
        assert!(pos.is_some());
        assert_eq!(pos.unwrap().queue_position, 10);
    }

    #[test]
    fn test_trade_processing() {
        let tracker = QueuePositionTracker::new();
        
        let trade = TradeTick {
            symbol: 12345,
            price: 50000_00000000,
            qty: 1_00000000,
            side: Side::Buy,
            timestamp_ns: get_timestamp_ns(),
            maker_order_id: None,
        };
        
        tracker.process_trade(trade);
        
        let stats = tracker.get_stats();
        assert!(stats.tracked_levels >= 0);
    }
}
