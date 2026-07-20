//! Order Cancellation Rate Analysis Module
//! 
//! This module tracks real-time order cancellation and spoofing rates
//! at specific price levels to gauge market conviction and identify
//! fake liquidity before it disappears.
//! 
//! Key features:
//! - Per-level cancellation tracking
//! - Spoofing detection algorithms
//! - Market conviction scoring
//! - Liquidity authenticity metrics
//! 
//! Target latency: < 3 microseconds per update

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Represents a single order in the tracking system
#[derive(Clone, Debug)]
pub struct TrackedOrder {
    pub order_id: u64,
    pub price: i64,
    pub quantity: u64,
    pub side: OrderSide,
    pub placement_time_ns: u64,
    pub is_cancelled: bool,
    pub cancellation_time_ns: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum OrderSide {
    Bid,
    Ask,
}

/// Tracks statistics for a single price level
#[derive(Debug, Clone)]
pub struct PriceLevelStats {
    /// Total orders placed at this level
    pub total_orders: u64,
    /// Total orders cancelled at this level
    pub cancelled_orders: u64,
    /// Total quantity placed
    pub total_quantity: u64,
    /// Total quantity cancelled
    pub cancelled_quantity: u64,
    /// Average time to cancellation (nanoseconds)
    pub avg_cancellation_time_ns: f64,
    /// Recent cancellation rate (last N events)
    pub recent_cancellation_rate: f64,
    /// Spoofing indicator score (0.0 to 1.0)
    pub spoofing_score: f64,
}

impl Default for PriceLevelStats {
    fn default() -> Self {
        Self {
            total_orders: 0,
            cancelled_orders: 0,
            total_quantity: 0,
            cancelled_quantity: 0,
            avg_cancellation_time_ns: 0.0,
            recent_cancellation_rate: 0.0,
            spoofing_score: 0.0,
        }
    }
}

/// Event types for the cancellation tracker
#[derive(Clone, Debug)]
pub enum OrderEvent {
    OrderPlaced {
        order_id: u64,
        price: i64,
        quantity: u64,
        side: OrderSide,
        timestamp_ns: u64,
    },
    OrderCancelled {
        order_id: u64,
        timestamp_ns: u64,
    },
    OrderFilled {
        order_id: u64,
        filled_quantity: u64,
        timestamp_ns: u64,
    },
}

/// Main cancellation rate analyzer
pub struct CancellationAnalyzer {
    /// Active orders being tracked
    active_orders: HashMap<u64, TrackedOrder>,
    /// Statistics per price level
    bid_level_stats: HashMap<i64, PriceLevelStats>,
    ask_level_stats: HashMap<i64, PriceLevelStats>,
    /// Rolling window of recent cancellation rates
    cancellation_history: VecDeque<f64>,
    max_history: usize,
    /// Global statistics
    pub total_placed: u64,
    pub total_cancelled: u64,
    pub global_cancellation_rate: f64,
    /// Timing configuration
    spoofing_threshold_ns: u64, // Orders cancelled within this time are suspicious
    min_spoofing_quantity: u64, // Minimum size to be considered potential spoofing
    /// Update counter for periodic recalculation
    update_count: u64,
    recalc_interval: u64,
}

impl CancellationAnalyzer {
    /// Create a new cancellation analyzer
    pub fn new(max_history: usize) -> Self {
        Self {
            active_orders: HashMap::with_capacity(10000),
            bid_level_stats: HashMap::with_capacity(100),
            ask_level_stats: HashMap::with_capacity(100),
            cancellation_history: VecDeque::with_capacity(max_history),
            max_history,
            total_placed: 0,
            total_cancelled: 0,
            global_cancellation_rate: 0.0,
            spoofing_threshold_ns: 500_000_000, // 500ms default
            min_spoofing_quantity: 1000,
            update_count: 0,
            recalc_interval: 100,
        }
    }

    /// Process an order event and update statistics
    /// 
    /// This is the hot path method - must complete in microseconds
    #[inline]
    pub fn process_event(&mut self, event: OrderEvent) -> Option<CancellationSignal> {
        match event {
            OrderEvent::OrderPlaced {
                order_id,
                price,
                quantity,
                side,
                timestamp_ns,
            } => {
                self.handle_order_placed(order_id, price, quantity, side.clone(), timestamp_ns);
                None
            }
            OrderEvent::OrderCancelled {
                order_id,
                timestamp_ns,
            } => self.handle_order_cancelled(order_id, timestamp_ns),
            OrderEvent::OrderFilled {
                order_id,
                filled_quantity,
                timestamp_ns,
            } => {
                self.handle_order_filled(order_id, filled_quantity, timestamp_ns);
                None
            }
        }
    }

    #[inline]
    fn handle_order_placed(
        &mut self,
        order_id: u64,
        price: i64,
        quantity: u64,
        side: OrderSide,
        timestamp_ns: u64,
    ) {
        let order = TrackedOrder {
            order_id,
            price,
            quantity,
            side: side.clone(),
            placement_time_ns: timestamp_ns,
            is_cancelled: false,
            cancellation_time_ns: None,
        };

        self.active_orders.insert(order_id, order);
        self.total_placed += 1;

        // Update level statistics
        let stats_map = match side {
            OrderSide::Bid => &mut self.bid_level_stats,
            OrderSide::Ask => &mut self.ask_level_stats,
        };

        let stats = stats_map.entry(price).or_default();
        stats.total_orders += 1;
        stats.total_quantity += quantity;
    }

    #[inline]
    fn handle_order_cancelled(
        &mut self,
        order_id: u64,
        timestamp_ns: u64,
    ) -> Option<CancellationSignal> {
        if let Some(mut order) = self.active_orders.remove(&order_id) {
            let duration_ns = timestamp_ns - order.placement_time_ns;
            
            // Check if this looks like spoofing
            let is_spoofing = duration_ns < self.spoofing_threshold_ns
                && order.quantity >= self.min_spoofing_quantity;

            order.is_cancelled = true;
            order.cancellation_time_ns = Some(timestamp_ns);

            // Update level statistics
            let (stats_map, side) = match order.side.clone() {
                OrderSide::Bid => (&mut self.bid_level_stats, OrderSide::Bid),
                OrderSide::Ask => (&mut self.ask_level_stats, OrderSide::Ask),
            };

            if let Some(stats) = stats_map.get_mut(&order.price) {
                stats.cancelled_orders += 1;
                stats.cancelled_quantity += order.quantity;

                // Update average cancellation time
                stats.avg_cancellation_time_ns = 
                    (stats.avg_cancellation_time_ns * (stats.cancelled_orders - 1) as f64
                        + duration_ns as f64) / stats.cancelled_orders as f64;

                // Recalculate spoofing score periodically
                self.update_count += 1;
                if self.update_count % self.recalc_interval == 0 {
                    stats.spoofing_score = self.calculate_spoofing_score(stats, duration_ns);
                }
            }

            self.total_cancelled += 1;
            self.global_cancellation_rate = 
                self.total_cancelled as f64 / self.total_placed.max(1) as f64;

            // Update rolling history
            let rate = if stats_map.get(&order.price).map_or(false, |s| s.total_orders > 0) {
                stats_map.get(&order.price).unwrap().cancelled_orders as f64
                    / stats_map.get(&order.price).unwrap().total_orders.max(1) as f64
            } else {
                0.0
            };

            self.cancellation_history.push_back(rate);
            if self.cancellation_history.len() > self.max_history {
                self.cancellation_history.pop_front();
            }

            // Return signal if significant cancellation detected
            if is_spoofing || rate > 0.7 {
                return Some(CancellationSignal {
                    price: order.price,
                    side,
                    cancellation_rate: rate,
                    is_spoofing,
                    duration_ns,
                    quantity: order.quantity,
                    timestamp_ns,
                });
            }
        }

        None
    }

    #[inline]
    fn handle_order_filled(
        &mut self,
        order_id: u64,
        filled_quantity: u64,
        _timestamp_ns: u64,
    ) {
        // Remove from active orders or reduce quantity
        if let Some(order) = self.active_orders.get_mut(&order_id) {
            if filled_quantity >= order.quantity {
                self.active_orders.remove(&order_id);
            } else {
                order.quantity -= filled_quantity;
            }
        }
        // Filled orders don't count as cancellations
    }

    #[inline]
    fn calculate_spoofing_score(&self, stats: &PriceLevelStats, last_duration_ns: u64) -> f64 {
        if stats.total_orders == 0 {
            return 0.0;
        }

        let cancel_ratio = stats.cancelled_orders as f64 / stats.total_orders.max(1) as f64;
        
        // Fast cancellations are more suspicious
        let speed_factor = {
            let avg_time_ms = stats.avg_cancellation_time_ns as f64 / 1_000_000.0;
            if avg_time_ms < 100.0 {
                1.0
            } else if avg_time_ms < 500.0 {
                0.7
            } else if avg_time_ms < 1000.0 {
                0.4
            } else {
                0.2
            }
        };

        // Large orders are more suspicious when cancelled
        let size_factor = {
            let avg_size = stats.cancelled_quantity as f64 / stats.cancelled_orders.max(1) as f64;
            (avg_size / self.min_spoofing_quantity as f64).min(2.0) / 2.0
        };

        (cancel_ratio * 0.5 + speed_factor * 0.3 + size_factor * 0.2).clamp(0.0, 1.0)
    }

    /// Get the cancellation rate for a specific price level
    #[inline]
    pub fn get_level_cancellation_rate(&self, price: i64, side: OrderSide) -> f64 {
        let stats_map = match side {
            OrderSide::Bid => &self.bid_level_stats,
            OrderSide::Ask => &self.ask_level_stats,
        };

        stats_map.get(&price)
            .map(|s| {
                if s.total_orders == 0 {
                    0.0
                } else {
                    s.cancelled_orders as f64 / s.total_orders as f64
                }
            })
            .unwrap_or(0.0)
    }

    /// Get the weighted average cancellation rate across all levels
    #[inline]
    pub fn get_weighted_cancellation_rate(&self, side: OrderSide) -> f64 {
        let stats_map = match side {
            OrderSide::Bid => &self.bid_level_stats,
            OrderSide::Ask => &self.ask_level_stats,
        };

        if stats_map.is_empty() {
            return 0.0;
        }

        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for stats in stats_map.values() {
            let weight = stats.total_quantity as f64;
            let rate = if stats.total_orders > 0 {
                stats.cancelled_orders as f64 / stats.total_orders as f64
            } else {
                0.0
            };

            weighted_sum += rate * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }

    /// Calculate market conviction score (inverse of cancellation rate)
    /// High conviction = low cancellation rate = real liquidity
    #[inline]
    pub fn market_conviction_score(&self) -> f64 {
        let bid_conviction = 1.0 - self.get_weighted_cancellation_rate(OrderSide::Bid);
        let ask_conviction = 1.0 - self.get_weighted_cancellation_rate(OrderSide::Ask);
        
        // Weighted average, slightly favoring the side with more volume
        let bid_volume: u64 = self.bid_level_stats.values()
            .map(|s| s.total_quantity)
            .sum();
        let ask_volume: u64 = self.ask_level_stats.values()
            .map(|s| s.total_quantity)
            .sum();

        let total_volume = (bid_volume + ask_volume) as f64;
        if total_volume == 0.0 {
            return 0.5;
        }

        (bid_conviction * bid_volume as f64 + ask_conviction * ask_volume as f64) / total_volume
    }

    /// Identify price levels with high spoofing probability
    pub fn detect_spoofing_levels(&self, threshold: f64) -> Vec<(i64, OrderSide, f64)> {
        let mut spoofing_levels = Vec::new();

        for (price, stats) in &self.bid_level_stats {
            if stats.spoofing_score >= threshold {
                spoofing_levels.push((*price, OrderSide::Bid, stats.spoofing_score));
            }
        }

        for (price, stats) in &self.ask_level_stats {
            if stats.spoofing_score >= threshold {
                spoofing_levels.push((*price, OrderSide::Ask, stats.spoofing_score));
            }
        }

        spoofing_levels.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        spoofing_levels
    }

    /// Get recent average cancellation rate
    #[inline]
    pub fn recent_average_cancellation_rate(&self) -> f64 {
        if self.cancellation_history.is_empty() {
            return self.global_cancellation_rate;
        }

        self.cancellation_history.iter().sum::<f64>() / self.cancellation_history.len() as f64
    }

    /// Check if current cancellation rate is abnormally high
    #[inline]
    pub fn is_elevated_cancellation(&self, standard_deviation_multiplier: f64) -> bool {
        if self.cancellation_history.len() < 10 {
            return false;
        }

        let mean = self.recent_average_cancellation_rate();
        let variance: f64 = self.cancellation_history.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / self.cancellation_history.len() as f64;
        let std_dev = variance.sqrt();

        let current_rate = self.cancellation_history.back().copied().unwrap_or(0.0);
        
        current_rate > mean + standard_deviation_multiplier * std_dev
    }

    /// Reset statistics (useful for regime change detection)
    pub fn reset(&mut self) {
        self.active_orders.clear();
        self.bid_level_stats.clear();
        self.ask_level_stats.clear();
        self.cancellation_history.clear();
        self.total_placed = 0;
        self.total_cancelled = 0;
        self.global_cancellation_rate = 0.0;
        self.update_count = 0;
    }
}

/// Signal generated when significant cancellation activity is detected
#[derive(Debug, Clone)]
pub struct CancellationSignal {
    pub price: i64,
    pub side: OrderSide,
    pub cancellation_rate: f64,
    pub is_spoofing: bool,
    pub duration_ns: u64,
    pub quantity: u64,
    pub timestamp_ns: u64,
}

/// Streaming processor for high-frequency cancellation analysis
pub struct StreamingCancellationProcessor {
    analyzer: CancellationAnalyzer,
    /// Signal buffer for downstream consumers
    signal_buffer: VecDeque<CancellationSignal>,
    max_signals: usize,
    /// Performance metrics
    pub events_processed: u64,
    pub signals_generated: u64,
    pub last_processing_time_ns: u64,
}

impl StreamingCancellationProcessor {
    pub fn new(max_history: usize, max_signals: usize) -> Self {
        Self {
            analyzer: CancellationAnalyzer::new(max_history),
            signal_buffer: VecDeque::with_capacity(max_signals),
            max_signals,
            events_processed: 0,
            signals_generated: 0,
            last_processing_time_ns: 0,
        }
    }

    /// Process an order event and optionally return a signal
    #[inline]
    pub fn process_event(&mut self, event: OrderEvent) -> Option<CancellationSignal> {
        let start = std::time::Instant::now();

        let result = self.analyzer.process_event(event);
        self.events_processed += 1;

        if let Some(signal) = result {
            self.signals_generated += 1;
            self.signal_buffer.push_back(signal.clone());
            if self.signal_buffer.len() > self.max_signals {
                self.signal_buffer.pop_front();
            }
            self.last_processing_time_ns = start.elapsed().as_nanos() as u64;
            Some(signal)
        } else {
            self.last_processing_time_ns = start.elapsed().as_nanos() as u64;
            None
        }
    }

    /// Get the most recent signals
    #[inline]
    pub fn recent_signals(&self, count: usize) -> Vec<CancellationSignal> {
        self.signal_buffer.iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }

    /// Get current market conviction
    #[inline]
    pub fn current_conviction(&self) -> f64 {
        self.analyzer.market_conviction_score()
    }

    /// Check for spoofing at specific levels
    #[inline]
    pub fn check_spoofing(&self, threshold: f64) -> Vec<(i64, OrderSide, f64)> {
        self.analyzer.detect_spoofing_levels(threshold)
    }
}

/// Lock-free version for multi-threaded scenarios using atomic operations
pub struct LockFreeCancellationTracker {
    /// Atomic counters for global stats
    total_placed: AtomicU64,
    total_cancelled: AtomicU64,
    /// Flag indicating data needs refresh
    dirty: AtomicBool,
    /// Internal analyzer protected by caller synchronization
    /// (In production, this would use RCU or similar)
    inner: std::sync::Mutex<CancellationAnalyzer>,
}

impl LockFreeCancellationTracker {
    pub fn new(max_history: usize) -> Self {
        Self {
            total_placed: AtomicU64::new(0),
            total_cancelled: AtomicU64::new(0),
            dirty: AtomicBool::new(false),
            inner: std::sync::Mutex::new(CancellationAnalyzer::new(max_history)),
        }
    }

    /// Fast path: increment counters atomically without locking
    #[inline]
    pub fn record_placement(&self) {
        self.total_placed.fetch_add(1, Ordering::Relaxed);
        self.dirty.store(true, Ordering::Release);
    }

    #[inline]
    pub fn record_cancellation(&self) {
        self.total_cancelled.fetch_add(1, Ordering::Relaxed);
        self.dirty.store(true, Ordering::Release);
    }

    /// Slow path: full event processing with lock
    pub fn process_full_event(&self, event: OrderEvent) -> Option<CancellationSignal> {
        let mut guard = self.inner.lock().unwrap();
        guard.process_event(event)
    }

    /// Get approximate global rate without locking
    #[inline]
    pub fn approximate_rate(&self) -> f64 {
        let placed = self.total_placed.load(Ordering::Acquire);
        let cancelled = self.total_cancelled.load(Ordering::Acquire);
        
        if placed == 0 {
            0.0
        } else {
            cancelled as f64 / placed as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancellation_tracking() {
        let mut analyzer = CancellationAnalyzer::new(100);
        
        // Place an order
        let event = OrderEvent::OrderPlaced {
            order_id: 1,
            price: 10000,
            quantity: 5000,
            side: OrderSide::Bid,
            timestamp_ns: 1000000,
        };
        analyzer.process_event(event);

        assert_eq!(analyzer.total_placed, 1);
        assert_eq!(analyzer.global_cancellation_rate, 0.0);

        // Cancel the order quickly (potential spoofing)
        let event = OrderEvent::OrderCancelled {
            order_id: 1,
            timestamp_ns: 1100000, // 100ms later
        };
        let signal = analyzer.process_event(event);

        assert_eq!(analyzer.total_cancelled, 1);
        assert!(signal.is_some());
        assert!(signal.unwrap().is_spoofing);
    }

    #[test]
    fn test_market_conviction() {
        let mut analyzer = CancellationAnalyzer::new(100);

        // Place and fill orders (real liquidity)
        for i in 0..10 {
            analyzer.process_event(OrderEvent::OrderPlaced {
                order_id: i,
                price: 10000 + i as i64,
                quantity: 1000,
                side: OrderSide::Bid,
                timestamp_ns: 1000000 + i * 1000,
            });

            analyzer.process_event(OrderEvent::OrderFilled {
                order_id: i,
                filled_quantity: 1000,
                timestamp_ns: 2000000 + i * 1000,
            });
        }

        // High conviction since no cancellations
        let conviction = analyzer.market_conviction_score();
        assert!(conviction > 0.9);
    }
}
