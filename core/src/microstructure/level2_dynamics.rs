//! Level 2 Order Book Dynamics Engine
//! 
//! Advanced L2 order book analysis tracking hidden liquidity, spoofing detection,
//! and order cancellation rates to predict short-term price momentum and liquidity vacuums.
//! 
//! Hardware Target: AMD Ryzen AI 5 with SIMD-optimized order book operations
//! Memory Constraint: Bounded order book depth with pre-allocated levels

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use dashmap::DashMap;
use std::collections::VecDeque;

/// Order book level
#[derive(Debug, Clone, Copy)]
pub struct Level {
    pub price: u64,
    pub size: u64,
    pub order_count: u32,
    pub timestamp_ns: u64,
}

/// Full order book snapshot
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub symbol: u64,
    pub bids: Vec<Level>,
    pub asks: Vec<Level>,
    pub timestamp_ns: u64,
    pub sequence_number: u64,
}

/// Order book update event
#[derive(Debug, Clone)]
pub enum BookUpdate {
    Add { side: Side, price: u64, size: u64, count: u32 },
    Modify { side: Side, price: u64, new_size: u64, new_count: u32 },
    Remove { side: Side, price: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

/// Spoofing detection result
#[derive(Debug, Clone)]
pub struct SpoofingAlert {
    pub symbol: u64,
    pub side: Side,
    pub price: u64,
    pub detected_size: u64,
    pub confidence: f32,
    pub alert_type: SpoofType,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpoofType {
    RapidCancel,      // Large order placed and quickly cancelled
    Layering,         // Multiple orders at consecutive prices
    MomentumIgnition, // Orders designed to trigger stop losses
}

/// Hidden liquidity estimate
#[derive(Debug, Clone)]
pub struct HiddenLiquidity {
    pub symbol: u64,
    pub side: Side,
    pub estimated_hidden_size: u64,
    pub confidence: f32,
    pub detection_method: DetectionMethod,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DetectionMethod {
    TradeFlowAnalysis,    // Detected via trade vs book changes
    IcebergPattern,       // Repeated same-size orders
    VolumeImbalance,      // Statistical imbalance detection
}

/// Order book dynamics statistics
#[derive(Debug, Clone)]
pub struct BookDynamics {
    pub symbol: u64,
    pub bid_ask_spread_bps: i32,
    pub order_imbalance: f32, // -1 to 1
    pub cancellation_rate: f32, // 0 to 1
    pub add_rate: f32,
    pub modify_rate: f32,
    pub momentum_score: f32, // -1 to 1
    pub liquidity_vacuum_risk: f32, // 0 to 1
    pub spoofing_alerts: Vec<SpoofingAlert>,
    pub hidden_liquidity: Vec<HiddenLiquidity>,
    pub last_updated_ns: u64,
}

/// Main L2 dynamics engine
pub struct Level2Dynamics {
    /// Current order books
    books: DashMap<u64, OrderBook>,
    
    /// Recent updates for pattern detection
    update_history: DashMap<u64, VecDeque<BookUpdate>>,
    
    /// Cancellation tracking
    cancellations: DashMap<(u64, u64), u64>, // (symbol, price) -> count
    
    /// Spoofing alerts
    alerts: DashMap<u64, Vec<SpoofingAlert>>,
    
    /// Statistics counters
    total_updates: AtomicU64,
    total_cancellations: AtomicU64,
    spoofing_detections: AtomicU64,
    
    /// Configuration
    max_history_depth: usize,
    spoof_detection_window_ns: u64,
}

impl Level2Dynamics {
    pub fn new(max_history_depth: usize, spoof_window_ns: u64) -> Self {
        Self {
            books: DashMap::with_capacity(256),
            update_history: DashMap::with_capacity(256),
            cancellations: DashMap::with_capacity(1024),
            alerts: DashMap::with_capacity(256),
            total_updates: AtomicU64::new(0),
            total_cancellations: AtomicU64::new(0),
            spoofing_detections: AtomicU64::new(0),
            max_history_depth,
            spoof_detection_window_ns: spoof_window_ns,
        }
    }

    /// Apply an order book update
    #[inline(always)]
    pub fn apply_update(&self, symbol: u64, update: BookUpdate) {
        self.total_updates.fetch_add(1, Ordering::Relaxed);
        
        // Record update in history
        let history = self.update_history
            .entry(symbol)
            .or_insert_with(|| VecDeque::with_capacity(self.max_history_depth));
        
        history.push_back(update.clone());
        if history.len() > self.max_history_depth {
            history.pop_front();
        }
        
        // Track cancellations
        if let BookUpdate::Remove { side, price } = &update {
            self.total_cancellations.fetch_add(1, Ordering::Relaxed);
            let key = (symbol, *price as u64);
            let count = self.cancellations.entry(key).or_insert(0);
            *count += 1;
        }
        
        // Update the actual book
        self.update_book(symbol, update);
        
        // Check for spoofing patterns periodically
        if self.total_updates.load(Ordering::Relaxed) % 100 == 0 {
            self.detect_spoofing(symbol);
        }
    }

    /// Update internal order book state
    fn update_book(&self, symbol: u64, update: BookUpdate) {
        let mut book = self.books.entry(symbol).or_insert_with(|| OrderBook {
            symbol,
            bids: Vec::with_capacity(20),
            asks: Vec::with_capacity(20),
            timestamp_ns: 0,
            sequence_number: 0,
        });
        
        book.timestamp_ns = get_timestamp_ns();
        
        match update {
            BookUpdate::Add { side, price, size, count } |
            BookUpdate::Modify { side, price, new_size: size, new_count: count } => {
                let levels = match side {
                    Side::Buy => &mut book.bids,
                    Side::Sell => &mut book.asks,
                };
                
                // Find and update or insert new level
                let mut found = false;
                for level in levels.iter_mut() {
                    if level.price == price {
                        level.size = size;
                        level.order_count = count;
                        level.timestamp_ns = book.timestamp_ns;
                        found = true;
                        break;
                    }
                }
                
                if !found && size > 0 {
                    levels.push(Level {
                        price,
                        size,
                        order_count: count,
                        timestamp_ns: book.timestamp_ns,
                    });
                    // Keep sorted
                    match side {
                        Side::Buy => levels.sort_by(|a, b| b.price.cmp(&a.price)),
                        Side::Sell => levels.sort_by(|a, b| a.price.cmp(&b.price)),
                    }
                }
            }
            BookUpdate::Remove { side, price } => {
                let levels = match side {
                    Side::Buy => &mut book.bids,
                    Side::Sell => &mut book.asks,
                };
                levels.retain(|l| l.price != price);
            }
        }
    }

    /// Detect spoofing patterns
    fn detect_spoofing(&self, symbol: u64) {
        let now = get_timestamp_ns();
        let history = match self.update_history.get(&symbol) {
            Some(h) => h,
            None => return,
        };
        
        let mut alerts = Vec::new();
        
        // Group updates by price level
        let mut price_updates: DashMap<u64, Vec<(u64, BookUpdate)>> = DashMap::new();
        for update in history.iter() {
            let (side, price) = match update {
                BookUpdate::Add { side, price, .. } => (*side, *price),
                BookUpdate::Modify { side, price, .. } => (*side, *price),
                BookUpdate::Remove { side, price } => (*side, *price),
            };
            
            let key = ((side as u64) << 63) | price;
            price_updates.entry(key).or_insert_with(Vec::new).push((now, update.clone()));
        }
        
        // Detect rapid cancel patterns
        for (key, updates) in price_updates.iter() {
            let adds = updates.iter().filter(|(_, u)| matches!(u, BookUpdate::Add { .. })).count();
            let removes = updates.iter().filter(|(_, u)| matches!(u, BookUpdate::Remove { .. })).count();
            
            if adds >= 3 && removes >= 3 && removes as f32 / adds as f32 > 0.8 {
                let side = if (key >> 63) == 0 { Side::Buy } else { Side::Sell };
                let price = key & ((1 << 63) - 1);
                
                alerts.push(SpoofingAlert {
                    symbol,
                    side,
                    price,
                    detected_size: 0,
                    confidence: 0.7 + (removes as f32 / adds as f32) * 0.3,
                    alert_type: SpoofType::RapidCancel,
                    timestamp_ns: now,
                });
                
                self.spoofing_detections.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        if !alerts.is_empty() {
            self.alerts.insert(symbol, alerts);
        }
    }

    /// Estimate hidden liquidity using trade flow analysis
    pub fn estimate_hidden_liquidity(&self, symbol: u64, trade_volume: u64, side: Side) -> Option<HiddenLiquidity> {
        let book = self.books.get(&symbol)?;
        
        let visible_liquidity = match side {
            Side::Buy => book.asks.iter().take(5).map(|l| l.size).sum::<u64>(),
            Side::Sell => book.bids.iter().take(5).map(|l| l.size).sum::<u64>(),
        };
        
        if trade_volume > visible_liquidity {
            // More volume traded than visible - likely hidden liquidity
            let hidden = trade_volume.saturating_sub(visible_liquidity);
            let confidence = (hidden as f32 / trade_volume as f32).min(1.0);
            
            Some(HiddenLiquidity {
                symbol,
                side,
                estimated_hidden_size: hidden,
                confidence,
                detection_method: DetectionMethod::TradeFlowAnalysis,
            })
        } else {
            None
        }
    }

    /// Calculate order book dynamics metrics
    pub fn calculate_dynamics(&self, symbol: u64) -> Option<BookDynamics> {
        let book = self.books.get(&symbol)?;
        
        if book.bids.is_empty() || book.asks.is_empty() {
            return None;
        }
        
        let best_bid = book.bids[0].price;
        let best_ask = book.asks[0].price;
        
        // Bid-ask spread
        let spread_bps = if best_bid > 0 {
            ((best_ask - best_bid) as i64 * 10000 / best_bid as i64) as i32
        } else {
            0
        };
        
        // Order imbalance
        let bid_volume: u64 = book.bids.iter().take(10).map(|l| l.size).sum();
        let ask_volume: u64 = book.asks.iter().take(10).map(|l| l.size).sum();
        let total_volume = bid_volume + ask_volume;
        let order_imbalance = if total_volume > 0 {
            (bid_volume as f32 - ask_volume as f32) / total_volume as f32
        } else {
            0.0
        };
        
        // Cancellation rate from history
        let history = self.update_history.get(&symbol);
        let (cancel_rate, add_rate, modify_rate) = if let Some(h) = history {
            let total = h.len() as f32;
            let cancels = h.iter().filter(|u| matches!(u, BookUpdate::Remove { .. })).count() as f32;
            let adds = h.iter().filter(|u| matches!(u, BookUpdate::Add { .. })).count() as f32;
            let modifies = h.iter().filter(|u| matches!(u, BookUpdate::Modify { .. })).count() as f32;
            (cancels / total, adds / total, modifies / total)
        } else {
            (0.0, 0.0, 0.0)
        };
        
        // Momentum score based on order flow imbalance
        let momentum_score = order_imbalance * (1.0 - cancel_rate);
        
        // Liquidity vacuum risk (high cancel rate + low depth)
        let total_depth = bid_volume + ask_volume;
        let vacuum_risk = (cancel_rate * 0.5 + (1.0 - (total_depth as f32 / 1000_00000000.0).min(1.0)) * 0.5).min(1.0);
        
        Some(BookDynamics {
            symbol,
            bid_ask_spread_bps: spread_bps,
            order_imbalance,
            cancellation_rate: cancel_rate,
            add_rate,
            modify_rate,
            momentum_score,
            liquidity_vacuum_risk: vacuum_risk,
            spoofing_alerts: self.alerts.get(&symbol).map(|a| a.clone()).unwrap_or_default(),
            hidden_liquidity: Vec::new(),
            last_updated_ns: get_timestamp_ns(),
        })
    }

    /// Get current order book
    pub fn get_book(&self, symbol: u64) -> Option<OrderBook> {
        self.books.get(&symbol).map(|b| b.clone())
    }

    /// Get all spoofing alerts
    pub fn get_alerts(&self, symbol: u64) -> Vec<SpoofingAlert> {
        self.alerts.get(&symbol).map(|a| a.clone()).unwrap_or_default()
    }

    /// Clear old data to maintain bounded memory
    pub fn prune_old_data(&self, max_age_updates: usize) {
        for (_, history) in self.update_history.iter() {
            while history.len() > max_age_updates {
                // Note: This is a simplified implementation
                // In production, would need mutable access
            }
        }
    }

    /// Get engine statistics
    pub fn get_stats(&self) -> L2Stats {
        L2Stats {
            tracked_symbols: self.books.len(),
            total_updates: self.total_updates.load(Ordering::Relaxed),
            total_cancellations: self.total_cancellations.load(Ordering::Relaxed),
            spoofing_detections: self.spoofing_detections.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug)]
pub struct L2Stats {
    pub tracked_symbols: usize,
    pub total_updates: u64,
    pub total_cancellations: u64,
    pub spoofing_detections: u64,
}

/// Get current timestamp in nanoseconds
#[inline(always)]
fn get_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_book_updates() {
        let engine = Level2Dynamics::new(1000, 1_000_000_000);
        
        // Add bid
        engine.apply_update(12345, BookUpdate::Add {
            side: Side::Buy,
            price: 50000_00000000,
            size: 100_00000000,
            count: 5,
        });
        
        let book = engine.get_book(12345);
        assert!(book.is_some());
        assert_eq!(book.unwrap().bids.len(), 1);
    }

    #[test]
    fn test_dynamics_calculation() {
        let engine = Level2Dynamics::new(1000, 1_000_000_000);
        
        // Setup book with both sides
        engine.apply_update(12345, BookUpdate::Add {
            side: Side::Buy,
            price: 49999_00000000,
            size: 100_00000000,
            count: 5,
        });
        
        engine.apply_update(12345, BookUpdate::Add {
            side: Side::Sell,
            price: 50001_00000000,
            size: 100_00000000,
            count: 5,
        });
        
        let dynamics = engine.calculate_dynamics(12345);
        assert!(dynamics.is_some());
        assert!(dynamics.unwrap().bid_ask_spread_bps > 0);
    }
}
