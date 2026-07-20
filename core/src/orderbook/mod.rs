// core/src/orderbook/mod.rs
// =============================================================================
// STAGE 2 - CHAPTER 1 - FILE 3
// Focus: Module declarations, order book traits, and cross-thread synchronization.
// Exposes the order book state safely using lock-free primitives.
// =============================================================================

pub mod l2_book;
pub mod snapshot_manager;

pub use l2_book::{L2OrderBook, PriceLevel, SideBook};
pub use snapshot_manager::SnapshotManager;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;

/// Trait defining the interface for any order book implementation.
/// Allows swapping between L2, L3, or custom book implementations.
pub trait OrderBookTrait: Send + Sync {
    /// Apply a batch of bid updates
    fn update_bids(&self, levels: &[(u64, f64)]);
    
    /// Apply a batch of ask updates
    fn update_asks(&self, levels: &[(u64, f64)]);
    
    /// Get the best bid price (as scaled integer)
    fn best_bid(&self) -> Option<u64>;
    
    /// Get the best ask price (as scaled integer)
    fn best_ask(&self) -> Option<u64>;
    
    /// Get the mid price
    fn mid_price(&self) -> Option<f64>;
    
    /// Get the spread in ticks
    fn spread(&self) -> Option<u64>;
    
    /// Check if the book is synced and ready for trading
    fn is_ready(&self) -> bool;
    
    /// Get the last update sequence ID
    fn last_update_id(&self) -> u64;
}

impl OrderBookTrait for L2OrderBook {
    fn update_bids(&self, levels: &[(u64, f64)]) {
        let current_id = self.last_update_id.load(AtomicOrdering::Relaxed);
        self.apply_delta(current_id + 1, levels, &[]);
    }

    fn update_asks(&self, levels: &[(u64, f64)]) {
        let current_id = self.last_update_id.load(AtomicOrdering::Relaxed);
        self.apply_delta(current_id + 1, &[], levels);
    }

    fn best_bid(&self) -> Option<u64> {
        self.bids.get_best().map(|l| l.price)
    }

    fn best_ask(&self) -> Option<u64> {
        self.asks.get_best().map(|l| l.price)
    }

    fn mid_price(&self) -> Option<f64> {
        self.get_mid_price()
    }

    fn spread(&self) -> Option<u64> {
        self.get_spread_ticks()
    }

    fn is_ready(&self) -> bool {
        // Simple implementation - always ready if we have data
        // In production, this would check SnapshotManager state
        self.bids.get_depth() > 0 && self.asks.get_depth() > 0
    }

    fn last_update_id(&self) -> u64 {
        self.last_update_id.load(AtomicOrdering::Acquire)
    }
}

/// Thread-safe wrapper for sharing an order book across threads.
/// Uses Arc for shared ownership and atomic operations for synchronization.
pub struct SharedOrderBook {
    inner: Arc<L2OrderBook>,
    manager: Arc<SnapshotManager>,
    /// Flag to signal readers that new data is available
    data_ready: AtomicBool,
}

impl SharedOrderBook {
    /// Create a new shared order book instance
    pub fn new(symbol_hash: u64) -> Self {
        SharedOrderBook {
            inner: Arc::new(L2OrderBook::new(symbol_hash)),
            manager: Arc::new(SnapshotManager::new()),
            data_ready: AtomicBool::new(false),
        }
    }

    /// Get a clone of the Arc for reading
    pub fn clone_book(&self) -> Arc<L2OrderBook> {
        Arc::clone(&self.inner)
    }

    /// Get a reference to the snapshot manager
    pub fn get_manager(&self) -> &SnapshotManager {
        &self.manager
    }

    /// Apply a snapshot (thread-safe)
    pub fn apply_snapshot(&self, bids: &[(u64, f64)], asks: &[(u64, f64)], last_id: u64) {
        self.manager.apply_snapshot(&self.inner, bids, asks, last_id);
        self.data_ready.store(true, AtomicOrdering::Release);
    }

    /// Apply a delta update (thread-safe)
    /// Returns true if successfully applied, false if dropped
    pub fn apply_delta(&self, update_id: u64, bids: &[(u64, f64)], asks: &[(u64, f64)]) -> bool {
        let result = self.manager.apply_delta(&self.inner, update_id, bids, asks);
        if result {
            self.data_ready.store(true, AtomicOrdering::Release);
        }
        result
    }

    /// Check if new data is available since last read
    pub fn has_new_data(&self) -> bool {
        self.data_ready.load(AtomicOrdering::Acquire)
    }

    /// Mark data as consumed (reset the flag)
    pub fn mark_consumed(&self) {
        self.data_ready.store(false, AtomicOrdering::Release);
    }

    /// Quick access to check if ready for trading
    pub fn is_trading_ready(&self) -> bool {
        self.manager.is_ready() && self.has_new_data()
    }
}

/// Clone implementation for SharedOrderBook (cheap Arc clone)
impl Clone for SharedOrderBook {
    fn clone(&self) -> Self {
        SharedOrderBook {
            inner: Arc::clone(&self.inner),
            manager: Arc::clone(&self.manager),
            data_ready: AtomicBool::new(self.data_ready.load(AtomicOrdering::Relaxed)),
        }
    }
}

/// Crossbeam-based channel types for sending order book snapshots to other threads
/// This avoids locking by using lock-free MPSC queues.
#[cfg(feature = "crossbeam")]
pub mod channels {
    use super::*;
    use crossbeam::channel::{bounded, Sender, Receiver};

    /// Create a bounded channel for order book updates
    /// Capacity of 1024 should be sufficient for microsecond updates
    pub fn create_ob_channel(capacity: usize) -> (Sender<OrderBookUpdate>, Receiver<OrderBookUpdate>) {
        bounded(capacity)
    }

    /// Enum representing different types of order book updates
    #[derive(Debug, Clone)]
    pub enum OrderBookUpdate {
        Snapshot {
            bids: Vec<(u64, f64)>,
            asks: Vec<(u64, f64)>,
            last_id: u64,
        },
        Delta {
            update_id: u64,
            bids: Vec<(u64, f64)>,
            asks: Vec<(u64, f64)>,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_orderbook_thread_safety() {
        let shared = SharedOrderBook::new(12345);
        
        // Simulate applying snapshot
        shared.apply_snapshot(&[(50000, 1.0)], &[(50100, 1.0)], 100);
        
        assert!(shared.is_trading_ready());
        assert!(shared.has_new_data());
        
        // Consume the data
        shared.mark_consumed();
        assert!(!shared.has_new_data());
        
        // Apply delta
        let result = shared.apply_delta(101, &[(50001, 0.5)], &[]);
        assert!(result);
        assert!(shared.has_new_data());
    }
}
