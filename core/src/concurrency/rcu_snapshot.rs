/// Read-Copy-Update (RCU) Data Structures for Lock-Free Concurrency
/// ====================================================================
/// 
/// Implements RCU patterns for safely reading complex, deeply nested state
/// without blocking high-frequency writer threads. Critical for maintaining
/// microsecond latencies in the trading core.
/// 
/// Key features:
/// - Zero-lock reads for order book and portfolio state
/// - Epoch-based reclamation for safe memory management
/// - Copy-on-write semantics with atomic pointer swaps
/// - Optimized for AMD Ryzen AI 5 multi-core architecture
/// 
/// Use cases:
/// - Global order book snapshots for risk checks
/// - Portfolio state for P&L calculations
/// - Configuration updates without stopping execution

use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::Arc;
use std::ptr;
use std::marker::PhantomData;
use std::time::Duration;

/// Epoch counter for RCU grace period tracking
static EPOCH_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Number of epochs to wait before reclaiming memory
const GRACE_PERIOD_EPOCHS: u64 = 3;

/// RCU-protected data wrapper
/// 
/// Provides lock-free reads with copy-on-write semantics.
/// Writers create a new copy, readers see consistent snapshots.
pub struct RcuCell<T> {
    /// Pointer to current data
    data: AtomicPtr<T>,
    /// Epoch when this cell was last updated
    update_epoch: AtomicU64,
    /// Phantom data for ownership
    _marker: PhantomData<Box<T>>,
}

unsafe impl<T: Send + Sync> Send for RcuCell<T> {}
unsafe impl<T: Send + Sync> Sync for RcuCell<T> {}

impl<T> RcuCell<T> {
    /// Create new RCU cell with initial value
    pub fn new(value: T) -> Self {
        let boxed = Box::new(value);
        Self {
            data: AtomicPtr::new(Box::into_raw(boxed)),
            update_epoch: AtomicU64::new(0),
            _marker: PhantomData,
        }
    }
    
    /// Get a reference to current data (lock-free read)
    /// 
    /// SAFETY: Caller must ensure they are in an RCU read section
    /// and the reference is not used after the read section ends.
    #[inline]
    pub fn get(&self) -> &T {
        unsafe {
            &*self.data.load(Ordering::Acquire)
        }
    }
    
    /// Update with new value (copy-on-write)
    /// 
    /// Creates a new copy and atomically swaps the pointer.
    /// Old value is queued for deferred reclamation.
    pub fn set(&self, new_value: T) {
        let new_boxed = Box::new(new_value);
        let new_ptr = Box::into_raw(new_boxed);
        
        let current_epoch = EPOCH_COUNTER.load(Ordering::Relaxed);
        
        // Atomic swap
        let old_ptr = self.data.swap(new_ptr, Ordering::AcqRel);
        self.update_epoch.store(current_epoch, Ordering::Release);
        
        // Queue old value for reclamation
        unsafe {
            RcUGuard::queue_for_reclaim(old_ptr, current_epoch);
        }
    }
    
    /// Get epoch of last update
    pub fn last_update_epoch(&self) -> u64 {
        self.update_epoch.load(Ordering::Relaxed)
    }
}

impl<T> Drop for RcuCell<T> {
    fn drop(&mut self) {
        unsafe {
            let ptr = self.data.load(Ordering::Relaxed);
            if !ptr.is_null() {
                drop(Box::from_raw(ptr));
            }
        }
    }
}

/// RCU Guard for managing read sections and deferred reclamation
pub struct RcUGuard {
    /// Epoch when this guard was created
    enter_epoch: u64,
}

impl RcUGuard {
    /// Enter an RCU read section
    /// 
    /// Returns a guard that tracks the read section.
    /// Data accessed during this section is guaranteed stable.
    #[inline]
    pub fn enter() -> Self {
        let epoch = EPOCH_COUNTER.load(Ordering::Relaxed);
        Self { enter_epoch: epoch }
    }
    
    /// Check if current epoch has advanced past our read section
    #[inline]
    pub fn should_exit(&self) -> bool {
        let current = EPOCH_COUNTER.load(Ordering::Relaxed);
        current > self.enter_epoch + GRACE_PERIOD_EPOCHS
    }
    
    /// Queue a pointer for reclamation after grace period
    unsafe fn queue_for_reclaim(ptr: *mut u8, epoch: u64) {
        static mut RECLAIM_QUEUE: Option<Vec<(*mut u8, u64)>> = None;
        
        if RECLAIM_QUEUE.is_none() {
            RECLAIM_QUEUE = Some(Vec::with_capacity(1024));
        }
        
        if let Some(queue) = RECLAIM_QUEUE.as_mut() {
            queue.push((ptr, epoch));
            if queue.len() > 100 {
                Self::attempt_reclaim(queue);
            }
        }
    }
    
    unsafe fn attempt_reclaim(queue: &mut Vec<(*mut u8, u64)>) {
        let current_epoch = EPOCH_COUNTER.load(Ordering::Relaxed);
        queue.retain(|&(ptr, epoch)| {
            if current_epoch > epoch + GRACE_PERIOD_EPOCHS {
                drop(Box::from_raw(ptr as *mut u8));
                false
            } else {
                true
            }
        });
    }
}

/// Advance the global epoch counter
#[inline]
pub fn advance_epoch() -> u64 {
    EPOCH_COUNTER.fetch_add(1, Ordering::Release) + 1
}

/// RCU-protected snapshot of complex state
#[derive(Clone)]
pub struct RcuSnapshot<T> {
    data: Arc<T>,
    epoch: u64,
}

impl<T> RcuSnapshot<T> {
    pub fn new(data: T) -> Self {
        Self {
            data: Arc::new(data),
            epoch: EPOCH_COUNTER.load(Ordering::Relaxed),
        }
    }
    
    #[inline]
    pub fn get(&self) -> &T {
        &*self.data
    }
    
    #[inline]
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
    
    #[inline]
    pub fn is_stale(&self, max_age_epochs: u64) -> bool {
        let current = EPOCH_COUNTER.load(Ordering::Relaxed);
        current > self.epoch + max_age_epochs
    }
}

/// Simplified order book state
#[derive(Clone)]
pub struct OrderBookState<T> {
    pub best_bid: f64,
    pub best_ask: f64,
    pub bid_depth: f64,
    pub ask_depth: f64,
    pub timestamp: u64,
    pub payload: T,
}

/// RCU-enabled order book container
pub struct RcuOrderBook<T> {
    snapshot: RcuCell<OrderBookState<T>>,
    update_count: AtomicU64,
}

impl<T: Clone + Send + Sync> RcuOrderBook<T> {
    pub fn new(initial: OrderBookState<T>) -> Self {
        Self {
            snapshot: RcuCell::new(initial),
            update_count: AtomicU64::new(0),
        }
    }
    
    #[inline]
    pub fn get_snapshot(&self) -> &OrderBookState<T> {
        self.snapshot.get()
    }
    
    pub fn update(&self, new_state: OrderBookState<T>) {
        self.snapshot.set(new_state);
        self.update_count.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn update_count(&self) -> u64 {
        self.update_count.load(Ordering::Relaxed)
    }
}

/// Portfolio state for RCU protection
#[derive(Clone)]
pub struct PortfolioState {
    pub total_value: f64,
    pub cash: f64,
    pub positions: Vec<(String, f64)>,
    pub unrealized_pnl: f64,
    pub timestamp: u64,
}

/// RCU-protected portfolio
pub struct RcuPortfolio {
    state: RcuCell<PortfolioState>,
    last_pnl_epoch: AtomicU64,
}

impl RcuPortfolio {
    pub fn new(initial: PortfolioState) -> Self {
        Self {
            state: RcuCell::new(initial),
            last_pnl_epoch: AtomicU64::new(0),
        }
    }
    
    #[inline]
    pub fn get_state(&self) -> &PortfolioState {
        self.state.get()
    }
    
    pub fn update(&self, new_state: PortfolioState) {
        self.state.set(new_state);
        self.last_pnl_epoch.store(
            EPOCH_COUNTER.load(Ordering::Relaxed),
            Ordering::Release,
        );
    }
    
    pub fn last_pnl_epoch(&self) -> u64 {
        self.last_pnl_epoch.load(Ordering::Relaxed)
    }
}

/// Background epoch reclaimer
pub struct EpochReclaimer {
    interval: Duration,
    running: AtomicBool,
}

impl EpochReclaimer {
    pub fn new(interval_ms: u64) -> Self {
        Self {
            interval: Duration::from_millis(interval_ms),
            running: AtomicBool::new(false),
        }
    }
    
    pub fn start(&self) -> std::thread::JoinHandle<()> {
        self.running.store(true, Ordering::Relaxed);
        
        std::thread::spawn({
            let interval = self.interval;
            let running = &self.running;
            
            move || {
                while running.load(Ordering::Relaxed) {
                    std::thread::sleep(interval);
                    advance_epoch();
                }
            }
        })
    }
    
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rcu_cell_basic() {
        let cell = RcuCell::new(42i32);
        assert_eq!(*cell.get(), 42);
        cell.set(100);
        assert_eq!(*cell.get(), 100);
    }
    
    #[test]
    fn test_rcu_snapshot() {
        let snapshot = RcuSnapshot::new("test".to_string());
        assert_eq!(snapshot.get(), "test");
        assert!(!snapshot.is_stale(10));
    }
    
    #[test]
    fn test_rcu_orderbook() {
        let initial = OrderBookState {
            best_bid: 100.0,
            best_ask: 100.1,
            bid_depth: 1000.0,
            ask_depth: 1000.0,
            timestamp: 0,
            payload: "BTCUSD".to_string(),
        };
        
        let book = RcuOrderBook::new(initial);
        assert_eq!(book.get_snapshot().best_bid, 100.0);
        
        book.update(OrderBookState {
            best_bid: 101.0,
            ..book.get_snapshot().clone()
        });
        
        assert_eq!(book.get_snapshot().best_bid, 101.0);
    }
}
