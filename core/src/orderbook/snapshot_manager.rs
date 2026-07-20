// core/src/orderbook/snapshot_manager.rs
// =============================================================================
// STAGE 2 - CHAPTER 1 - FILE 2
// Focus: Binance WebSocket snapshot synchronization and delta update validation.
// Handles sequence ID validation, automatic reconnection, and resync logic.
// =============================================================================

use super::l2_book::L2OrderBook;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Manages the state of order book synchronization with Binance.
/// Binance sends a snapshot followed by delta updates with sequence IDs.
/// We must ensure no gaps in sequence IDs to maintain book integrity.
pub struct SnapshotManager {
    /// Expected next update ID
    expected_update_id: AtomicU64,
    /// Flag indicating if we are currently synced and ready to trade
    is_synced: AtomicBool,
    /// Last successful sync timestamp for timeout detection
    last_sync_time: AtomicU64, // Stored as unix timestamp millis
    /// Reconnection attempt counter
    reconnect_count: AtomicU64,
    /// Maximum allowed sequence gap before forcing resync
    max_sequence_gap: u64,
}

impl SnapshotManager {
    pub fn new() -> Self {
        SnapshotManager {
            expected_update_id: AtomicU64::new(0),
            is_synced: AtomicBool::new(false),
            last_sync_time: AtomicU64::new(0),
            reconnect_count: AtomicU64::new(0),
            max_sequence_gap: 1000, // Allow some buffer for network jitter
        }
    }

    /// Initialize the order book with a snapshot from Binance REST API or WS snapshot stream.
    /// This should be called once when starting or when a resync is required.
    /// 
    /// # Arguments
    /// * `book` - The L2OrderBook to populate
    /// * `snapshot_bids` - Vector of (price, quantity) tuples for bids
    /// * `snapshot_asks` - Vector of (price, quantity) tuples for asks
    /// * `last_update_id` - The lastUpdateId from the snapshot response
    pub fn apply_snapshot(
        &self,
        book: &L2OrderBook,
        snapshot_bids: &[(u64, f64)],
        snapshot_asks: &[(u64, f64)],
        last_update_id: u64,
    ) {
        // Clear existing book by creating new one (or implement clear method)
        // For now, we just apply the snapshot as the base state
        
        // Apply snapshot data - note: snapshot is the state AT last_update_id
        // We need to store this ID to validate subsequent deltas
        book.apply_delta(last_update_id, snapshot_bids, snapshot_asks);
        
        // The next valid delta should have updateId > last_update_id
        self.expected_update_id.store(last_update_id + 1, AtomicOrdering::Release);
        self.is_synced.store(true, AtomicOrdering::Release);
        
        let now = Instant::now();
        self.last_sync_time.store(
            now.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64,
            AtomicOrdering::Release,
        );
        
        log_info(&format!("Snapshot applied. Next expected ID: {}", last_update_id + 1));
    }

    /// Validate and apply a delta update from the WebSocket stream.
    /// Returns true if the delta was applied, false if it was dropped (wrong sequence).
    /// 
    /// # Binance Sequence Rules:
    /// 1. First update ID in delta should be > snapshot's lastUpdateId
    /// 2. Each delta has a 'u' (updateId) field
    /// 3. Updates must be sequential (no gaps)
    pub fn apply_delta(&self, book: &L2OrderBook, update_id: u64, bids: &[(u64, f64)], asks: &[(u64, f64)]) -> bool {
        if !self.is_synced.load(AtomicOrdering::Acquire) {
            log_warn("Received delta while not synced. Dropping.");
            return false;
        }

        let expected = self.expected_update_id.load(AtomicOrdering::Acquire);

        // Check for sequence gap
        if update_id < expected {
            // Old message, possibly retransmission or late arrival - drop silently
            return false;
        }

        if update_id > expected {
            // Gap detected! This is critical.
            log_error(&format!(
                "Sequence gap detected! Expected {}, got {}. Initiating resync.",
                expected, update_id
            ));
            self.is_synced.store(false, AtomicOrdering::Release);
            self.reconnect_count.fetch_add(1, AtomicOrdering::Release);
            return false;
        }

        // Sequence matches - apply the delta
        book.apply_delta(update_id, bids, asks);
        
        // Update expected ID for next message
        self.expected_update_id.fetch_add(1, AtomicOrdering::Release);
        
        // Update sync timestamp
        let now = Instant::now();
        self.last_sync_time.store(
            now.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64,
            AtomicOrdering::Release,
        );

        true
    }

    /// Check if the connection has timed out (no updates for N seconds)
    pub fn check_timeout(&self, timeout_ms: u64) -> bool {
        if !self.is_synced.load(AtomicOrdering::Acquire) {
            return false; // Already unsynced
        }

        let now = Instant::now();
        let current_time = now.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
        let last_sync = self.last_sync_time.load(AtomicOrdering::Acquire);

        if current_time.saturating_sub(last_sync) > timeout_ms {
            log_warn(&format!("Sync timeout after {}ms. Forcing resync.", timeout_ms));
            self.is_synced.store(false, AtomicOrdering::Release);
            return true;
        }

        false
    }

    /// Get current sync status
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.is_synced.load(AtomicOrdering::Acquire)
    }

    /// Get reconnect count for monitoring
    pub fn get_reconnect_count(&self) -> u64 {
        self.reconnect_count.load(AtomicOrdering::Acquire)
    }

    /// Reset state for manual resync trigger
    pub fn mark_for_resync(&self) {
        self.is_synced.store(false, AtomicOrdering::Release);
        log_info("Manual resync triggered.");
    }
}

/// Simple logging macros for the orderbook module
#[inline]
fn log_info(msg: &str) {
    println!("[ORDERBOOK-INFO] {}", msg);
}

#[inline]
fn log_warn(msg: &str) {
    eprintln!("[ORDERBOOK-WARN] {}", msg);
}

#[inline]
fn log_error(msg: &str) {
    eprintln!("[ORDERBOOK-ERROR] {}", msg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_and_delta_sequence() {
        let manager = SnapshotManager::new();
        let book = L2OrderBook::new(12345);

        // Apply snapshot with lastUpdateId = 100
        manager.apply_snapshot(&book, &[(50000, 1.0)], &[(50100, 1.0)], 100);
        
        assert!(manager.is_ready());
        assert_eq!(manager.expected_update_id.load(AtomicOrdering::Relaxed), 101);

        // Apply valid delta with ID 101
        let result = manager.apply_delta(&book, 101, &[(50001, 0.5)], &[(50099, 0.5)]);
        assert!(result);
        assert_eq!(manager.expected_update_id.load(AtomicOrdering::Relaxed), 102);

        // Try to apply old delta (ID 100) - should be dropped
        let result = manager.apply_delta(&book, 100, &[], &[]);
        assert!(!result);

        // Try to apply future delta with gap (ID 105) - should trigger resync
        let result = manager.apply_delta(&book, 105, &[], &[]);
        assert!(!result);
        assert!(!manager.is_ready());
    }
}
