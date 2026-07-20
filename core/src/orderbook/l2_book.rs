// core/src/orderbook/l2_book.rs
// =============================================================================
// STAGE 2 - CHAPTER 1 - FILE 1
// Focus: Lock-free, cache-aligned L2 Order Book for microsecond lookups.
// Target: AMD Ryzen AI 5 (AVX2/AVX-512 capable), minimizing cache misses.
// =============================================================================

use std::cmp::Ordering;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Cache line size for x86_64 (typically 64 bytes).
/// We use this to pad structures to prevent false sharing between threads.
const CACHE_LINE_SIZE: usize = 64;

/// Represents a single price level in the order book.
/// Packed to minimize memory footprint and maximize cache locality.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PriceLevel {
    pub price: u64, // Stored as integer (scaled) to avoid float overhead
    pub quantity: f64,
    pub order_count: u32,
}

/// A fixed-size, contiguous array for bids or asks.
/// Using a fixed size avoids dynamic allocation during hot paths.
/// Depth of 50 is standard for HFT L2 data; can be increased if RAM allows.
const MAX_DEPTH: usize = 50;

#[repr(C)]
pub struct SideBook {
    levels: [PriceLevel; MAX_DEPTH],
    depth: AtomicU64, // Current number of valid levels
    // Padding to ensure next data structure starts on new cache line
    _pad: [u8; CACHE_LINE_SIZE - std::mem::size_of::<[PriceLevel; MAX_DEPTH]>() - 8],
}

impl SideBook {
    pub fn new() -> Self {
        SideBook {
            levels: [PriceLevel { price: 0, quantity: 0.0, order_count: 0 }; MAX_DEPTH],
            depth: AtomicU64::new(0),
            _pad: [0; CACHE_LINE_SIZE - std::mem::size_of::<[PriceLevel; MAX_DEPTH]>() - 8],
        }
    }

    /// Updates a level or inserts maintaining sorted order.
    /// Bids: Descending (Highest first)
    /// Asks: Ascending (Lowest first)
    /// Returns true if updated, false if ignored (quantity 0 removal handled externally).
    #[inline]
    pub fn update(&self, price: u64, quantity: f64, count: u32, is_bid: bool) {
        let current_depth = self.depth.load(AtomicOrdering::Relaxed) as usize;
        
        // Binary search for existing price or insertion point
        let mut low = 0;
        let mut high = current_depth;
        let mut found_idx = None;

        while low < high {
            let mid = (low + high) / 2;
            let mid_price = self.levels[mid].price;
            
            let cmp = if is_bid {
                // Bids: Higher price comes first (Descending)
                price.cmp(&mid_price)
            } else {
                // Asks: Lower price comes first (Ascending)
                mid_price.cmp(&price)
            };

            match cmp {
                Ordering::Equal => {
                    found_idx = Some(mid);
                    break;
                }
                Ordering::Less => low = mid + 1,
                Ordering::Greater => high = mid,
            }
        }

        if let Some(idx) = found_idx {
            if quantity == 0.0 {
                // Remove level: Shift subsequent elements left
                // Note: In a true lock-free scenario, this requires careful coordination.
                // Here we assume single-threaded updater per side for simplicity in this stage.
                for i in idx..current_depth - 1 {
                    self.levels[i] = self.levels[i + 1];
                }
                self.levels[current_depth - 1] = PriceLevel { price: 0, quantity: 0.0, order_count: 0 };
                self.depth.fetch_sub(1, AtomicOrdering::Release);
            } else {
                // Update existing
                // Direct memory write: Safe because we own the update thread
                unsafe {
                    let ptr = &self.levels[idx] as *const PriceLevel as *mut PriceLevel;
                    (*ptr).quantity = quantity;
                    (*ptr).order_count = count;
                }
            }
        } else if quantity > 0.0 && current_depth < MAX_DEPTH {
            // Insert new level: Shift right
            let insert_idx = low;
            for i in (insert_idx + 1..=current_depth).rev() {
                if i < MAX_DEPTH {
                    self.levels[i] = self.levels[i - 1];
                }
            }
            if insert_idx < MAX_DEPTH {
                self.levels[insert_idx] = PriceLevel { price, quantity, order_count: count };
                self.depth.fetch_add(1, AtomicOrdering::Release);
            }
        }
    }

    #[inline]
    pub fn get_best(&self) -> Option<&PriceLevel> {
        if self.depth.load(AtomicOrdering::Acquire) > 0 {
            Some(&self.levels[0])
        } else {
            None
        }
    }
    
    #[inline]
    pub fn get_depth(&self) -> usize {
        self.depth.load(AtomicOrdering::Acquire) as usize
    }
}

/// The complete L2 Order Book containing both Bids and Asks.
/// Designed for cache-line separation to allow parallel updates if needed.
pub struct L2OrderBook {
    pub bids: SideBook,
    pub asks: SideBook,
    pub last_update_id: AtomicU64,
    pub symbol_hash: u64, // Fast identifier
}

impl L2OrderBook {
    pub fn new(symbol_hash: u64) -> Self {
        L2OrderBook {
            bids: SideBook::new(),
            asks: SideBook::new(),
            last_update_id: AtomicU64::new(0),
            symbol_hash,
        }
    }

    /// Apply a delta update from Binance WebSocket
    pub fn apply_delta(&self, update_id: u64, bids: &[(u64, f64)], asks: &[(u64, f64)]) {
        // Sequence validation happens in snapshot_manager, here we assume valid ordered updates
        self.last_update_id.store(update_id, AtomicOrdering::Release);

        for &(price, qty) in bids {
            self.bids.update(price, qty, 1, true); // Count approx 1 for simplicity in hot path
        }
        for &(price, qty) in asks {
            self.asks.update(price, qty, 1, false);
        }
    }

    /// Get Mid Price (fast calculation for indicators)
    #[inline]
    pub fn get_mid_price(&self) -> Option<f64> {
        let best_bid = self.bids.get_best()?;
        let best_ask = self.asks.get_best()?;
        // Convert u64 scaled price back to f64 only when necessary
        Some((best_bid.price as f64 + best_ask.price as f64) / 2.0)
    }

    /// Get Spread in ticks
    #[inline]
    pub fn get_spread_ticks(&self) -> Option<u64> {
        let best_bid = self.bids.get_best()?;
        let best_ask = self.asks.get_best()?;
        Some(best_ask.price.checked_sub(best_bid.price)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_book_insertion() {
        let book = L2OrderBook::new(12345);
        // Insert asks: 100, 102, 105
        book.apply_delta(1, &[], &[(10500, 1.0), (10000, 1.0), (10200, 1.0)]);
        
        assert_eq!(book.asks.get_best().unwrap().price, 10000);
        assert_eq!(book.asks.get_depth(), 3);
    }
}
