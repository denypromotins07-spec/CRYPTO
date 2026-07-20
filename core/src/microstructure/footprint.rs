// core/src/microstructure/footprint.rs
// =============================================================================
// HIGH-PERFORMANCE FOOTPRINT CHART ENGINE
// =============================================================================
// Purpose: Tracks bid/ask volume, delta, and cumulative delta at every price level.
// Architecture: Uses pre-allocated contiguous memory arrays (Vec<u64>) to avoid
// heap allocations during the hot path. Designed for cache-line locality on AMD Ryzen.
//
// Memory Strategy:
// - Fixed-size ring buffer for price levels to prevent unbounded growth.
// - Zero-copy updates using direct index arithmetic.
// - SIMD-ready structures for batch delta calculations.

use std::collections::HashMap;

/// Represents a single price level in the footprint chart.
/// Aligned to 64 bytes (cache line) to prevent false sharing in multi-threaded contexts.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct FootprintLevel {
    pub price_tick: i64,          // Price in ticks (integer)
    pub bid_volume: u64,          // Total volume traded at bid (aggressive sells)
    pub ask_volume: u64,          // Total volume traded at ask (aggressive buys)
    pub delta: i64,               // Ask Volume - Bid Volume
    pub cumulative_delta: i64,    // Running sum of delta up to this level
    pub trade_count: u32,         // Number of trades at this level
    pub _padding: [u8; 12],       // Explicit padding to ensure 64-byte alignment
}

impl Default for FootprintLevel {
    fn default() -> Self {
        Self {
            price_tick: 0,
            bid_volume: 0,
            ask_volume: 0,
            delta: 0,
            cumulative_delta: 0,
            trade_count: 0,
            _padding: [0; 12],
        }
    }
}

/// The core Footprint Engine.
/// Manages a sliding window of price levels around the current market price.
pub struct FootprintEngine {
    /// Map from price_tick to index in the levels array for O(1) lookup
    price_map: HashMap<i64, usize>,
    /// Contiguous memory block for price levels
    levels: Vec<FootprintLevel>,
    /// Current market price tick
    current_price_tick: i64,
    /// Range of levels to track (e.g., +/- 50 ticks)
    range_ticks: i64,
    /// Global cumulative delta for the session
    global_cumulative_delta: i64,
    /// Tick size of the asset (e.g., 0.01 -> 1 tick)
    tick_size_inv: f64, // Inverse tick size for fast conversion
}

impl FootprintEngine {
    /// Initialize the engine with a specific range and tick size.
    /// Pre-allocates memory to avoid runtime allocations.
    pub fn new(range_ticks: i64, tick_size: f64) -> Self {
        let capacity = (range_ticks * 2 + 1) as usize;
        let mut levels = Vec::with_capacity(capacity);
        levels.resize_with(capacity, FootprintLevel::default);

        Self {
            price_map: HashMap::with_capacity(capacity),
            levels,
            current_price_tick: 0,
            range_ticks,
            global_cumulative_delta: 0,
            tick_size_inv: 1.0 / tick_size,
        }
    }

    /// Convert float price to integer tick representation.
    #[inline(always)]
    fn price_to_tick(&self, price: f64) -> i64 {
        (price * self.tick_size_inv).round() as i64
    }

    /// Process a single trade update.
    /// This is the HOT PATH. Must be extremely fast.
    #[inline]
    pub fn process_trade(&mut self, price: f64, quantity: u64, is_buyer_maker: bool) {
        let tick = self.price_to_tick(price);
        
        let delta_increment = if is_buyer_maker { -(quantity as i64) } else { quantity as i64 };
        self.global_cumulative_delta += delta_increment;

        if (tick - self.current_price_tick).abs() > self.range_ticks {
            self.shift_window(tick);
        }

        let idx = self.get_or_create_level(tick);
        let level = &mut self.levels[idx];

        if is_buyer_maker {
            level.bid_volume += quantity;
        } else {
            level.ask_volume += quantity;
        }
        
        level.delta = (level.ask_volume as i64) - (level.bid_volume as i64);
        level.trade_count += 1;
    }

    fn shift_window(&mut self, new_center_tick: i64) {
        self.current_price_tick = new_center_tick;
        let min_tick = new_center_tick - self.range_ticks;
        let max_tick = new_center_tick + self.range_ticks;

        self.price_map.retain(|&tick, &idx| {
            if tick < min_tick || tick > max_tick {
                self.levels[idx] = FootprintLevel::default();
                false
            } else {
                true
            }
        });
    }

    #[inline]
    fn get_or_create_level(&mut self, tick: i64) -> usize {
        if let Some(&idx) = self.price_map.get(&tick) {
            idx
        } else {
            let idx = ((tick - (self.current_price_tick - self.range_ticks)) as usize) % self.levels.len();
            if self.levels[idx].price_tick != tick && self.levels[idx].price_tick != 0 {
                 self.levels[idx] = FootprintLevel::default();
            }
            self.levels[idx].price_tick = tick;
            self.price_map.insert(tick, idx);
            idx
        }
    }

    pub fn get_imbalance_ratio(&self, tick: i64) -> f64 {
        if let Some(&idx) = self.price_map.get(&tick) {
            let level = self.levels[idx];
            let total = level.bid_volume + level.ask_volume;
            if total == 0 { return 0.0; }
            (level.ask_volume as f64 - level.bid_volume as f64) / total as f64
        } else {
            0.0
        }
    }

    pub fn snapshot(&self) -> Vec<(f64, u64, u64, i64)> {
        let mut result = Vec::with_capacity(self.price_map.len());
        for (&tick, &idx) in &self.price_map {
            let lvl = self.levels[idx];
            let price = tick as f64 / self.tick_size_inv;
            result.push((price, lvl.bid_volume, lvl.ask_volume, lvl.delta));
        }
        result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        result
    }
}
