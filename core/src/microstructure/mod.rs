// core/src/microstructure/mod.rs
// =============================================================================
// MARKET MICROSTRUCTURE MODULE
// =============================================================================
// Aggregates footprint charts, liquidity sweeps, and order flow imbalance
// into a unified high-performance analytics engine.

pub mod footprint;
pub mod liquidity_sweeps;
pub mod imbalance;

pub use footprint::{FootprintEngine, FootprintLevel};
pub use liquidity_sweeps::{LiquiditySweepDetector, LiquidityEvent, SweepConfig};
pub use imbalance::{OrderFlowImbalance, ImbalanceSignal, ImbalanceConfig, BookSnapshot};

/// Unified microstructure state exposed to the main event loop
#[derive(Debug)]
pub struct MicrostructureState {
    pub footprint_snapshot: Vec<(f64, u64, u64, i64)>,
    pub liquidity_events: Vec<LiquidityEvent>,
    pub imbalance_signal: ImbalanceSignal,
}

/// High-level facade for all microstructure analytics
pub struct MicrostructureEngine {
    pub footprint: FootprintEngine,
    pub sweeps: LiquiditySweepDetector,
    pub imbalance: OrderFlowImbalance,
    tick_size_inv: f64,
}

impl MicrostructureEngine {
    pub fn new(tick_size: f64, range_ticks: i64, bucket_volume: u64) -> Self {
        let tick_size_inv = 1.0 / tick_size;
        Self {
            footprint: FootprintEngine::new(range_ticks, tick_size),
            sweeps: LiquiditySweepDetector::new(SweepConfig::default()),
            imbalance: OrderFlowImbalance::new(ImbalanceConfig::default(), bucket_volume),
            tick_size_inv,
        }
    }

    /// Process a single trade through all analytics engines
    #[inline]
    pub fn process_trade(&mut self, timestamp_ns: u64, price: f64, quantity: u64, 
                         is_buyer_maker: bool) {
        let price_tick = (price * self.tick_size_inv) as i64;
        
        // Update footprint
        self.footprint.process_trade(price, quantity, is_buyer_maker);
        
        // Update sweep detector
        self.sweeps.process_trade(timestamp_ns, price, quantity, is_buyer_maker, self.tick_size_inv);
        
        // Update imbalance tracker
        self.imbalance.process_trade(price_tick, quantity, is_buyer_maker);
    }

    /// Update order book state for imbalance calculation
    pub fn update_book(&mut self, bid: f64, ask: f64, bid_size: u64, ask_size: u64, timestamp_ns: u64) {
        self.sweeps.update_book(bid, ask, self.tick_size_inv);
        
        let snapshot = BookSnapshot {
            bid_price_tick: (bid * self.tick_size_inv) as i64,
            ask_price_tick: (ask * self.tick_size_inv) as i64,
            bid_size,
            ask_size,
            timestamp_ns,
        };
        self.imbalance.add_snapshot(snapshot);
    }

    /// Get unified state snapshot
    pub fn get_state(&mut self, timestamp_ns: u64) -> MicrostructureState {
        MicrostructureState {
            footprint_snapshot: self.footprint.snapshot(),
            liquidity_events: self.sweeps.drain_events(),
            imbalance_signal: self.imbalance.get_signal(timestamp_ns),
        }
    }

    /// Check for stop hunt patterns from candle data
    pub fn check_candle_stop_hunt(&mut self, ts: u64, high: f64, low: f64, 
                                   open: f64, close: f64, candle_ts: u64) {
        let high_tick = (high * self.tick_size_inv) as i64;
        let low_tick = (low * self.tick_size_inv) as i64;
        let open_tick = (open * self.tick_size_inv) as i64;
        let close_tick = (close * self.tick_size_inv) as i64;
        
        self.sweeps.check_stop_hunt(ts, high_tick, low_tick, open_tick, close_tick, candle_ts);
    }
}
