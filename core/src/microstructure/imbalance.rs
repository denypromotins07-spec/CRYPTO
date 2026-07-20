// core/src/microstructure/imbalance.rs
// =============================================================================
// ORDER FLOW IMBALANCE & HIDDEN LIQUIDITY DETECTOR
// =============================================================================
// Purpose: Calculates real-time order flow imbalance and detects hidden liquidity
// (Iceberg orders) using statistical thresholds and volume-weighted analysis.
//
// Key Metrics:
// - OFI (Order Flow Imbalance): Net pressure from bid/ask queue changes
// - Iceberg Detection: Repeated trades at same price without queue depletion
// - VPIN (Volume-Synchronized Probability of Informed Trading)

use std::collections::VecDeque;

/// Snapshot of order book state for imbalance calculation
#[derive(Debug, Clone, Copy)]
pub struct BookSnapshot {
    pub bid_price_tick: i64,
    pub ask_price_tick: i64,
    pub bid_size: u64,
    pub ask_size: u64,
    pub timestamp_ns: u64,
}

/// Result of imbalance analysis
#[derive(Debug, Clone)]
pub struct ImbalanceSignal {
    pub timestamp_ns: u64,
    pub ofi: f64,           // Order Flow Imbalance (-1 to 1)
    pub vpins: f64,         // VPIN metric
    pub iceberg_prob: f64,  // Probability of iceberg [0, 1]
    pub hidden_bid_vol: u64,
    pub hidden_ask_vol: u64,
}

/// Configuration for detection algorithms
#[derive(Debug, Clone)]
pub struct ImbalanceConfig {
    pub window_size: usize,      // Number of snapshots for rolling window
    pub iceberg_threshold: f64,  // Confidence threshold for iceberg detection
    pub vpin_buckets: usize,     // Number of volume buckets for VPIN
}

impl Default for ImbalanceConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            iceberg_threshold: 0.75,
            vpin_buckets: 50,
        }
    }
}

/// Tracks a single volume bucket for VPIN calculation
struct VolumeBucket {
    buy_volume: u64,
    sell_volume: u64,
}

/// Core Imbalance & Hidden Liquidity Engine
pub struct OrderFlowImbalance {
    config: ImbalanceConfig,
    /// Rolling window of book snapshots
    snapshots: VecDeque<BookSnapshot>,
    /// Volume buckets for VPIN
    buckets: VecDeque<VolumeBucket>,
    current_bucket: VolumeBucket,
    current_bucket_volume: u64,
    bucket_target_volume: u64,
    /// Iceberg detection state
    last_trade_price_tick: i64,
    consecutive_same_price: u32,
    volume_at_price: u64,
    /// Running totals for VPIN
    total_buy_vol: u64,
    total_sell_vol: u64,
}

impl OrderFlowImbalance {
    pub fn new(config: ImbalanceConfig, target_bucket_volume: u64) -> Self {
        Self {
            config,
            snapshots: VecDeque::with_capacity(config.window_size),
            buckets: VecDeque::with_capacity(config.vpin_buckets),
            current_bucket: VolumeBucket { buy_volume: 0, sell_volume: 0 },
            current_bucket_volume: 0,
            bucket_target_volume: target_bucket_volume,
            last_trade_price_tick: 0,
            consecutive_same_price: 0,
            volume_at_price: 0,
            total_buy_vol: 0,
            total_sell_vol: 0,
        }
    }

    /// Add a new order book snapshot
    #[inline]
    pub fn add_snapshot(&mut self, snapshot: BookSnapshot) {
        if self.snapshots.len() >= self.config.window_size {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(snapshot);
    }

    /// Process a trade for iceberg detection and VPIN
    #[inline]
    pub fn process_trade(&mut self, price_tick: i64, quantity: u64, is_buyer_maker: bool) {
        // Update VPIN buckets
        if is_buyer_maker {
            self.current_bucket.sell_volume += quantity;
            self.total_sell_vol += quantity;
        } else {
            self.current_bucket.buy_volume += quantity;
            self.total_buy_vol += quantity;
        }
        
        self.current_bucket_volume += quantity;
        
        // Rotate bucket if full
        if self.current_bucket_volume >= self.bucket_target_volume {
            self.buckets.push_back(std::mem::replace(
                &mut self.current_bucket, 
                VolumeBucket { buy_volume: 0, sell_volume: 0 }
            ));
            self.current_bucket_volume = 0;
            
            if self.buckets.len() > self.config.vpin_buckets {
                let old = self.buckets.pop_front().unwrap();
                self.total_buy_vol = self.total_buy_vol.saturating_sub(old.buy_volume);
                self.total_sell_vol = self.total_sell_vol.saturating_sub(old.sell_volume);
            }
        }

        // Iceberg detection: track repeated trades at same price
        if price_tick == self.last_trade_price_tick {
            self.consecutive_same_price += 1;
            self.volume_at_price += quantity;
        } else {
            self.consecutive_same_price = 1;
            self.volume_at_price = quantity;
            self.last_trade_price_tick = price_tick;
        }
    }

    /// Calculate Order Flow Imbalance (OFI)
    /// OFI measures the net flow from changes in bid/ask queues
    pub fn calculate_ofi(&self) -> f64 {
        if self.snapshots.len() < 2 {
            return 0.0;
        }

        let mut ofi_sum = 0.0;
        let mut count = 0;

        // Calculate OFI as sum of bid size changes + negative ask size changes
        // Positive OFI = buying pressure, Negative OFI = selling pressure
        let mut prev = self.snapshots.front().unwrap();
        
        for snap in self.snapshots.iter().skip(1) {
            let bid_change = snap.bid_size as i64 - prev.bid_size as i64;
            let ask_change = snap.ask_size as i64 - prev.ask_size as i64;
            
            // OFI contribution: bid increase is positive, ask increase is negative
            let contribution = (bid_change - ask_change) as f64;
            ofi_sum += contribution;
            count += 1;
            
            prev = snap;
        }

        if count == 0 { return 0.0; }
        
        // Normalize to [-1, 1] using tanh for smooth bounding
        let raw_ofi = ofi_sum / count as f64;
        raw_ofi.tanh()
    }

    /// Calculate VPIN (Volume-Synchronized Probability of Informed Trading)
    pub fn calculate_vpin(&self) -> f64 {
        if self.buckets.is_empty() {
            return 0.0;
        }

        let mut total_abs_imbalance = 0u64;
        let mut total_volume = 0u64;

        for bucket in &self.buckets {
            let imbalance = (bucket.buy_volume as i64 - bucket.sell_volume as i64).unsigned_abs();
            total_abs_imbalance += imbalance;
            total_volume += bucket.buy_volume + bucket.sell_volume;
        }

        // Also include current partial bucket
        let curr_imbalance = (self.current_bucket.buy_volume as i64 - self.current_bucket.sell_volume as i64).unsigned_abs();
        total_abs_imbalance += curr_imbalance;
        total_volume += self.current_bucket.buy_volume + self.current_bucket.sell_volume;

        if total_volume == 0 { return 0.0; }

        total_abs_imbalance as f64 / total_volume as f64
    }

    /// Detect iceberg orders based on repeated trades at same price
    pub fn detect_iceberg(&self) -> (f64, u64, u64) {
        // Heuristic: Many trades at same price with high volume suggests hidden liquidity
        // Probability increases with consecutive trades and total volume
        
        if self.consecutive_same_price < 3 {
            return (0.0, 0, 0);
        }

        // Simple logistic-like function for probability
        let consec_factor = (self.consecutive_same_price as f64 / 10.0).min(1.0);
        let vol_factor = (self.volume_at_price as f64 / 100_000.0).min(1.0);
        
        let probability = (consec_factor * 0.6 + vol_factor * 0.4).min(1.0);
        
        // Estimate hidden volume (volume beyond what's visible)
        // This is a simplification; real implementation would track queue depletion
        let estimated_hidden = if probability > self.config.iceberg_threshold {
            (self.volume_at_price as f64 * 0.5) as u64 // Assume 50% is hidden
        } else {
            0
        };

        (probability, estimated_hidden, estimated_hidden)
    }

    /// Generate comprehensive signal
    pub fn get_signal(&mut self, timestamp_ns: u64) -> ImbalanceSignal {
        let ofi = self.calculate_ofi();
        let vpin = self.calculate_vpin();
        let (iceberg_prob, hidden_bid, hidden_ask) = self.detect_iceberg();

        ImbalanceSignal {
            timestamp_ns,
            ofi,
            vpins: vpin,
            iceberg_prob,
            hidden_bid_vol: hidden_bid,
            hidden_ask_vol: hidden_ask,
        }
    }

    /// Reset all state (e.g., on session change)
    pub fn reset(&mut self) {
        self.snapshots.clear();
        self.buckets.clear();
        self.current_bucket = VolumeBucket { buy_volume: 0, sell_volume: 0 };
        self.current_bucket_volume = 0;
        self.consecutive_same_price = 0;
        self.volume_at_price = 0;
        self.total_buy_vol = 0;
        self.total_sell_vol = 0;
    }
}
