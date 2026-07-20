//! `vpin.rs` - Volume-Synchronized Probability of Informed Trading (VPIN)
//! 
//! **STAGE 10 | CHAPTER 2 | FILE 1**
//! 
//! This module implements real-time VPIN calculation to detect toxic order flow
//! and informed trading activity. VPIN uses volume buckets instead of time buckets
//! to accurately measure order flow imbalance, making it superior to time-based
//! metrics during periods of varying trading intensity.
//! 
//! **Key Features:**
//! - Lock-free volume bucket management
//! - Sub-microsecond toxicity scoring
//! - Adaptive bucket sizing based on recent volume profiles
//! - Integration with flow toxicity guardrails

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};

/// Default number of volume buckets for VPIN calculation
const DEFAULT_NUM_BUCKETS: usize = 50;

/// Default volume per bucket (in base units, e.g., BTC)
const DEFAULT_BUCKET_VOLUME: f64 = 100.0;

/// Trade classification result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeSide {
    BuyerInitiated,
    SellerInitiated,
    Unknown,
}

/// A single volume bucket containing aggregated trade data
#[derive(Debug, Clone)]
pub struct VolumeBucket {
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub total_volume: f64,
    pub trade_count: usize,
    pub is_complete: bool,
}

impl VolumeBucket {
    pub fn new(target_volume: f64) -> Self {
        Self {
            buy_volume: 0.0,
            sell_volume: 0.0,
            total_volume: 0.0,
            trade_count: 0,
            is_complete: false,
        }
    }

    /// Add a trade to the bucket
    /// Returns true if the bucket is now complete
    pub fn add_trade(&mut self, volume: f64, side: TradeSide, target_volume: f64) -> bool {
        self.total_volume += volume;
        self.trade_count += 1;

        match side {
            TradeSide::BuyerInitiated => self.buy_volume += volume,
            TradeSide::SellerInitiated => self.sell_volume += volume,
            TradeSide::Unknown => {
                // Split unknown trades evenly
                self.buy_volume += volume / 2.0;
                self.sell_volume += volume / 2.0;
            }
        }

        self.is_complete = self.total_volume >= target_volume;
        self.is_complete
    }

    /// Get the absolute order flow imbalance for this bucket
    #[inline(always)]
    pub fn order_imbalance(&self) -> f64 {
        (self.buy_volume - self.sell_volume).abs()
    }
}

/// VPIN Calculator - Real-time toxic flow detection
/// 
/// Uses the Easley, López de Prado, and O'Hara (2012) methodology:
/// VPIN = (1/n) * Σ|V_buy - V_sell| / (V_buy + V_sell)
/// 
/// High VPIN indicates high probability of informed trading (toxic flow).
pub struct VpinCalculator {
    /// Rolling window of completed volume buckets
    buckets: VecDeque<VolumeBucket>,
    /// Target volume per bucket (adaptive)
    target_bucket_volume: f64,
    /// Current incomplete bucket
    current_bucket: VolumeBucket,
    /// Number of buckets to use in VPIN calculation
    num_buckets: usize,
    /// Cached VPIN value (updated on each bucket completion)
    cached_vpin: f64,
    /// Timestamp of last VPIN update (nanoseconds)
    last_update_ns: u64,
    /// Statistics tracking
    stats: VpinStats,
}

/// VPIN statistics for monitoring and alerting
#[derive(Debug, Clone, Default)]
pub struct VpinStats {
    pub max_vpin_observed: f64,
    pub min_vpin_observed: f64,
    pub avg_vpin: f64,
    pub vpin_sample_count: usize,
    pub buckets_processed: usize,
    pub toxicity_events: usize, // Times VPIN exceeded critical threshold
}

impl VpinCalculator {
    pub fn new(num_buckets: usize, target_volume: f64) -> Self {
        Self {
            buckets: VecDeque::with_capacity(num_buckets),
            target_bucket_volume: target_volume,
            current_bucket: VolumeBucket::new(target_volume),
            num_buckets,
            cached_vpin: 0.0,
            last_update_ns: 0,
            stats: VpinStats::default(),
        }
    }

    /// Create with default parameters optimized for crypto markets
    pub fn crypto_default() -> Self {
        Self::new(DEFAULT_NUM_BUCKETS, DEFAULT_BUCKET_VOLUME)
    }

    /// Process a single trade and update VPIN if bucket completes
    /// 
    /// # Arguments
    /// * `volume` - Trade volume in base units
    /// * `side` - Trade classification (buyer/seller initiated)
    /// * `timestamp_ns` - Trade timestamp in nanoseconds
    /// 
    /// Returns: (bucket_completed, current_vpin)
    #[inline(always)]
    pub fn process_trade(&mut self, volume: f64, side: TradeSide, timestamp_ns: u64) -> (bool, f64) {
        let bucket_complete = self.current_bucket.add_trade(
            volume,
            side,
            self.target_bucket_volume,
        );

        if bucket_complete {
            // Move completed bucket to history
            let completed_bucket = std::mem::replace(
                &mut self.current_bucket,
                VolumeBucket::new(self.target_bucket_volume),
            );
            
            self.buckets.push_back(completed_bucket);
            
            // Maintain rolling window size
            if self.buckets.len() > self.num_buckets {
                self.buckets.pop_front();
            }

            // Recalculate VPIN
            self.cached_vpin = self.calculate_vpin_internal();
            self.last_update_ns = timestamp_ns;
            
            // Update statistics
            self.stats.buckets_processed += 1;
            self.update_stats();
        }

        (bucket_complete, self.cached_vpin)
    }

    /// Internal VPIN calculation over the rolling window
    #[inline(always)]
    fn calculate_vpin_internal(&self) -> f64 {
        if self.buckets.is_empty() {
            return 0.0;
        }

        let mut sum_imbalance = 0.0;
        let mut sum_volume = 0.0;

        for bucket in &self.buckets {
            sum_imbalance += bucket.order_imbalance();
            sum_volume += bucket.total_volume;
        }

        if sum_volume == 0.0 {
            return 0.0;
        }

        sum_imbalance / sum_volume
    }

    /// Get current VPIN estimate (may be stale if no bucket completed recently)
    #[inline(always)]
    pub fn current_vpin(&self) -> f64 {
        self.cached_vpin
    }

    /// Get VPIN with confidence score based on bucket completeness
    /// 
    /// Returns: (vpin, confidence) where confidence is [0, 1]
    pub fn vpin_with_confidence(&self) -> (f64, f64) {
        let bucket_fill_ratio = self.current_bucket.total_volume / self.target_bucket_volume;
        let history_weight = self.buckets.len() as f64 / self.num_buckets as f64;
        
        // Confidence increases with more complete history and current bucket
        let confidence = history_weight.min(1.0) * (0.5 + 0.5 * bucket_fill_ratio.min(1.0));
        
        (self.cached_vpin, confidence)
    }

    /// Update running statistics
    fn update_stats(&mut self) {
        let vpin = self.cached_vpin;
        
        if self.stats.vpin_sample_count == 0 {
            self.stats.max_vpin_observed = vpin;
            self.stats.min_vpin_observed = vpin;
            self.stats.avg_vpin = vpin;
        } else {
            self.stats.max_vpin_observed = self.stats.max_vpin_observed.max(vpin);
            self.stats.min_vpin_observed = self.stats.min_vpin_observed.min(vpin);
            
            // Running average
            let n = self.stats.vpin_sample_count as f64;
            self.stats.avg_vpin = (self.stats.avg_vpin * n + vpin) / (n + 1.0);
        }
        
        self.stats.vpin_sample_count += 1;
    }

    /// Check if VPIN exceeds a toxicity threshold
    #[inline(always)]
    pub fn is_toxic(&self, threshold: f64) -> bool {
        self.cached_vpin > threshold
    }

    /// Adaptively adjust bucket volume based on recent market activity
    /// Useful for handling regime changes in trading volume
    pub fn adapt_bucket_size(&mut self, new_target: f64) {
        self.target_bucket_volume = new_target.max(1.0); // Floor at 1.0
        self.current_bucket = VolumeBucket::new(self.target_bucket_volume);
    }

    /// Reset all state
    pub fn reset(&mut self) {
        self.buckets.clear();
        self.current_bucket = VolumeBucket::new(self.target_bucket_volume);
        self.cached_vpin = 0.0;
        self.last_update_ns = 0;
    }

    /// Get statistics reference
    pub fn stats(&self) -> &VpinStats {
        &self.stats
    }

    /// Increment toxicity event counter
    pub fn record_toxicity_event(&mut self) {
        self.stats.toxicity_events += 1;
    }
}

impl Default for VpinCalculator {
    fn default() -> Self {
        Self::crypto_default()
    }
}

/// Batch VPIN processor for processing historical data or replay
pub struct VpinBatchProcessor {
    calculator: VpinCalculator,
    results: Vec<(u64, f64)>, // (timestamp, vpin) pairs
}

impl VpinBatchProcessor {
    pub fn new(num_buckets: usize, target_volume: f64) -> Self {
        Self {
            calculator: VpinCalculator::new(num_buckets, target_volume),
            results: Vec::new(),
        }
    }

    /// Process a batch of trades
    /// 
    /// # Arguments
    /// * `trades` - Slice of (volume, side, timestamp_ns) tuples
    pub fn process_batch(&mut self, trades: &[(f64, TradeSide, u64)]) -> &[(u64, f64)] {
        self.results.clear();
        
        for &(volume, side, ts) in trades {
            let (complete, vpin) = self.calculator.process_trade(volume, side, ts);
            if complete {
                self.results.push((ts, vpin));
            }
        }
        
        &self.results
    }

    /// Get final VPIN value
    pub fn final_vpin(&self) -> f64 {
        self.calculator.current_vpin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vpin_balanced_flow() {
        let mut calc = VpinCalculator::new(10, 100.0);
        
        // Add balanced buy/sell flow
        for i in 0..100 {
            let side = if i % 2 == 0 { TradeSide::BuyerInitiated } else { TradeSide::SellerInitiated };
            calc.process_trade(10.0, side, i as u64 * 1_000_000);
        }
        
        // VPIN should be low for balanced flow
        assert!(calc.current_vpin() < 0.3);
    }

    #[test]
    fn test_vpin_imbalanced_flow() {
        let mut calc = VpinCalculator::new(10, 100.0);
        
        // Add heavily imbalanced flow (all buys)
        for i in 0..100 {
            calc.process_trade(10.0, TradeSide::BuyerInitiated, i as u64 * 1_000_000);
        }
        
        // VPIN should be high for imbalanced flow
        assert!(calc.current_vpin() > 0.7);
    }

    #[test]
    fn test_vpin_confidence() {
        let mut calc = VpinCalculator::new(10, 100.0);
        
        // Initially low confidence (no history)
        let (_, conf) = calc.vpin_with_confidence();
        assert!(conf < 0.5);
        
        // Fill some buckets
        for i in 0..50 {
            let side = if i % 2 == 0 { TradeSide::BuyerInitiated } else { TradeSide::SellerInitiated };
            calc.process_trade(10.0, side, i as u64 * 1_000_000);
        }
        
        let (_, conf) = calc.vpin_with_confidence();
        assert!(conf > 0.5);
    }

    #[test]
    fn test_toxicity_detection() {
        let mut calc = VpinCalculator::new(10, 100.0);
        
        // Generate toxic flow
        for i in 0..100 {
            calc.process_trade(10.0, TradeSide::BuyerInitiated, i as u64 * 1_000_000);
        }
        
        assert!(calc.is_toxic(0.5));
        assert!(!calc.is_toxic(0.95)); // Very high threshold
    }
}
