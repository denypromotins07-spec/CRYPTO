//! core/src/execution/dynamic_algo_router.rs
//!
//! Dynamic router that automatically switches between TWAP, VWAP, and Arrival Price
//! (Implementation Shortfall) algorithms based on real-time market conditions.
//!
//! Decision Matrix:
//! - Low Volatility + Large Size -> TWAP (Time-Weighted)
//! - High Volume + Normal Size -> VWAP (Volume-Weighted)
//! - High Volatility + Urgent -> Arrival Price (Minimize Market Impact)
//!
//! Target Hardware: AMD Ryzen AI 5 (Cache-line aligned structures)
//! Memory Constraint: Pre-allocated strategy buffers.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Execution algorithm types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExecAlgo {
    Twap = 0,
    Vwap = 1,
    ArrivalPrice = 2,
    Iceberg = 3,
}

/// Market condition snapshot used for decision making.
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    pub volatility_1m: f64,      // Annualized vol over last 1min
    pub volume_ratio: f64,       // Current vol / Avg vol
    pub spread_bps: f64,         // Bid-ask spread in bps
    pub order_book_depth: u64,   // Depth within 10bps (in quote units)
    pub urgency_score: f64,      // 0.0 to 1.0 (higher = more urgent)
}

/// Parameters for TWAP execution.
#[derive(Debug, Clone)]
pub struct TwapParams {
    pub total_quantity: f64,
    pub duration_secs: u64,
    pub num_slices: u64,
    pub randomize_timing: bool,
}

/// Parameters for VWAP execution.
#[derive(Debug, Clone)]
pub struct VwapParams {
    pub total_quantity: f64,
    pub volume_profile: Vec<f64>, // Normalized volume profile for the day
    pub max_participation_rate: f64, // Max % of market volume to take
}

/// Parameters for Arrival Price (Implementation Shortfall) execution.
#[derive(Debug, Clone)]
pub struct ArrivalPriceParams {
    pub total_quantity: f64,
    pub risk_aversion: f64,      // Higher = more aggressive
    pub price_impact_coeff: f64, // Estimated market impact coefficient
    pub max_duration_secs: u64,
}

/// Result of the algo selection process.
#[derive(Debug, Clone)]
pub struct AlgoDecision {
    pub algo: ExecAlgo,
    pub confidence: f64,         // 0.0 to 1.0
    pub reason_code: u8,         // Diagnostic code
    pub estimated_slippage_bps: f64,
}

/// Lock-free dynamic algorithm router.
pub struct DynamicAlgoRouter {
    current_algo: AtomicU8,
    last_switch_time: AtomicU64,
    switch_count: AtomicU64,
    
    // Thresholds (tunable at runtime via config)
    pub vol_threshold_low: f64,
    pub vol_threshold_high: f64,
    pub volume_ratio_threshold: f64,
    pub min_switch_interval_ms: u64,
}

impl DynamicAlgoRouter {
    pub fn new() -> Self {
        Self {
            current_algo: AtomicU8::new(ExecAlgo::Vwap as u8),
            last_switch_time: AtomicU64::new(0),
            switch_count: AtomicU64::new(0),
            
            // Default thresholds (should be calibrated per asset)
            vol_threshold_low: 0.3,    // 30% annualized
            vol_threshold_high: 0.8,   // 80% annualized
            volume_ratio_threshold: 1.5,
            min_switch_interval_ms: 5000, // Min 5 seconds between switches
        }
    }

    /// Evaluate market conditions and select optimal execution algorithm.
    /// 
    /// This function is designed to be called frequently (every tick or every 100ms)
    /// but will only update the internal state if a switch is warranted.
    pub fn evaluate_and_select(&self, snapshot: &MarketSnapshot, order_size_quote: f64) -> AlgoDecision {
        let mut decision = self.select_best_algo(snapshot, order_size_quote);
        
        // Check if we should actually switch
        let now_ms = Instant::now().elapsed().as_millis() as u64;
        let last_switch = self.last_switch_time.load(Ordering::Relaxed);
        let current_algo_byte = self.current_algo.load(Ordering::Relaxed);
        
        if current_algo_byte != decision.algo as u8 {
            // Only switch if enough time has passed (prevent churning)
            if now_ms - last_switch >= self.min_switch_interval_ms {
                self.current_algo.store(decision.algo as u8, Ordering::Relaxed);
                self.last_switch_time.store(now_ms, Ordering::Relaxed);
                self.switch_count.fetch_add(1, Ordering::Relaxed);
                decision.reason_code |= 0x80; // Set "switched" bit
            } else {
                // Keep current algo, but report what we would have chosen
                decision.algo = unsafe { std::mem::transmute(current_algo_byte) };
                decision.reason_code |= 0x40; // Set "cooldown" bit
            }
        }
        
        decision
    }

    /// Core logic for selecting the best algorithm based on conditions.
    fn select_best_algo(&self, snapshot: &MarketSnapshot, order_size_quote: f64) -> AlgoDecision {
        let MarketSnapshot {
            volatility_1m,
            volume_ratio,
            spread_bps,
            order_book_depth,
            urgency_score,
        } = *snapshot;

        // Heuristic scoring system
        let mut twap_score = 0.0;
        let mut vwap_score = 0.0;
        let mut arrival_score = 0.0;
        let mut iceberg_score = 0.0;

        // TWAP prefers: Low vol, large size relative to depth, low urgency
        if volatility_1m < self.vol_threshold_low {
            twap_score += 0.3;
        }
        if order_size_quote > *order_book_depth as f64 * 0.5 {
            twap_score += 0.2;
        }
        if urgency_score < 0.3 {
            twap_score += 0.2;
        }
        twap_score += 0.3 * (1.0 - urgency_score); // Linear component

        // VWAP prefers: Normal vol, high volume ratio, normal size
        if volatility_1m >= self.vol_threshold_low && volatility_1m < self.vol_threshold_high {
            vwap_score += 0.3;
        }
        if volume_ratio > self.volume_ratio_threshold {
            vwap_score += 0.3;
        }
        if urgency_score >= 0.3 && urgency_score < 0.7 {
            vwap_score += 0.2;
        }
        vwap_score += 0.2 * volume_ratio.min(1.0);

        // Arrival Price prefers: High vol, high urgency, any size
        if volatility_1m >= self.vol_threshold_high {
            arrival_score += 0.3;
        }
        if urgency_score >= 0.7 {
            arrival_score += 0.4;
        }
        if spread_bps > 10.0 {
            arrival_score += 0.1; // Wide spreads favor aggressive execution
        }
        arrival_score += 0.2 * urgency_score;

        // Iceberg: Very large orders relative to depth
        if order_size_quote > *order_book_depth as f64 * 2.0 {
            iceberg_score += 0.5;
        }
        if spread_bps > 15.0 {
            iceberg_score += 0.3;
        }
        iceberg_score += 0.2 * (order_size_quote / (*order_book_depth as f64 + 1.0)).min(1.0);

        // Select winner
        let scores = [
            (ExecAlgo::Twap, twap_score),
            (ExecAlgo::Vwap, vwap_score),
            (ExecAlgo::ArrivalPrice, arrival_score),
            (ExecAlgo::Iceberg, iceberg_score),
        ];

        let (best_algo, best_score) = scores.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        // Estimate slippage based on algo and conditions
        let est_slippage = self.estimate_slippage(*best_algo, snapshot, order_size_quote);

        AlgoDecision {
            algo: *best_algo,
            confidence: best_score / 1.5, // Normalize roughly to 0-1
            reason_code: 0x00,
            estimated_slippage_bps: est_slippage,
        }
    }

    /// Rough slippage estimation model.
    fn estimate_slippage(&self, algo: ExecAlgo, snapshot: &MarketSnapshot, size: f64) -> f64 {
        let base_impact = size / (snapshot.order_book_depth as f64 + 1.0);
        
        match algo {
            ExecAlgo::Twap => {
                // TWAP: Time-based, moderate impact
                base_impact * 0.5 + snapshot.spread_bps * 0.5
            }
            ExecAlgo::Vwap => {
                // VWAP: Volume-based, lower impact in liquid markets
                base_impact * 0.3 + snapshot.spread_bps * 0.3
            }
            ExecAlgo::ArrivalPrice => {
                // Arrival: Aggressive, higher immediate impact but lower timing risk
                base_impact * 0.8 + snapshot.spread_bps * 0.2
            }
            ExecAlgo::Iceberg => {
                // Iceberg: Hidden liquidity, lowest visible impact but longer duration
                base_impact * 0.2 + snapshot.spread_bps * 0.8
            }
        }
    }

    pub fn get_current_algo(&self) -> ExecAlgo {
        unsafe { std::mem::transmute(self.current_algo.load(Ordering::Relaxed)) }
    }

    pub fn get_switch_count(&self) -> u64 {
        self.switch_count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_low_vol_large_order_selects_twap() {
        let router = DynamicAlgoRouter::new();
        let snapshot = MarketSnapshot {
            volatility_1m: 0.2, // Low vol
            volume_ratio: 0.8,
            spread_bps: 5.0,
            order_book_depth: 100_000,
            urgency_score: 0.1,
        };
        
        let decision = router.evaluate_and_select(&snapshot, 200_000.0); // Large order
        
        assert_eq!(decision.algo, ExecAlgo::Twap);
        assert!(decision.confidence > 0.5);
    }

    #[test]
    fn test_high_vol_urgent_selects_arrival() {
        let router = DynamicAlgoRouter::new();
        let snapshot = MarketSnapshot {
            volatility_1m: 1.2, // High vol
            volume_ratio: 1.0,
            spread_bps: 15.0,
            order_book_depth: 50_000,
            urgency_score: 0.9,
        };
        
        let decision = router.evaluate_and_select(&snapshot, 50_000.0);
        
        assert_eq!(decision.algo, ExecAlgo::ArrivalPrice);
    }

    #[test]
    fn test_normal_conditions_selects_vwap() {
        let router = DynamicAlgoRouter::new();
        let snapshot = MarketSnapshot {
            volatility_1m: 0.5, // Normal vol
            volume_ratio: 2.0,  // High volume
            spread_bps: 3.0,
            order_book_depth: 500_000,
            urgency_score: 0.4,
        };
        
        let decision = router.evaluate_and_select(&snapshot, 10_000.0);
        
        assert_eq!(decision.algo, ExecAlgo::Vwap);
    }
}
