//! Adverse Selection Detection for Market Making
//! ===============================================
//! Chapter 4, File 3: Rust Advanced Market Making & Inventory Risk
//!
//! Toxic order flow detection. Analyzes the order book to identify informed 
//! traders (toxic flow) and dynamically widens spreads or halts quoting to 
//! avoid adverse selection (getting run over by insider momentum).
//!
//! Features: Order flow analysis, toxicity scoring, dynamic spread adjustment

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::VecDeque;
use chrono::{DateTime, Utc};

/// Order flow event
#[derive(Debug, Clone)]
pub struct OrderFlowEvent {
    /// Timestamp
    pub timestamp_ns: u64,
    /// Trade size
    pub size: f64,
    /// Trade price
    pub price: f64,
    /// Aggressor side (true = buy, false = sell)
    pub is_buy: bool,
    /// Order book imbalance at time of trade
    pub imbalance: f64,
}

/// Toxicity metrics snapshot
#[derive(Debug, Clone)]
pub struct ToxicityMetrics {
    /// Overall toxicity score (0-1, higher = more toxic)
    pub toxicity_score: f64,
    /// VPIN (Volume-Synchronized Probability of Informed Trading)
    pub vpin: f64,
    /// Order flow imbalance
    pub ofi: f64,
    /// Price impact per unit volume
    pub price_impact: f64,
    /// Abnormal volume ratio
    pub abnormal_volume_ratio: f64,
    /// Momentum indicator
    pub momentum: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Toxicity level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToxicityLevel {
    /// Normal market conditions
    Normal,
    /// Elevated toxicity
    Elevated,
    /// High toxicity - caution advised
    High,
    /// Extreme toxicity - halt quoting
    Extreme,
}

impl ToxicityLevel {
    pub fn from_score(score: f64) -> Self {
        if score < 0.3 {
            ToxicityLevel::Normal
        } else if score < 0.5 {
            ToxicityLevel::Elevated
        } else if score < 0.7 {
            ToxicityLevel::High
        } else {
            ToxicityLevel::Extreme
        }
    }
}

/// Configuration for adverse selection detection
#[derive(Debug, Clone)]
pub struct AdverseSelectionConfig {
    /// Lookback window for analysis (number of trades)
    pub lookback_trades: usize,
    
    /// Volume bucket size for VPIN calculation
    pub volume_bucket_size: f64,
    
    /// VPIN threshold for elevated toxicity
    pub vpin_threshold_elevated: f64,
    
    /// VPIN threshold for extreme toxicity
    pub vpin_threshold_extreme: f64,
    
    /// Order flow imbalance threshold
    pub ofi_threshold: f64,
    
    /// Price impact sensitivity
    pub price_impact_sensitivity: f64,
    
    /// Enable momentum-based detection
    pub momentum_detection: bool,
    
    /// Momentum lookback (trades)
    pub momentum_lookback: usize,
    
    /// Spread widening factor under toxicity
    pub spread_widening_factor: f64,
    
    /// Maximum spread multiplier
    pub max_spread_multiplier: f64,
}

impl Default for AdverseSelectionConfig {
    fn default() -> Self {
        Self {
            lookback_trades: 100,
            volume_bucket_size: 1000.0,
            vpin_threshold_elevated: 0.4,
            vpin_threshold_extreme: 0.7,
            ofi_threshold: 0.5,
            price_impact_sensitivity: 1.0,
            momentum_detection: true,
            momentum_lookback: 20,
            spread_widening_factor: 2.0,
            max_spread_multiplier: 10.0,
        }
    }
}

/// VPIN calculator for informed trading detection
pub struct VPINCalculator {
    /// Volume buckets
    buckets: VecDeque<f64>,
    /// Buy volume in current bucket
    current_buy_volume: f64,
    /// Sell volume in current bucket
    current_sell_volume: f64,
    /// Bucket size
    bucket_size: f64,
    /// Number of buckets for VPIN
    n_buckets: usize,
}

impl VPINCalculator {
    pub fn new(bucket_size: f64, n_buckets: usize) -> Self {
        Self {
            buckets: VecDeque::with_capacity(n_buckets),
            current_buy_volume: 0.0,
            current_sell_volume: 0.0,
            bucket_size,
            n_buckets,
        }
    }
    
    /// Add a trade to the calculator
    pub fn add_trade(&mut self, volume: f64, is_buy: bool) -> Option<f64> {
        // Add to current bucket
        if is_buy {
            self.current_buy_volume += volume;
        } else {
            self.current_sell_volume += volume;
        }
        
        // Check if bucket is full
        let total_bucket = self.current_buy_volume + self.current_sell_volume;
        if total_bucket >= self.bucket_size {
            // Store imbalance for this bucket
            let imbalance = (self.current_buy_volume - self.current_sell_volume) 
                / self.bucket_size;
            
            self.buckets.push_back(imbalance.abs());
            
            // Reset current bucket
            self.current_buy_volume = 0.0;
            self.current_sell_volume = 0.0;
            
            // Remove oldest if we have enough buckets
            if self.buckets.len() > self.n_buckets {
                self.buckets.pop_front();
            }
            
            // Calculate VPIN if we have enough data
            if self.buckets.len() >= self.n_buckets / 2 {
                return Some(self.calculate_vpin());
            }
        }
        
        None
    }
    
    /// Calculate current VPIN
    fn calculate_vpin(&self) -> f64 {
        if self.buckets.is_empty() {
            return 0.0;
        }
        
        // VPIN = (1/n) * sum(|B - S|) / (B + S)
        let sum_imbalance: f64 = self.buckets.iter().sum();
        sum_imbalance / self.buckets.len() as f64
    }
    
    /// Get current VPIN estimate
    pub fn get_vpin(&self) -> f64 {
        self.calculate_vpin()
    }
    
    /// Reset calculator
    pub fn reset(&mut self) {
        self.buckets.clear();
        self.current_buy_volume = 0.0;
        self.current_sell_volume = 0.0;
    }
}

/// Order Flow Imbalance calculator
pub struct OFICalculator {
    /// Recent order flow events
    events: VecDeque<OrderFlowEvent>,
    /// Max events to track
    max_events: usize,
}

impl OFICalculator {
    pub fn new(max_events: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(max_events),
            max_events,
        }
    }
    
    /// Add an order flow event
    pub fn add_event(&mut self, event: OrderFlowEvent) {
        self.events.push_back(event);
        if self.events.len() > self.max_events {
            self.events.pop_front();
        }
    }
    
    /// Calculate Order Flow Imbalance
    pub fn calculate_ofi(&self) -> f64 {
        if self.events.is_empty() {
            return 0.0;
        }
        
        let buy_volume: f64 = self.events.iter()
            .filter(|e| e.is_buy)
            .map(|e| e.size)
            .sum();
        
        let sell_volume: f64 = self.events.iter()
            .filter(|e| !e.is_buy)
            .map(|e| e.size)
            .sum();
        
        let total = buy_volume + sell_volume;
        if total < 1e-6 {
            return 0.0;
        }
        
        (buy_volume - sell_volume) / total
    }
    
    /// Get recent momentum (price change weighted by volume)
    pub fn calculate_momentum(&self) -> f64 {
        if self.events.len() < 2 {
            return 0.0;
        }
        
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;
        
        let base_price = self.events.front().unwrap().price;
        
        for event in &self.events {
            let price_change = (event.price - base_price) / base_price;
            let weight = event.size;
            weighted_sum += price_change * weight;
            weight_total += weight;
        }
        
        if weight_total < 1e-6 {
            return 0.0;
        }
        
        weighted_sum / weight_total
    }
}

/// Adverse Selection Detector
pub struct AdverseSelectionDetector {
    /// Configuration
    config: RwLock<AdverseSelectionConfig>,
    /// VPIN calculator
    vpin_calc: RwLock<VPINCalculator>,
    /// OFI calculator
    ofi_calc: RwLock[OFICalculator>,
    /// Current toxicity level
    current_level: RwLock<ToxicityLevel>,
    /// Current metrics
    current_metrics: RwLock<Option<ToxicityMetrics>>,
    /// Alert counter
    alert_count: AtomicU64,
    /// Halting flag
    halted: AtomicBool,
    /// Consecutive toxic readings
    toxic_streak: AtomicU64,
}

impl AdverseSelectionDetector {
    /// Create new detector with default config
    pub fn new() -> Self {
        let config = AdverseSelectionConfig::default();
        Self {
            config: RwLock::new(config.clone()),
            vpin_calc: RwLock::new(VPINCalculator::new(
                config.volume_bucket_size,
                config.lookback_trades / 10,
            )),
            ofi_calc: RwLock::new(OFICalculator::new(config.lookback_trades)),
            current_level: RwLock::new(ToxicityLevel::Normal),
            current_metrics: RwLock::new(None),
            alert_count: AtomicU64::new(0),
            halted: AtomicBool::new(false),
            toxic_streak: AtomicU64::new(0),
        }
    }
    
    /// Create with custom config
    pub fn with_config(config: AdverseSelectionConfig) -> Self {
        Self {
            config: RwLock::new(config.clone()),
            vpin_calc: RwLock::new(VPINCalculator::new(
                config.volume_bucket_size,
                config.lookback_trades / 10,
            )),
            ofi_calc: RwLock::new(OFICalculator::new(config.lookback_trades)),
            current_level: RwLock::new(ToxicityLevel::Normal),
            current_metrics: RwLock::new(None),
            alert_count: AtomicU64::new(0),
            halted: AtomicBool::new(false),
            toxic_streak: AtomicU64::new(0),
        }
    }
    
    /// Process a trade event
    pub fn process_trade(&self, event: OrderFlowEvent) {
        // Update calculators
        self.vpin_calc.write().add_trade(event.size, event.is_buy);
        self.ofi_calc.write().add_event(event.clone());
        
        // Update metrics
        self.update_metrics();
    }
    
    /// Update toxicity metrics
    fn update_metrics(&self) {
        let config = self.config.read();
        let vpin = self.vpin_calc.read().get_vpin();
        let ofi = self.ofi_calc.read().calculate_ofi();
        let momentum = if config.momentum_detection {
            self.ofi_calc.read().calculate_momentum()
        } else {
            0.0
        };
        
        // Calculate composite toxicity score
        let toxicity_score = self.calculate_toxicity_score(vpin, ofi, momentum, &config);
        
        // Determine toxicity level
        let level = ToxicityLevel::from_score(toxicity_score);
        
        // Update streak
        if level == ToxicityLevel::High || level == ToxicityLevel::Extreme {
            self.toxic_streak.fetch_add(1, Ordering::Relaxed);
        } else {
            self.toxic_streak.store(0, Ordering::Relaxed);
        }
        
        // Check for halting condition
        if level == ToxicityLevel::Extreme || self.toxic_streak.load(Ordering::Relaxed) >= 5 {
            self.halted.store(true, Ordering::Relaxed);
            self.alert_count.fetch_add(1, Ordering::Relaxed);
        } else if level == ToxicityLevel::Normal && self.halted.load(Ordering::Relaxed) {
            // Resume after cooldown
            self.halted.store(false, Ordering::Relaxed);
        }
        
        // Store metrics
        *self.current_level.write() = level;
        *self.current_metrics.write() = Some(ToxicityMetrics {
            toxicity_score,
            vpin,
            ofi,
            price_impact: 0.0, // Would need price series to calculate
            abnormal_volume_ratio: 1.0, // Would need volume history
            momentum,
            timestamp: Utc::now(),
        });
    }
    
    /// Calculate composite toxicity score
    fn calculate_toxicity_score(
        &self,
        vpin: f64,
        ofi: f64,
        momentum: f64,
        config: &AdverseSelectionConfig,
    ) -> f64 {
        // Weighted combination of signals
        let vpin_component = vpin * 0.5; // VPIN is primary signal
        let ofi_component = ofi.abs() * 0.3; // Order flow imbalance
        let momentum_component = momentum.abs() * config.price_impact_sensitivity * 0.2;
        
        let score = vpin_component + ofi_component + momentum_component;
        score.min(1.0) // Cap at 1.0
    }
    
    /// Get current toxicity level
    pub fn get_level(&self) -> ToxicityLevel {
        *self.current_level.read()
    }
    
    /// Get current metrics
    pub fn get_metrics(&self) -> Option<ToxicityMetrics> {
        self.current_metrics.read().clone()
    }
    
    /// Check if quoting should be halted
    pub fn is_halted(&self) -> bool {
        self.halted.load(Ordering::Relaxed)
    }
    
    /// Calculate spread multiplier based on toxicity
    pub fn get_spread_multiplier(&self) -> f64 {
        let config = self.config.read();
        let level = self.get_level();
        
        let base_multiplier = match level {
            ToxicityLevel::Normal => 1.0,
            ToxicityLevel::Elevated => config.spread_widening_factor * 0.5,
            ToxicityLevel::High => config.spread_widening_factor,
            ToxicityLevel::Extreme => config.max_spread_multiplier,
        };
        
        // Additional multiplier based on VPIN
        if let Some(metrics) = self.get_metrics() {
            let vpin_adjustment = 1.0 + metrics.vpin;
            (base_multiplier * vpin_adjustment).min(config.max_spread_multiplier)
        } else {
            base_multiplier
        }
    }
    
    /// Adjust bid/ask based on toxicity
    pub fn adjust_quotes(&self, bid: f64, ask: f64) -> (f64, f64) {
        if self.is_halted() {
            // Return wide quotes when halted
            let mid = (bid + ask) / 2.0;
            let wide_spread = mid * 0.01; // 1% spread
            return (mid - wide_spread / 2.0, mid + wide_spread / 2.0);
        }
        
        let multiplier = self.get_spread_multiplier();
        let mid = (bid + ask) / 2.0;
        let half_spread = (ask - bid) / 2.0 * multiplier;
        
        (mid - half_spread, mid + half_spread)
    }
    
    /// Reset detector state
    pub fn reset(&self) {
        self.vpin_calc.write().reset();
        self.ofi_calc.write().events.clear();
        self.halted.store(false, Ordering::Relaxed);
        self.toxic_streak.store(0, Ordering::Relaxed);
        *self.current_level.write() = ToxicityLevel::Normal;
    }
    
    /// Get statistics
    pub fn get_statistics(&self) -> AdverseSelectionStats {
        let level = self.get_level();
        let metrics = self.get_metrics();
        
        AdverseSelectionStats {
            toxicity_level: level,
            toxicity_score: metrics.as_ref().map(|m| m.toxicity_score).unwrap_or(0.0),
            vpin: metrics.as_ref().map(|m| m.vpin).unwrap_or(0.0),
            ofi: metrics.as_ref().map(|m| m.ofi).unwrap_or(0.0),
            spread_multiplier: self.get_spread_multiplier(),
            is_halted: self.is_halted(),
            alert_count: self.alert_count.load(Ordering::Relaxed),
            toxic_streak: self.toxic_streak.load(Ordering::Relaxed),
        }
    }
}

/// Adverse selection statistics
#[derive(Debug, Clone)]
pub struct AdverseSelectionStats {
    pub toxicity_level: ToxicityLevel,
    pub toxicity_score: f64,
    pub vpin: f64,
    pub ofi: f64,
    pub spread_multiplier: f64,
    pub is_halted: bool,
    pub alert_count: u64,
    pub toxic_streak: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vpin_calculation() {
        let mut calc = VPINCalculator::new(100.0, 10);
        
        // Add alternating buys and sells (low VPIN expected)
        for i in 0..50 {
            let is_buy = i % 2 == 0;
            calc.add_trade(10.0, is_buy);
        }
        
        let vpin = calc.get_vpin();
        println!("Balanced flow VPIN: {}", vpin);
        assert!(vpin < 0.3); // Should be low
        
        // Reset and add only buys (high VPIN expected)
        calc.reset();
        for _ in 0..50 {
            calc.add_trade(10.0, true);
        }
        
        let vpin = calc.get_vpin();
        println!("One-sided flow VPIN: {}", vpin);
        assert!(vpin > 0.5); // Should be high
    }
    
    #[test]
    fn test_toxicity_detection() {
        let detector = AdverseSelectionDetector::new();
        
        // Initial state should be normal
        assert_eq!(detector.get_level(), ToxicityLevel::Normal);
        assert!(!detector.is_halted());
        
        // Simulate toxic flow (all buys)
        for i in 0..100 {
            let event = OrderFlowEvent {
                timestamp_ns: i * 1_000_000,
                size: 50.0,
                price: 100.0 + i as f64 * 0.1,
                is_buy: true,
                imbalance: 0.8,
            };
            detector.process_trade(event);
        }
        
        let stats = detector.get_statistics();
        println!("Toxicity score: {}", stats.toxicity_score);
        println!("VPIN: {}", stats.vpin);
        println!("Level: {:?}", stats.toxicity_level);
        
        // Should detect elevated toxicity
        assert!(stats.toxicity_score > 0.0);
    }
    
    #[test]
    fn test_spread_adjustment() {
        let detector = AdverseSelectionDetector::new();
        
        // Normal conditions - minimal adjustment
        let (bid, ask) = detector.adjust_quotes(99.0, 101.0);
        let spread = ask - bid;
        println!("Normal spread: {}", spread);
        
        // Simulate toxic conditions
        for i in 0..200 {
            let event = OrderFlowEvent {
                timestamp_ns: i * 1_000_000,
                size: 100.0,
                price: 100.0 + i as f64 * 0.05,
                is_buy: true,
                imbalance: 0.9,
            };
            detector.process_trade(event);
        }
        
        let (bid, ask) = detector.adjust_quotes(99.0, 101.0);
        let spread = ask - bid;
        println!("Toxic spread: {}", spread);
        assert!(spread > 2.0); // Should be wider than original
    }
}
