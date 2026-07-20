//! `trade_classification.rs` - High-Speed Trade Classification (Lee-Ready & Tick Rule)
//! 
//! **STAGE 10 | CHAPTER 2 | FILE 2**
//! 
//! This module implements ultra-fast trade classification algorithms to determine
//! whether each trade was buyer-initiated or seller-initiated. This is critical for:
//! - VPIN calculation
//! - Order flow analysis
//! - Market microstructure research
//! - Toxic flow detection
//! 
//! **Algorithms Implemented:**
//! 1. Tick Rule - Simple, fast, works on trade data alone
//! 2. Lee-Ready Algorithm - More accurate, requires order book context
//! 3. Bulk Volume Classification (BVC) - For aggregated data
//! 
//! **Performance:**
//! - Sub-microsecond classification per trade
//! - Zero heap allocation in hot path
//! - SIMD-ready batch processing

use crate::microstructure::vpin::TradeSide;

/// Previous tick information for Tick Rule
#[derive(Debug, Clone, Copy)]
pub struct TickState {
    pub last_price: f64,
    pub last_side: TradeSide,
}

impl Default for TickState {
    fn default() -> Self {
        Self {
            last_price: 0.0,
            last_side: TradeSide::Unknown,
        }
    }
}

/// Tick Rule Trade Classifier
/// 
/// The Tick Rule classifies trades based on price movement:
/// - If price > previous price → Buyer initiated
/// - If price < previous price → Seller initiated  
/// - If price == previous price → Same as previous trade
/// 
/// This is the fastest method but can be inaccurate during rapid price movements.
pub struct TickRuleClassifier {
    state: TickState,
    stats: ClassificationStats,
}

/// Classification statistics for quality monitoring
#[derive(Debug, Clone, Default)]
pub struct ClassificationStats {
    pub total_classified: usize,
    pub buyer_initiated: usize,
    pub seller_initiated: usize,
    pub unknown: usize,
    pub uptick_count: usize,
    pub downtick_count: usize,
    pub zero_tick_count: usize,
}

impl TickRuleClassifier {
    pub fn new() -> Self {
        Self {
            state: TickState::default(),
            stats: ClassificationStats::default(),
        }
    }

    /// Classify a single trade using the Tick Rule
    /// 
    /// # Arguments
    /// * `price` - Trade execution price
    /// * `volume` - Trade volume (used for statistics only)
    /// 
    /// Returns: TradeSide classification
    #[inline(always)]
    pub fn classify(&mut self, price: f64, volume: f64) -> TradeSide {
        if self.state.last_price == 0.0 {
            // First trade - cannot classify
            self.state.last_price = price;
            self.state.last_side = TradeSide::Unknown;
            self.stats.unknown += 1;
            return TradeSide::Unknown;
        }

        let side = if price > self.state.last_price {
            self.stats.uptick_count += 1;
            TradeSide::BuyerInitiated
        } else if price < self.state.last_price {
            self.stats.downtick_count += 1;
            TradeSide::SellerInitiated
        } else {
            // Zero tick - inherit from previous
            self.stats.zero_tick_count += 1;
            self.state.last_side
        };

        self.state.last_price = price;
        self.state.last_side = side;
        
        self.stats.total_classified += 1;
        match side {
            TradeSide::BuyerInitiated => self.stats.buyer_initiated += 1,
            TradeSide::SellerInitiated => self.stats.seller_initiated += 1,
            TradeSide::Unknown => self.stats.unknown += 1,
        }

        side
    }

    /// Batch classification for vectorized processing
    /// 
    /// # Arguments
    /// * `prices` - Slice of trade prices
    /// * `output` - Pre-allocated output slice for classifications
    pub fn classify_batch(&mut self, prices: &[f64], output: &mut [TradeSide]) {
        assert!(output.len() >= prices.len());
        
        for (i, &price) in prices.iter().enumerate() {
            output[i] = self.classify(price, 0.0);
        }
    }

    /// Get current state (for checkpointing or debugging)
    pub fn state(&self) -> &TickState {
        &self.state
    }

    /// Set state (for restoring from checkpoint)
    pub fn set_state(&mut self, state: TickState) {
        self.state = state;
    }

    /// Get statistics reference
    pub fn stats(&self) -> &ClassificationStats {
        &self.stats
    }

    /// Reset classifier state
    pub fn reset(&mut self) {
        self.state = TickState::default();
        self.stats = ClassificationStats::default();
    }

    /// Get buyer/seller ratio (quality metric)
    pub fn buy_sell_ratio(&self) -> f64 {
        if self.stats.seller_initiated == 0 {
            if self.stats.buyer_initiated == 0 {
                return 1.0;
            }
            return f64::INFINITY;
        }
        self.stats.buyer_initiated as f64 / self.stats.seller_initiated as f64
    }
}

impl Default for TickRuleClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Lee-Ready Algorithm Trade Classifier
/// 
/// The Lee-Ready algorithm (1991) is more accurate than the Tick Rule as it uses
/// order book context to classify trades:
/// 1. Compare trade price to prevailing bid/ask at time of trade
/// 2. If price > midpoint → Buyer initiated
/// 3. If price < midpoint → Seller initiated
/// 4. If price == midpoint → Use tick rule as tiebreaker
/// 
/// Requires access to L1 order book data at trade time.
pub struct LeeReadyClassifier {
    tick_classifier: TickRuleClassifier,
    current_bid: f64,
    current_ask: f64,
    current_midpoint: f64,
    stats: ClassificationStats,
}

impl LeeReadyClassifier {
    pub fn new() -> Self {
        Self {
            tick_classifier: TickRuleClassifier::new(),
            current_bid: 0.0,
            current_ask: 0.0,
            current_midpoint: 0.0,
            stats: ClassificationStats::default(),
        }
    }

    /// Update the current order book state
    /// 
    /// # Arguments
    /// * `bid` - Best bid price
    /// * `ask` - Best ask price
    #[inline(always)]
    pub fn update_quote(&mut self, bid: f64, ask: f64) {
        if bid > 0.0 && ask > 0.0 && ask > bid {
            self.current_bid = bid;
            self.current_ask = ask;
            self.current_midpoint = (bid + ask) / 2.0;
        }
    }

    /// Classify a trade using Lee-Ready algorithm
    /// 
    /// # Arguments
    /// * `price` - Trade execution price
    /// * `timestamp_ns` - Trade timestamp (must be >= quote timestamp)
    /// 
    /// Returns: TradeSide classification
    #[inline(always)]
    pub fn classify(&mut self, price: f64, timestamp_ns: u64) -> TradeSide {
        if self.current_midpoint == 0.0 {
            // No quote data - fall back to tick rule
            let side = self.tick_classifier.classify(price, 0.0);
            self.update_stats(side);
            return side;
        }

        let epsilon = 0.0001 * self.current_midpoint; // Price tolerance

        let side = if price > self.current_midpoint + epsilon {
            TradeSide::BuyerInitiated
        } else if price < self.current_midpoint - epsilon {
            TradeSide::SellerInitiated
        } else {
            // Trade at midpoint - use tick rule as tiebreaker
            self.tick_classifier.classify(price, 0.0)
        };

        self.update_stats(side);
        side
    }

    /// Classify with explicit quote data (for historical replay)
    /// 
    /// # Arguments
    /// * `price` - Trade price
    /// * `bid` - Bid at time of trade
    /// * `ask` - Ask at time of trade
    pub fn classify_with_quote(&mut self, price: f64, bid: f64, ask: f64) -> TradeSide {
        if bid <= 0.0 || ask <= 0.0 || ask <= bid {
            return self.tick_classifier.classify(price, 0.0);
        }

        let midpoint = (bid + ask) / 2.0;
        let epsilon = 0.0001 * midpoint;

        let side = if price > midpoint + epsilon {
            TradeSide::BuyerInitiated
        } else if price < midpoint - epsilon {
            TradeSide::SellerInitiated
        } else {
            self.tick_classifier.classify(price, 0.0)
        };

        self.update_stats(side);
        side
    }

    fn update_stats(&mut self, side: TradeSide) {
        self.stats.total_classified += 1;
        match side {
            TradeSide::BuyerInitiated => self.stats.buyer_initiated += 1,
            TradeSide::SellerInitiated => self.stats.seller_initiated += 1,
            TradeSide::Unknown => self.stats.unknown += 1,
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &ClassificationStats {
        &self.stats
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.tick_classifier.reset();
        self.current_bid = 0.0;
        self.current_ask = 0.0;
        self.current_midpoint = 0.0;
        self.stats = ClassificationStats::default();
    }
}

impl Default for LeeReadyClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Bulk Volume Classification (BVC)
/// 
/// For situations where individual trade data is not available,
/// BVC classifies aggregated volume based on price position within the bar.
/// 
/// P_buy = Φ((close - low) / (high - low) - 0.5) * total_volume
/// P_sell = total_volume - P_buy
/// 
/// Where Φ is the standard normal CDF.
pub struct BulkVolumeClassifier;

impl BulkVolumeClassifier {
    /// Classify a volume bar into buyer/seller components
    /// 
    /// # Arguments
    /// * `open` - Bar open price
    /// * `high` - Bar high price
    /// * `low` - Bar low price  
    /// * `close` - Bar close price
    /// * `total_volume` - Total volume in the bar
    /// 
    /// Returns: (buyer_volume, seller_volume)
    pub fn classify_bar(
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        total_volume: f64,
    ) -> (f64, f64) {
        if high == low || total_volume <= 0.0 {
            return (total_volume / 2.0, total_volume / 2.0);
        }

        // Calculate price position within range
        let price_position = (close - low) / (high - low);
        
        // Transform to Z-score-like value centered at 0.5
        let z = (price_position - 0.5) * 3.464; // Scale factor for reasonable spread
        
        // Approximate normal CDF using polynomial approximation
        let p_buy = norm_cdf_approx(z);
        
        let buyer_volume = p_buy * total_volume;
        let seller_volume = (1.0 - p_buy) * total_volume;
        
        (buyer_volume, seller_volume)
    }

    /// Batch process multiple bars
    pub fn classify_bars(bars: &[(f64, f64, f64, f64, f64)]) -> Vec<(f64, f64)> {
        bars.iter()
            .map(|&(o, h, l, c, v)| Self::classify_bar(o, h, l, c, v))
            .collect()
    }
}

/// Fast approximation of standard normal CDF
#[inline(always)]
fn norm_cdf_approx(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    const A1: f64 = 0.319381530;
    const A2: f64 = -0.356563782;
    const A3: f64 = 1.781477937;
    const A4: f64 = -1.821255978;
    const A5: f64 = 1.330274429;
    const P: f64 = 0.2316419;

    if x >= 0.0 {
        let t = 1.0 / (1.0 + P * x);
        let poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))));
        1.0 - poly * (-0.5 * x * x).exp() * 0.3989422804014327
    } else {
        1.0 - norm_cdf_approx(-x)
    }
}

/// Hybrid classifier that combines multiple methods for robustness
pub enum TradeClassifier {
    Tick(TickRuleClassifier),
    LeeReady(LeeReadyClassifier),
}

impl TradeClassifier {
    pub fn tick() -> Self {
        TradeClassifier::Tick(TickRuleClassifier::new())
    }

    pub fn lee_ready() -> Self {
        TradeClassifier::LeeReady(LeeReadyClassifier::new())
    }

    pub fn classify(&mut self, price: f64) -> TradeSide {
        match self {
            TradeClassifier::Tick(tc) => tc.classify(price, 0.0),
            TradeClassifier::LeeReady(lrc) => lrc.classify(price, 0),
        }
    }

    pub fn update_quote(&mut self, bid: f64, ask: f64) {
        if let TradeClassifier::LeeReady(lrc) = self {
            lrc.update_quote(bid, ask);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tick_rule_uptrend() {
        let mut classifier = TickRuleClassifier::new();
        
        // Simulate uptrend
        assert_eq!(classifier.classify(100.0, 1.0), TradeSide::Unknown);
        assert_eq!(classifier.classify(101.0, 1.0), TradeSide::BuyerInitiated);
        assert_eq!(classifier.classify(102.0, 1.0), TradeSide::BuyerInitiated);
        assert_eq!(classifier.classify(101.5, 1.0), TradeSide::SellerInitiated);
    }

    #[test]
    fn test_lee_ready_basic() {
        let mut classifier = LeeReadyClassifier::new();
        
        // Set quotes
        classifier.update_quote(99.0, 101.0);
        
        // Trade above midpoint (100) should be buyer
        assert_eq!(classifier.classify(100.5, 0), TradeSide::BuyerInitiated);
        
        // Trade below midpoint should be seller
        assert_eq!(classifier.classify(99.5, 0), TradeSide::SellerInitiated);
    }

    #[test]
    fn test_bvc_classification() {
        // Strong bullish bar
        let (buy_vol, sell_vol) = BulkVolumeClassifier::classify_bar(
            100.0, 110.0, 95.0, 109.0, 1000.0,
        );
        assert!(buy_vol > sell_vol);
        assert!(buy_vol > 700.0); // Should be heavily buyer-dominated

        // Strong bearish bar
        let (buy_vol, sell_vol) = BulkVolumeClassifier::classify_bar(
            100.0, 105.0, 90.0, 91.0, 1000.0,
        );
        assert!(sell_vol > buy_vol);
        assert!(sell_vol > 700.0);
    }

    #[test]
    fn test_classifier_stats() {
        let mut classifier = TickRuleClassifier::new();
        
        classifier.classify(100.0, 1.0);
        classifier.classify(101.0, 1.0);
        classifier.classify(102.0, 1.0);
        classifier.classify(101.0, 1.0);
        
        let stats = classifier.stats();
        assert_eq!(stats.total_classified, 4);
        assert_eq!(stats.buyer_initiated, 2);
        assert_eq!(stats.seller_initiated, 1);
    }
}
