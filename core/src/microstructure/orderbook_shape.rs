//! Order Book Shape Analysis Module
//! 
//! This module analyzes the geometric shape of the L2 order book to predict
//! short-term momentum and liquidity vacuums in real-time.
//! 
//! Key metrics:
//! - Convexity/Concavity of bid/ask depth curves
//! - Depth imbalances across price levels
//! - Bid-ask micro-jumps and spread dynamics
//! - Liquidity density gradients
//! 
//! Target latency: < 5 microseconds per analysis cycle

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

/// Represents a single price level in the order book
#[derive(Clone, Debug)]
pub struct PriceLevel {
    pub price: i64,      // Price in smallest tick unit (e.g., satoshis)
    pub quantity: u64,   // Quantity at this level
    pub order_count: u32, // Number of orders at this level
    pub timestamp_ns: u64, // Nanosecond timestamp
}

/// Circular buffer for maintaining rolling order book snapshots
/// Uses pre-allocated memory to avoid allocations during hot path
pub struct OrderBookSnapshot {
    bids: Vec<PriceLevel>,
    asks: Vec<PriceLevel>,
    max_levels: usize,
    last_update_ns: AtomicU64,
}

impl OrderBookSnapshot {
    pub fn new(max_levels: usize) -> Self {
        Self {
            bids: Vec::with_capacity(max_levels),
            asks: Vec::with_capacity(max_levels),
            max_levels,
            last_update_ns: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn update(&mut self, bids: Vec<PriceLevel>, asks: Vec<PriceLevel>) {
        self.bids = bids.into_iter().take(self.max_levels).collect();
        self.asks = asks.into_iter().take(self.max_levels).collect();
        self.last_update_ns.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            Ordering::Relaxed,
        );
    }

    #[inline]
    pub fn best_bid(&self) -> Option<&PriceLevel> {
        self.bids.first()
    }

    #[inline]
    pub fn best_ask(&self) -> Option<&PriceLevel> {
        self.asks.first()
    }

    #[inline]
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid.price as f64 + ask.price as f64) / 2.0),
            _ => None,
        }
    }

    #[inline]
    pub fn spread_bps(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) if bid.price > 0 => {
                Some(((ask.price - bid.price) as f64 / bid.price as f64) * 10000.0)
            }
            _ => None,
        }
    }
}

/// Geometric shape descriptors for the order book
#[derive(Debug, Clone)]
pub struct OrderBookShape {
    /// Convexity of the bid side (positive = convex, negative = concave)
    pub bid_convexity: f64,
    /// Convexity of the ask side
    pub ask_convexity: f64,
    /// Ratio of bid depth to ask depth across all levels
    pub depth_imbalance_ratio: f64,
    /// Weighted depth imbalance (closer levels weighted more heavily)
    pub weighted_depth_imbalance: f64,
    /// Rate of change of spread over recent updates
    pub spread_momentum: f64,
    /// Liquidity density gradient (how quickly depth falls off from best price)
    pub bid_liquidity_gradient: f64,
    pub ask_liquidity_gradient: f64,
    /// Micro-jump indicator: sudden changes in best bid/ask
    pub bid_micro_jump: f64,
    pub ask_micro_jump: f64,
    /// Overall shape score (-1.0 to 1.0): positive suggests upward pressure
    pub shape_score: f64,
    /// Predicted short-term momentum (price change over next N milliseconds)
    pub predicted_momentum_bps: f64,
    /// Confidence in the prediction (0.0 to 1.0)
    pub prediction_confidence: f64,
    /// Timestamp of this analysis
    pub timestamp_ns: u64,
}

impl Default for OrderBookShape {
    fn default() -> Self {
        Self {
            bid_convexity: 0.0,
            ask_convexity: 0.0,
            depth_imbalance_ratio: 1.0,
            weighted_depth_imbalance: 0.0,
            bid_liquidity_gradient: 0.0,
            ask_liquidity_gradient: 0.0,
            bid_micro_jump: 0.0,
            ask_micro_jump: 0.0,
            shape_score: 0.0,
            predicted_momentum_bps: 0.0,
            prediction_confidence: 0.0,
            timestamp_ns: 0,
            spread_momentum: 0.0,
        }
    }
}

/// Main analyzer for order book shape dynamics
pub struct OrderBookShapeAnalyzer {
    /// Rolling history of shapes for momentum calculation
    shape_history: VecDeque<OrderBookShape>,
    max_history: usize,
    /// Previous snapshot for micro-jump detection
    prev_best_bid: Option<i64>,
    prev_best_ask: Option<i64>,
    prev_spread_bps: Option<f64>,
    /// Calibration parameters (can be learned online)
    convexity_weight: f64,
    imbalance_weight: f64,
    gradient_weight: f64,
    micro_jump_weight: f64,
}

impl OrderBookShapeAnalyzer {
    pub fn new(max_history: usize) -> Self {
        Self {
            shape_history: VecDeque::with_capacity(max_history),
            max_history,
            prev_best_bid: None,
            prev_best_ask: None,
            prev_spread_bps: None,
            // Initial weights - should be calibrated via backtesting
            convexity_weight: 0.25,
            imbalance_weight: 0.35,
            gradient_weight: 0.20,
            micro_jump_weight: 0.20,
        }
    }

    /// Analyze the current order book shape and generate predictions
    /// 
    /// This is the main entry point called on every order book update.
    /// Must complete within microseconds to be useful for HFT.
    #[inline]
    pub fn analyze(&mut self, snapshot: &OrderBookSnapshot) -> OrderBookShape {
        let mut shape = OrderBookShape::default();
        shape.timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Calculate convexity of bid and ask curves
        shape.bid_convexity = self.calculate_convexity(&snapshot.bids);
        shape.ask_convexity = self.calculate_convexity(&snapshot.asks);

        // Calculate depth imbalances
        shape.depth_imbalance_ratio = self.calculate_depth_imbalance(snapshot);
        shape.weighted_depth_imbalance = self.calculate_weighted_depth_imbalance(snapshot);

        // Calculate liquidity gradients
        (shape.bid_liquidity_gradient, shape.ask_liquidity_gradient) = 
            self.calculate_liquidity_gradients(snapshot);

        // Detect micro-jumps
        (shape.bid_micro_jump, shape.ask_micro_jump) = 
            self.detect_micro_jumps(snapshot);

        // Calculate spread momentum
        shape.spread_momentum = snapshot.spread_bps()
            .zip(self.prev_spread_bps)
            .map(|(curr, prev)| curr - prev)
            .unwrap_or(0.0);

        // Compute overall shape score
        shape.shape_score = self.compute_shape_score(&shape);

        // Generate momentum prediction
        (shape.predicted_momentum_bps, shape.prediction_confidence) = 
            self.predict_momentum(&shape);

        // Update history
        self.shape_history.push_back(shape.clone());
        if self.shape_history.len() > self.max_history {
            self.shape_history.pop_front();
        }

        // Update previous state
        self.prev_best_bid = snapshot.best_bid().map(|l| l.price);
        self.prev_best_ask = snapshot.best_ask().map(|l| l.price);
        self.prev_spread_bps = snapshot.spread_bps();

        shape
    }

    /// Calculate convexity of the depth curve using second derivative approximation
    /// 
    /// A convex bid curve (positive value) indicates increasing depth at lower prices,
    /// suggesting strong support. A concave curve suggests weak support.
    /// 
    /// Formula: sum((d[i+1] - d[i]) - (d[i] - d[i-1])) / num_points
    #[inline]
    fn calculate_convexity(&self, levels: &[PriceLevel]) -> f64 {
        if levels.len() < 3 {
            return 0.0;
        }

        let mut second_deriv_sum = 0.0;
        let mut total_quantity = 0.0;

        for i in 1..levels.len() - 1 {
            let q_prev = levels[i - 1].quantity as f64;
            let q_curr = levels[i].quantity as f64;
            let q_next = levels[i + 1].quantity as f64;

            // Second derivative (discrete approximation)
            let second_deriv = (q_next - q_curr) - (q_curr - q_prev);
            second_deriv_sum += second_deriv;
            total_quantity += q_curr;
        }

        // Normalize by total quantity to get scale-invariant measure
        if total_quantity > 0.0 {
            second_deriv_sum / total_quantity * 100.0
        } else {
            0.0
        }
    }

    /// Calculate simple depth imbalance ratio
    #[inline]
    fn calculate_depth_imbalance(&self, snapshot: &OrderBookSnapshot) -> f64 {
        let bid_depth: f64 = snapshot.bids.iter()
            .map(|l| l.quantity as f64)
            .sum();
        
        let ask_depth: f64 = snapshot.asks.iter()
            .map(|l| l.quantity as f64)
            .sum();

        if ask_depth == 0.0 {
            return if bid_depth > 0.0 { f64::INFINITY } else { 1.0 };
        }

        bid_depth / ask_depth
    }

    /// Calculate weighted depth imbalance (nearby levels weighted more)
    /// 
    /// Uses exponential decay weighting based on distance from best price
    #[inline]
    fn calculate_weighted_depth_imbalance(&self, snapshot: &OrderBookSnapshot) -> f64 {
        if snapshot.bids.is_empty() || snapshot.asks.is_empty() {
            return 0.0;
        }

        let best_bid = snapshot.bids[0].price as f64;
        let best_ask = snapshot.asks[0].price as f64;
        let tick_size = (best_ask - best_bid).max(1.0);

        let mut bid_weighted = 0.0;
        let mut ask_weighted = 0.0;

        for (i, level) in snapshot.bids.iter().enumerate() {
            let distance = ((best_bid - level.price) as f64 / tick_size).max(0.0);
            let weight = (-0.5 * distance).exp(); // Exponential decay
            bid_weighted += level.quantity as f64 * weight;
        }

        for (i, level) in snapshot.asks.iter().enumerate() {
            let distance = ((level.price - best_ask) as f64 / tick_size).max(0.0);
            let weight = (-0.5 * distance).exp();
            ask_weighted += level.quantity as f64 * weight;
        }

        if ask_weighted == 0.0 {
            return if bid_weighted > 0.0 { 1.0 } else { 0.0 };
        }

        // Normalize to [-1, 1] range
        let total = bid_weighted + ask_weighted;
        if total == 0.0 {
            0.0
        } else {
            (bid_weighted - ask_weighted) / total
        }
    }

    /// Calculate how quickly liquidity falls off from the best price
    #[inline]
    fn calculate_liquidity_gradients(&self, snapshot: &OrderBookSnapshot) -> (f64, f64) {
        let bid_gradient = if snapshot.bids.len() >= 2 {
            let q0 = snapshot.bids[0].quantity as f64;
            let q1 = snapshot.bids[1].quantity as f64;
            if q0 > 0.0 {
                (q1 - q0) / q0
            } else {
                0.0
            }
        } else {
            0.0
        };

        let ask_gradient = if snapshot.asks.len() >= 2 {
            let q0 = snapshot.asks[0].quantity as f64;
            let q1 = snapshot.asks[1].quantity as f64;
            if q0 > 0.0 {
                (q1 - q0) / q0
            } else {
                0.0
            }
        } else {
            0.0
        };

        (bid_gradient, ask_gradient)
    }

    /// Detect sudden micro-jumps in best bid/ask prices
    #[inline]
    fn detect_micro_jumps(&mut self, snapshot: &OrderBookSnapshot) -> (f64, f64) {
        let bid_jump = match (snapshot.best_bid(), self.prev_best_bid) {
            (Some(curr), Some(prev)) if prev != 0 => {
                (curr.price - prev) as f64 / prev as f64 * 10000.0 // In basis points
            }
            _ => 0.0,
        };

        let ask_jump = match (snapshot.best_ask(), self.prev_best_ask) {
            (Some(curr), Some(prev)) if prev != 0 => {
                (curr.price - prev) as f64 / prev as f64 * 10000.0
            }
            _ => 0.0,
        };

        (bid_jump, ask_jump)
    }

    /// Compute overall shape score combining all signals
    #[inline]
    fn compute_shape_score(&self, shape: &OrderBookShape) -> f64 {
        // Normalize each component to roughly [-1, 1] range
        let convexity_signal = shape.bid_convexity - shape.ask_convexity;
        let convexity_normalized = (convexity_signal / 10.0).tanh();

        let imbalance_normalized = shape.weighted_depth_imbalance.clamp(-1.0, 1.0);

        let gradient_signal = shape.ask_liquidity_gradient - shape.bid_liquidity_gradient;
        let gradient_normalized = (gradient_signal / 2.0).tanh();

        let jump_signal = shape.bid_micro_jump - shape.ask_micro_jump;
        let jump_normalized = (jump_signal / 100.0).tanh();

        // Weighted combination
        let score = self.convexity_weight * convexity_normalized
            + self.imbalance_weight * imbalance_normalized
            + self.gradient_weight * gradient_normalized
            + self.micro_jump_weight * jump_normalized;

        score.clamp(-1.0, 1.0)
    }

    /// Predict short-term momentum based on shape analysis
    #[inline]
    fn predict_momentum(&self, shape: &OrderBookShape) -> (f64, f64) {
        // Base prediction on shape score
        let base_prediction = shape.shape_score * 5.0; // Scale to reasonable bps

        // Adjust based on historical accuracy (simple adaptive mechanism)
        let mut confidence = shape.shape_score.abs();

        // Boost confidence if multiple signals agree
        let signal_agreement = self.calculate_signal_agreement(shape);
        confidence = (confidence + signal_agreement) / 2.0;

        // Apply momentum from recent shape history
        if self.shape_history.len() >= 3 {
            let recent_scores: Vec<f64> = self.shape_history.iter()
                .take(3)
                .map(|s| s.shape_score)
                .collect();
            
            let momentum = recent_scores[0] - recent_scores[2];
            let adjusted_prediction = base_prediction + momentum * 2.0;
            
            (adjusted_prediction.clamp(-50.0, 50.0), confidence)
        } else {
            (base_prediction.clamp(-50.0, 50.0), confidence)
        }
    }

    /// Calculate how many signals are pointing in the same direction
    #[inline]
    fn calculate_signal_agreement(&self, shape: &OrderBookShape) -> f64 {
        let signals = [
            shape.bid_convexity - shape.ask_convexity,
            shape.weighted_depth_imbalance,
            shape.bid_micro_jump - shape.ask_micro_jump,
        ];

        let positive = signals.iter().filter(|&&s| s > 0.0).count();
        let negative = signals.iter().filter(|&&s| s < 0.0).count();

        (positive.max(negative) as f64 / signals.len() as f64)
    }

    /// Update calibration weights based on recent performance
    /// Should be called periodically with feedback from actual price movements
    pub fn calibrate(&mut self, actual_movement_bps: f64, predictions: &[f64]) {
        // Simple gradient descent on weights
        // In production, this would use more sophisticated online learning
        
        if predictions.is_empty() {
            return;
        }

        let error = actual_movement_bps - predictions.iter().sum::<f64>() / predictions.len() as f64;
        let learning_rate = 0.001;

        // Adjust weights slightly based on error
        // This is a simplified version - full implementation would track
        // which features contributed most to the error
        self.convexity_weight += learning_rate * error * 0.1;
        self.imbalance_weight += learning_rate * error * 0.1;
        self.gradient_weight += learning_rate * error * 0.1;
        self.micro_jump_weight += learning_rate * error * 0.1;

        // Renormalize weights to sum to 1.0
        let total = self.convexity_weight + self.imbalance_weight 
            + self.gradient_weight + self.micro_jump_weight;
        
        if total > 0.0 {
            self.convexity_weight /= total;
            self.imbalance_weight /= total;
            self.gradient_weight /= total;
            self.micro_jump_weight /= total;
        }
    }
}

/// Streaming analyzer that maintains internal state and provides continuous updates
pub struct StreamingShapeAnalyzer {
    core_analyzer: OrderBookShapeAnalyzer,
    /// Sliding window of recent order books for comparative analysis
    book_window: VecDeque<OrderBookSnapshot>,
    window_size: usize,
    /// Statistics for monitoring analyzer health
    pub analysis_count: u64,
    pub avg_latency_ns: f64,
}

impl StreamingShapeAnalyzer {
    pub fn new(window_size: usize, history_size: usize) -> Self {
        Self {
            core_analyzer: OrderBookShapeAnalyzer::new(history_size),
            book_window: VecDeque::with_capacity(window_size),
            window_size,
            analysis_count: 0,
            avg_latency_ns: 0.0,
        }
    }

    /// Process a new order book snapshot and return shape analysis
    /// 
    /// This method is optimized for the hot path and should be called
    /// directly from the order book update callback.
    pub fn process_snapshot(&mut self, snapshot: OrderBookSnapshot) -> OrderBookShape {
        let start = std::time::Instant::now();

        let shape = self.core_analyzer.analyze(&snapshot);

        // Update sliding window
        self.book_window.push_back(snapshot);
        if self.book_window.len() > self.window_size {
            self.book_window.pop_front();
        }

        // Update statistics
        let latency_ns = start.elapsed().as_nanos() as f64;
        self.analysis_count += 1;
        self.avg_latency_ns = (self.avg_latency_ns * (self.analysis_count - 1) as f64 
            + latency_ns) / self.analysis_count as f64;

        shape
    }

    /// Get the current shape score without processing a new snapshot
    pub fn current_score(&self) -> Option<f64> {
        self.core_analyzer.shape_history.back().map(|s| s.shape_score)
    }

    /// Get the predicted momentum
    pub fn current_prediction(&self) -> Option<(f64, f64)> {
        self.core_analyzer.shape_history.back()
            .map(|s| (s.predicted_momentum_bps, s.prediction_confidence))
    }

    /// Check if there's a liquidity vacuum forming
    /// Returns true if both sides show rapidly decreasing depth
    pub fn detecting_liquidity_vacuum(&self) -> bool {
        if self.book_window.len() < 3 {
            return false;
        }

        let recent: Vec<_> = self.book_window.iter().rev().take(3).collect();
        
        // Check if total depth is rapidly decreasing on both sides
        let depths: Vec<(f64, f64)> = recent.iter().map(|b| {
            let bid_depth: f64 = b.bids.iter().map(|l| l.quantity as f64).sum();
            let ask_depth: f64 = b.asks.iter().map(|l| l.quantity as f64).sum();
            (bid_depth, ask_depth)
        }).collect();

        let bid_declining = depths[0].0 < depths[1].0 * 0.9 && depths[1].0 < depths[2].0 * 0.9;
        let ask_declining = depths[0].1 < depths[1].1 * 0.9 && depths[1].1 < depths[2].1 * 0.9;

        bid_declining && ask_declining
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_level(price: i64, quantity: u64) -> PriceLevel {
        PriceLevel {
            price,
            quantity,
            order_count: 1,
            timestamp_ns: 0,
        }
    }

    #[test]
    fn test_convexity_calculation() {
        let mut analyzer = OrderBookShapeAnalyzer::new(10);
        
        // Create a convex bid curve (increasing depth)
        let bids = vec![
            create_test_level(100, 100),
            create_test_level(99, 150),
            create_test_level(98, 225),
        ];
        let asks = vec![
            create_test_level(101, 100),
            create_test_level(102, 100),
            create_test_level(103, 100),
        ];

        let convexity = analyzer.calculate_convexity(&bids);
        assert!(convexity > 0.0, "Convex curve should have positive convexity");
    }

    #[test]
    fn test_depth_imbalance() {
        let mut analyzer = OrderBookShapeAnalyzer::new(10);
        
        let bids = vec![
            create_test_level(100, 200),
            create_test_level(99, 200),
        ];
        let asks = vec![
            create_test_level(101, 100),
            create_test_level(102, 100),
        ];

        let mut snapshot = OrderBookSnapshot::new(10);
        snapshot.update(bids, asks);

        let ratio = analyzer.calculate_depth_imbalance(&snapshot);
        assert!(ratio > 1.0, "Bid-heavy book should have ratio > 1");
    }
}
