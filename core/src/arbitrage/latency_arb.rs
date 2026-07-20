//! Latency Arbitrage Engine
//! 
//! Exploits microsecond latency differences between exchanges using predictive
//! order book modeling and highly optimized network I/O.
//! 
//! This module implements "safe front-running" by predicting price movements
//! on slower exchanges based on faster exchange data, accounting for network
//! latency and execution risk.
//! 
//! Hardware Target: AMD Ryzen AI 5 with DPDK-compatible network stack
//! Memory Constraint: Zero-allocation hot paths, pre-allocated buffers

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Instant, Duration};
use crossbeam::queue::SegQueue;
use dashmap::DashMap;

/// Network latency measurement between two endpoints
#[derive(Debug, Clone, Copy)]
pub struct LatencyMeasurement {
    pub source_exchange: u16,
    pub dest_exchange: u16,
    pub latency_ns: u64,
    pub jitter_ns: u64,
    pub last_measured: u64,
    pub sample_count: u32,
}

/// Predictive order book state for latency arbitrage
#[derive(Debug, Clone)]
pub struct PredictedBook {
    pub symbol: u64,
    pub exchange_id: u16,
    pub predicted_bid: u64,
    pub predicted_ask: u64,
    pub predicted_bid_size: u64,
    pub predicted_ask_size: u64,
    pub confidence: f32,
    pub prediction_horizon_ns: u64,
    pub created_at_ns: u64,
}

/// Latency arbitrage opportunity
#[derive(Debug, Clone)]
pub struct LatencyArbOpportunity {
    pub fast_exchange: u16,
    pub slow_exchange: u16,
    pub symbol: u64,
    pub direction: TradeDirection, // Buy or Sell on slow exchange
    pub expected_price_move_bps: i32,
    pub latency_advantage_ns: u64,
    pub recommended_order_type: OrderType,
    pub max_execution_time_ns: u64,
    pub confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeDirection {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderType {
    Market,
    Limit(i64), // Price offset in basis points
    ImmediateOrCancel,
}

/// Main latency arbitrage engine
pub struct LatencyArbEngine {
    /// Measured latencies between all exchange pairs
    latency_matrix: DashMap<(u16, u16), LatencyMeasurement>,
    
    /// Latest known prices per exchange
    latest_prices: DashMap<(u16, u64), PriceData>,
    
    /// Predicted book states
    predicted_books: DashMap<(u16, u64), PredictedBook>,
    
    /// Opportunity queue for execution
    opportunity_queue: Arc<SegQueue<LatencyArbOpportunity>>,
    
    /// Historical price moves for pattern recognition
    price_history: DashMap<u64, RingBuffer<PriceMove>>,
    
    /// Minimum latency advantage required (nanoseconds)
    min_latency_advantage_ns: u64,
    
    /// Running flag
    is_running: AtomicBool,
    
    /// Statistics
    predictions_made: AtomicU64,
    successful_arbs: AtomicU64,
    failed_arbs: AtomicU64,
}

#[derive(Debug, Clone, Copy)]
pub struct PriceData {
    pub bid: u64,
    pub ask: u64,
    pub bid_size: u64,
    pub ask_size: u64,
    pub timestamp_ns: u64,
    pub sequence_number: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct PriceMove {
    pub delta_bps: i32,
    pub timestamp_ns: u64,
    pub volume: u64,
}

/// Fixed-size ring buffer for zero-allocation history storage
struct RingBuffer<T: Copy + Default> {
    buffer: Vec<T>,
    head: AtomicU64,
    size: usize,
}

impl<T: Copy + Default> RingBuffer<T> {
    fn new(size: usize) -> Self {
        let mut buffer = Vec::with_capacity(size);
        buffer.resize_with(size, || T::default());
        Self {
            buffer,
            head: AtomicU64::new(0),
            size,
        }
    }

    #[inline]
    fn push(&self, item: T) {
        let idx = self.head.fetch_add(1, Ordering::Relaxed) % self.size as u64;
        unsafe {
            let ptr = &self.buffer[idx as usize] as *const T as *mut T;
            ptr.write(item);
        }
    }

    #[inline]
    fn get_recent(&self, count: usize) -> Vec<T> {
        let head = self.head.load(Ordering::Relaxed);
        let mut result = Vec::with_capacity(count.min(self.size));
        
        for i in 0..count.min(self.size) {
            let idx = ((head as usize).saturating_sub(i + 1)) % self.size;
            result.push(self.buffer[idx]);
        }
        
        result
    }
}

impl LatencyArbEngine {
    pub fn new(min_latency_advantage_ns: u64) -> Self {
        Self {
            latency_matrix: DashMap::with_capacity(256),
            latest_prices: DashMap::with_capacity(4096),
            predicted_books: DashMap::with_capacity(4096),
            opportunity_queue: Arc::new(SegQueue::new()),
            price_history: DashMap::with_capacity(1024),
            min_latency_advantage_ns,
            is_running: AtomicBool::new(false),
            predictions_made: AtomicU64::new(0),
            successful_arbs: AtomicU64::new(0),
            failed_arbs: AtomicU64::new(0),
        }
    }

    /// Record a latency measurement between two exchanges
    pub fn record_latency(&self, measurement: LatencyMeasurement) {
        let key = (measurement.source_exchange, measurement.dest_exchange);
        
        self.latency_matrix.entry(key).and_modify(|existing| {
            // Exponential moving average for latency
            let alpha = 0.1;
            existing.latency_ns = ((existing.latency_ns as f64 * (1.0 - alpha)) 
                + (measurement.latency_ns as f64 * alpha)) as u64;
            
            // Update jitter estimate
            let diff = (measurement.latency_ns as i64 - existing.latency_ns as i64).abs();
            existing.jitter_ns = ((existing.jitter_ns as f64 * (1.0 - alpha)) 
                + (diff as f64 * alpha)) as u64;
            
            existing.last_measured = measurement.last_measured;
            existing.sample_count = existing.sample_count.saturating_add(1);
        }).or_insert(measurement);
    }

    /// Update price data from an exchange
    #[inline(always)]
    pub fn update_price(&self, exchange_id: u16, symbol: u64, price: PriceData) {
        let now_ns = price.timestamp_ns;
        
        // Record price move for pattern analysis
        if let Some(existing) = self.latest_prices.get(&(exchange_id, symbol)) {
            let delta_bps = if existing.ask > 0 {
                ((price.bid as i64 - existing.bid as i64) * 10000 / existing.bid as i64) as i32
            } else {
                0
            };
            
            let move_data = PriceMove {
                delta_bps,
                timestamp_ns: now_ns,
                volume: price.bid_size + price.ask_size,
            };
            
            self.price_history
                .entry(symbol)
                .or_insert_with(|| RingBuffer::new(1024))
                .push(move_data);
        }
        
        self.latest_prices.insert((exchange_id, symbol), price);
    }

    /// Predict price movement on slow exchange based on fast exchange data
    pub fn predict_price_movement(&self, fast_exchange: u16, slow_exchange: u16, symbol: u64) -> Option<PredictedBook> {
        let fast_price = self.latest_prices.get(&(fast_exchange, symbol))?;
        let slow_price = self.latest_prices.get(&(slow_exchange, symbol))?;
        
        let latency_info = self.latency_matrix.get(&(fast_exchange, slow_exchange))?;
        
        // Calculate expected price on slow exchange given latency
        let latency_ms = latency_info.latency_ns as f64 / 1_000_000.0;
        
        // Get recent price momentum from fast exchange
        let history = self.price_history.get(&symbol)?;
        let recent_moves = history.get_recent(10);
        
        if recent_moves.is_empty() {
            return None;
        }
        
        // Calculate weighted momentum (recent moves weighted more heavily)
        let mut momentum = 0.0;
        let mut total_weight = 0.0;
        
        for (i, move_data) in recent_moves.iter().enumerate() {
            let weight = 1.0 - (i as f64 / recent_moves.len() as f64);
            momentum += move_data.delta_bps as f64 * weight;
            total_weight += weight;
        }
        
        let avg_momentum = if total_weight > 0.0 {
            momentum / total_weight
        } else {
            0.0
        };
        
        // Predict price movement over latency period
        let expected_move_bps = (avg_momentum * latency_ms / 10.0) as i64;
        
        let predicted_bid = if expected_move_bps > 0 {
            slow_price.bid as i64 + (slow_price.bid as i64 * expected_move_bps / 10000)
        } else {
            slow_price.bid as i64 + (slow_price.bid as i64 * expected_move_bps / 10000)
        } as u64;
        
        let predicted_ask = if expected_move_bps > 0 {
            slow_price.ask as i64 + (slow_price.ask as i64 * expected_move_bps / 10000)
        } else {
            slow_price.ask as i64 + (slow_price.ask as i64 * expected_move_bps / 10000)
        } as u64;
        
        // Confidence based on latency certainty and momentum consistency
        let momentum_variance = recent_moves.iter()
            .map(|m| (m.delta_bps as f64 - avg_momentum).powi(2))
            .sum::<f64>() / recent_moves.len() as f64;
        
        let confidence = (1.0 / (1.0 + momentum_variance / 100.0)) 
            * (1.0 - (latency_info.jitter_ns as f64 / latency_info.latency_ns as f64).min(1.0));
        
        let prediction = PredictedBook {
            symbol,
            exchange_id: slow_exchange,
            predicted_bid,
            predicted_ask,
            predicted_bid_size: slow_price.bid_size,
            predicted_ask_size: slow_price.ask_size,
            confidence: confidence as f32,
            prediction_horizon_ns: latency_info.latency_ns,
            created_at_ns: get_timestamp_ns(),
        };
        
        self.predictions_made.fetch_add(1, Ordering::Relaxed);
        self.predicted_books.insert((slow_exchange, symbol), prediction.clone());
        
        Some(prediction)
    }

    /// Detect latency arbitrage opportunities
    pub fn detect_opportunities(&self) -> Vec<LatencyArbOpportunity> {
        let mut opportunities = Vec::with_capacity(64);
        
        // Iterate through all exchange pairs
        for latency_entry in self.latency_matrix.iter() {
            let (fast_ex, slow_ex) = *latency_entry.key();
            let latency = latency_entry.value();
            
            // Skip if latency advantage is insufficient
            if latency.latency_ns < self.min_latency_advantage_ns {
                continue;
            }
            
            // Check all symbols for this exchange pair
            for price_entry in self.latest_prices.iter() {
                let (ex_id, symbol) = *price_entry.key();
                if ex_id != fast_ex {
                    continue;
                }
                
                if let Some(predicted) = self.predict_price_movement(fast_ex, slow_ex, *symbol) {
                    let current_slow = self.latest_prices.get(&(slow_ex, *symbol));
                    
                    if let Some(slow_price) = current_slow {
                        // Determine if there's an arbitrage opportunity
                        let expected_upside = (predicted.predicted_bid as i64 - slow_price.ask as i64) * 10000 / slow_price.ask as i64;
                        let expected_downside = (slow_price.bid as i64 - predicted.predicted_ask as i64) * 10000 / slow_price.bid as i64;
                        
                        if expected_upside > 5 && predicted.confidence > 0.7 {
                            // Buy on slow exchange before price goes up
                            opportunities.push(LatencyArbOpportunity {
                                fast_exchange: fast_ex,
                                slow_exchange: slow_ex,
                                symbol: *symbol,
                                direction: TradeDirection::Buy,
                                expected_price_move_bps: expected_upside as i32,
                                latency_advantage_ns: latency.latency_ns,
                                recommended_order_type: OrderType::ImmediateOrCancel,
                                max_execution_time_ns: latency.latency_ns / 2,
                                confidence: predicted.confidence,
                            });
                        } else if expected_downside > 5 && predicted.confidence > 0.7 {
                            // Sell on slow exchange before price goes down
                            opportunities.push(LatencyArbOpportunity {
                                fast_exchange: fast_ex,
                                slow_exchange: slow_ex,
                                symbol: *symbol,
                                direction: TradeDirection::Sell,
                                expected_price_move_bps: expected_downside as i32,
                                latency_advantage_ns: latency.latency_ns,
                                recommended_order_type: OrderType::ImmediateOrCancel,
                                max_execution_time_ns: latency.latency_ns / 2,
                                confidence: predicted.confidence,
                            });
                        }
                    }
                }
            }
        }
        
        // Queue opportunities for execution
        for opp in &opportunities {
            self.opportunity_queue.push(opp.clone());
        }
        
        opportunities
    }

    /// Get next opportunity for execution
    pub fn next_opportunity(&self) -> Option<LatencyArbOpportunity> {
        self.opportunity_queue.pop()
    }

    /// Record execution result for learning
    pub fn record_result(&self, success: bool) {
        if success {
            self.successful_arbs.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_arbs.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get engine statistics
    pub fn get_stats(&self) -> LatencyArbStats {
        LatencyArbStats {
            predictions_made: self.predictions_made.load(Ordering::Relaxed),
            successful_arbs: self.successful_arbs.load(Ordering::Relaxed),
            failed_arbs: self.failed_arbs.load(Ordering::Relaxed),
            measured_paths: self.latency_matrix.len(),
            tracked_symbols: self.latest_prices.len(),
        }
    }
}

#[derive(Debug)]
pub struct LatencyArbStats {
    pub predictions_made: u64,
    pub successful_arbs: u64,
    pub failed_arbs: u64,
    pub measured_paths: usize,
    pub tracked_symbols: usize,
}

/// Get current timestamp in nanoseconds
#[inline(always)]
fn get_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_recording() {
        let engine = LatencyArbEngine::new(100_000); // 100us minimum
        
        let measurement = LatencyMeasurement {
            source_exchange: 1,
            dest_exchange: 2,
            latency_ns: 5_000_000, // 5ms
            jitter_ns: 500_000,
            last_measured: get_timestamp_ns(),
            sample_count: 1,
        };
        
        engine.record_latency(measurement);
        
        let stats = engine.get_stats();
        assert_eq!(stats.measured_paths, 1);
    }

    #[test]
    fn test_price_prediction() {
        let engine = LatencyArbEngine::new(100_000);
        
        // Setup latency
        engine.record_latency(LatencyMeasurement {
            source_exchange: 1,
            dest_exchange: 2,
            latency_ns: 10_000_000, // 10ms
            jitter_ns: 1_000_000,
            last_measured: get_timestamp_ns(),
            sample_count: 100,
        });
        
        // Setup prices
        engine.update_price(1, 12345, PriceData {
            bid: 50000_00000000,
            ask: 50001_00000000,
            bid_size: 100_00000000,
            ask_size: 100_00000000,
            timestamp_ns: get_timestamp_ns(),
            sequence_number: 1,
        });
        
        engine.update_price(2, 12345, PriceData {
            bid: 49990_00000000,
            ask: 49991_00000000,
            bid_size: 100_00000000,
            ask_size: 100_00000000,
            timestamp_ns: get_timestamp_ns() - 5_000_000, // 5ms behind
            sequence_number: 1,
        });
        
        // Add some price history for momentum calculation
        if let Some(history) = engine.price_history.get(&12345) {
            for i in 0..10 {
                history.push(PriceMove {
                    delta_bps: 2, // Upward momentum
                    timestamp_ns: get_timestamp_ns() - i * 1_000_000,
                    volume: 100_00000000,
                });
            }
        }
        
        let prediction = engine.predict_price_movement(1, 2, 12345);
        assert!(prediction.is_some());
    }
}
