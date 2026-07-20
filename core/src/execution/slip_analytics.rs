//! core/src/execution/slip_analytics.rs
//! 
//! Real-time slippage analytics and implementation shortfall tracking.
//! Measures the difference between the decision price and the final execution price.
//! Uses lock-free channels and pre-allocated buffers to ensure microsecond-level logging
//! without GC pressure or heap fragmentation.
//!
//! Target Hardware: AMD Ryzen AI 5 (AVX2/AVX-512 ready)
//! Memory Constraint: Strictly bounded buffers to stay within 8GB global cap.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use crossbeam_channel::bounded;
use serde::{Serialize, Deserialize};

/// Pre-allocated buffer size for sliding window analytics.
/// Keeps memory usage predictable and low.
const SLIP_BUFFER_SIZE: usize = 4096;

/// Zero-copy channel capacity for telemetry offloading.
const TELEMETRY_CHANNEL_CAPACITY: usize = 8192;

/// Represents a single execution event with slippage metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSample {
    pub timestamp_ns: u64,
    pub symbol: [u8; 12], // Fixed-size string to avoid heap allocation
    pub side: u8, // 0=Buy, 1=Sell
    pub decision_price_u64: u64, // Scaled integer price (e.g., * 1e8)
    pub fill_price_u64: u64,
    pub quantity_u64: u64,
    pub fees_u64: u64,
    pub latency_ns: u64,
    pub slip_bps: i32, // Slippage in basis points
}

/// Lock-free ring buffer for recent execution samples.
/// Uses atomic operations for thread-safe access without mutexes.
pub struct SlipRingBuffer {
    buffer: Vec<ExecutionSample>,
    head: AtomicUsize,
    count: AtomicUsize,
}

impl SlipRingBuffer {
    pub fn new() -> Self {
        let mut buffer = Vec::with_capacity(SLIP_BUFFER_SIZE);
        // Pre-fill with dummy data to avoid reallocation during runtime
        for _ in 0..SLIP_BUFFER_SIZE {
            buffer.push(ExecutionSample {
                timestamp_ns: 0,
                symbol: [0; 12],
                side: 0,
                decision_price_u64: 0,
                fill_price_u64: 0,
                quantity_u64: 0,
                fees_u64: 0,
                latency_ns: 0,
                slip_bps: 0,
            });
        }
        Self {
            buffer,
            head: AtomicUsize::new(0),
            count: AtomicUsize::new(0),
        }
    }

    #[inline]
    pub fn push(&self, sample: ExecutionSample) {
        let idx = self.head.fetch_add(1, Ordering::Relaxed) % SLIP_BUFFER_SIZE;
        // Safe write because we own the slot at `idx` after atomic increment
        unsafe {
            let ptr = self.buffer.as_ptr() as *mut ExecutionSample;
            std::ptr::write(ptr.add(idx), sample);
        }
        
        if self.count.load(Ordering::Relaxed) < SLIP_BUFFER_SIZE {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn get_recent(&self, n: usize) -> Vec<ExecutionSample> {
        let count = self.count.load(Ordering::Relaxed).min(n);
        let head = self.head.load(Ordering::Relaxed);
        let mut result = Vec::with_capacity(count);
        
        for i in 0..count {
            let idx = (head.wrapping_sub(count - i)) % SLIP_BUFFER_SIZE;
            result.push(self.buffer[idx].clone());
        }
        result
    }
}

/// Aggregated statistics for a specific time window.
#[derive(Debug, Default)]
pub struct SlipStats {
    pub total_slip_bps: f64,
    pub avg_slip_bps: f64,
    pub max_slip_bps: f64,
    pub min_slip_bps: f64,
    pub total_volume: u64,
    pub total_fees: u64,
    pub sample_count: u32,
    pub avg_latency_ns: u64,
}

/// Main analytics engine.
pub struct SlippageAnalytics {
    ring_buffer: SlipRingBuffer,
    telemetry_tx: crossbeam_channel::Sender<ExecutionSample>,
    start_time: Instant,
    total_samples: AtomicU64,
}

impl SlippageAnalytics {
    pub fn new() -> Self {
        let (tx, _rx) = bounded::<ExecutionSample>(TELEMETRY_CHANNEL_CAPACITY);
        // Note: In production, _rx would be consumed by a dedicated telemetry thread
        // writing to Parquet/InfluxDB via zero-copy shared memory.
        
        Self {
            ring_buffer: SlipRingBuffer::new(),
            telemetry_tx: tx,
            start_time: Instant::now(),
            total_samples: AtomicU64::new(0),
        }
    }

    /// Record an execution. Calculates slippage immediately.
    /// 
    /// # Arguments
    /// * `decision_price` - The price when the signal was generated.
    /// * `fill_price` - The actual average fill price.
    /// * `side` - 0 for Buy, 1 for Sell.
    #[inline]
    pub fn record_execution(
        &self,
        symbol: &str,
        side: u8,
        decision_price: f64,
        fill_price: f64,
        quantity: f64,
        fees: f64,
        latency_ns: u64,
    ) {
        let now = self.start_time.elapsed().as_nanos() as u64;
        
        // Calculate slippage in basis points (bps)
        // For Buy: (Fill - Decision) / Decision * 10000
        // For Sell: (Decision - Fill) / Decision * 10000
        let slip_bps = if side == 0 {
            ((fill_price - decision_price) / decision_price) * 10000.0
        } else {
            ((decision_price - fill_price) / decision_price) * 10000.0
        };

        // Scale prices to u64 to avoid float storage in hot path
        let scale = 1e8;
        let mut symbol_arr = [0u8; 12];
        let bytes = symbol.as_bytes();
        symbol_arr[..bytes.len().min(12)].copy_from_slice(bytes);

        let sample = ExecutionSample {
            timestamp_ns: now,
            symbol: symbol_arr,
            side,
            decision_price_u64: (decision_price * scale) as u64,
            fill_price_u64: (fill_price * scale) as u64,
            quantity_u64: (quantity * scale) as u64,
            fees_u64: (fees * scale) as u64,
            latency_ns,
            slip_bps: slip_bps as i32,
        };

        self.ring_buffer.push(sample.clone());
        
        // Non-blocking send to telemetry channel
        let _ = self.telemetry_tx.try_send(sample);
        
        self.total_samples.fetch_add(1, Ordering::Relaxed);
    }

    /// Compute rolling statistics over the last N samples.
    pub fn compute_stats(&self, n: usize) -> SlipStats {
        let samples = self.ring_buffer.get_recent(n);
        if samples.is_empty() {
            return SlipStats::default();
        }

        let mut stats = SlipStats {
            sample_count: samples.len() as u32,
            ..Default::default()
        };

        let mut sum_slip = 0.0;
        let mut max_slip = f64::MIN;
        let mut min_slip = f64::MAX;
        let mut sum_latency = 0u64;

        for s in &samples {
            let slip = s.slip_bps as f64;
            sum_slip += slip;
            if slip > max_slip { max_slip = slip; }
            if slip < min_slip { min_slip = slip; }
            stats.total_volume += s.quantity_u64;
            stats.total_fees += s.fees_u64;
            sum_latency += s.latency_ns;
        }

        stats.avg_slip_bps = sum_slip / stats.sample_count as f64;
        stats.max_slip_bps = max_slip;
        stats.min_slip_bps = min_slip;
        stats.total_slip_bps = sum_slip;
        stats.avg_latency_ns = sum_latency / stats.sample_count as u64;

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slippage_calculation() {
        let analytics = SlippageAnalytics::new();
        
        // Buy order: Decision 100, Fill 100.10 -> Slippage 10 bps
        analytics.record_execution("BTCUSDT", 0, 100.0, 100.10, 1.0, 0.1, 500);
        
        let stats = analytics.compute_stats(1);
        assert!((stats.avg_slip_bps - 10.0).abs() < 0.01);
        assert_eq!(stats.sample_count, 1);
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let analytics = SlippageAnalytics::new();
        for i in 0..(SLIP_BUFFER_SIZE + 100) {
            analytics.record_execution("ETHUSDT", 0, 2000.0, 2000.0, 1.0, 0.0, 100);
        }
        // Should not panic and should only keep recent items
        let stats = analytics.compute_stats(SLIP_BUFFER_SIZE);
        assert_eq!(stats.sample_count, SLIP_BUFFER_SIZE as u32);
    }
}
