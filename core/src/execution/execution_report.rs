//! core/src/execution/execution_report.rs
//!
//! Lock-free aggregation of execution reports, fill rates, and latency metrics.
//! Normalizes exchange-specific fill messages into a unified internal format
//! for the risk engine.
//!
//! Features:
//! - Zero-copy message normalization
//! - Atomic counters for real-time metrics
//! - Pre-allocated buffers to prevent heap churn
//!
//! Target Hardware: AMD Ryzen AI 5 (NUMA-aware allocation)
//! Memory Constraint: Strictly bounded report queues.

use std::sync::atomic::{AtomicU64, AtomicUsize, AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use crossbeam_channel::{bounded, Sender, Receiver};

/// Maximum number of pending reports in the queue.
const REPORT_QUEUE_SIZE: usize = 16384;

/// Unified fill status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FillStatus {
    New = 0,
    PartiallyFilled = 1,
    Filled = 2,
    Cancelled = 3,
    Rejected = 4,
    Expired = 5,
}

/// Normalized execution report.
/// Designed to be cache-line friendly (64 bytes aligned where possible).
#[derive(Debug, Clone)]
pub struct ExecutionReport {
    pub report_id: u64,
    pub order_id: u64,
    pub client_order_id: [u8; 32], // Fixed size for zero-copy
    pub symbol: [u8; 12],
    pub side: u8, // 0=Buy, 1=Sell
    pub order_type: u8, // 0=Limit, 1=Market, 2=StopLimit
    pub status: FillStatus,
    pub price_u64: u64, // Scaled by 1e8
    pub quantity_u64: u64,
    pub filled_quantity_u64: u64,
    pub remaining_quantity_u64: u64,
    pub avg_fill_price_u64: u64,
    pub commission_u64: u64,
    pub commission_asset: [u8; 8],
    pub timestamp_ns: u64,
    pub exchange_timestamp_ns: u64,
    pub latency_ns: u64,
    pub venue_id: u8,
}

impl ExecutionReport {
    /// Create a new report from raw exchange data.
    #[inline]
    pub fn new(
        order_id: u64,
        client_order_id: &str,
        symbol: &str,
        side: u8,
        order_type: u8,
        status: FillStatus,
        price: f64,
        quantity: f64,
        filled_qty: f64,
        avg_fill_price: f64,
        commission: f64,
        commission_asset: &str,
        exchange_ts_ns: u64,
        venue_id: u8,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let mut client_oid = [0u8; 32];
        let mut sym = [0u8; 12];
        let mut comm_asset = [0u8; 8];

        client_oid[..client_order_id.len().min(32)].copy_from_slice(&client_order_id.as_bytes()[..client_order_id.len().min(32)]);
        sym[..symbol.len().min(12)].copy_from_slice(&symbol.as_bytes()[..symbol.len().min(12)]);
        comm_asset[..commission_asset.len().min(8)].copy_from_slice(&commission_asset.as_bytes()[..commission_asset.len().min(8)]);

        let scale = 1e8;
        let remaining = (quantity - filled_qty).max(0.0);

        Self {
            report_id: 0, // Set by aggregator
            order_id,
            client_order_id: client_oid,
            symbol: sym,
            side,
            order_type,
            status,
            price_u64: (price * scale) as u64,
            quantity_u64: (quantity * scale) as u64,
            filled_quantity_u64: (filled_qty * scale) as u64,
            remaining_quantity_u64: (remaining * scale) as u64,
            avg_fill_price_u64: (avg_fill_price * scale) as u64,
            commission_u64: (commission * scale) as u64,
            commission_asset: comm_asset,
            timestamp_ns: now,
            exchange_timestamp_ns: exchange_ts_ns,
            latency_ns: now.saturating_sub(exchange_ts_ns),
            venue_id,
        }
    }
}

/// Aggregated metrics for a single order.
#[derive(Debug, Default)]
pub struct OrderMetrics {
    pub total_filled: u64,
    pub avg_fill_price: f64,
    pub total_commission: f64,
    pub fill_count: u32,
    pub first_fill_ts: u64,
    pub last_fill_ts: u64,
    pub is_complete: bool,
}

/// Lock-free execution report aggregator.
pub struct ExecutionReportAggregator {
    report_queue: Sender<ExecutionReport>,
    report_receiver: Receiver<ExecutionReport>,
    next_report_id: AtomicU64,
    
    // Real-time counters (lock-free)
    total_reports_received: AtomicU64,
    total_fills: AtomicU64,
    total_rejects: AtomicU64,
    total_cancels: AtomicU64,
    
    // Latency tracking
    min_latency_ns: AtomicU64,
    max_latency_ns: AtomicU64,
    sum_latency_ns: AtomicU64,
    
    // Fill rate tracking
    orders_tracked: AtomicUsize,
    completed_orders: AtomicUsize,
    
    running: AtomicBool,
}

impl ExecutionReportAggregator {
    pub fn new() -> Self {
        let (tx, rx) = bounded::<ExecutionReport>(REPORT_QUEUE_SIZE);
        
        Self {
            report_queue: tx,
            report_receiver: rx,
            next_report_id: AtomicU64::new(1),
            total_reports_received: AtomicU64::new(0),
            total_fills: AtomicU64::new(0),
            total_rejects: AtomicU64::new(0),
            total_cancels: AtomicU64::new(0),
            min_latency_ns: AtomicU64::new(u64::MAX),
            max_latency_ns: AtomicU64::new(0),
            sum_latency_ns: AtomicU64::new(0),
            orders_tracked: AtomicUsize::new(0),
            completed_orders: AtomicUsize::new(0),
            running: AtomicBool::new(true),
        }
    }

    /// Submit a report for aggregation (non-blocking).
    #[inline]
    pub fn submit_report(&self, mut report: ExecutionReport) -> Result<(), ()> {
        if !self.running.load(Ordering::Relaxed) {
            return Err(());
        }
        
        report.report_id = self.next_report_id.fetch_add(1, Ordering::Relaxed);
        
        // Update counters based on status
        match report.status {
            FillStatus::Filled | FillStatus::PartiallyFilled => {
                self.total_fills.fetch_add(1, Ordering::Relaxed);
                self.update_latency(report.latency_ns);
                
                if report.status == FillStatus::Filled {
                    self.completed_orders.fetch_add(1, Ordering::Relaxed);
                }
            }
            FillStatus::Rejected => {
                self.total_rejects.fetch_add(1, Ordering::Relaxed);
            }
            FillStatus::Cancelled | FillStatus::Expired => {
                self.total_cancels.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        
        self.total_reports_received.fetch_add(1, Ordering::Relaxed);
        
        // Non-blocking send - drop if queue is full (should not happen with proper sizing)
        self.report_queue.try_send(report).map_err(|_| ())
    }

    /// Update latency statistics atomically.
    fn update_latency(&self, latency_ns: u64) {
        // Update min (CAS loop)
        let mut current_min = self.min_latency_ns.load(Ordering::Relaxed);
        while latency_ns < current_min {
            match self.min_latency_ns.compare_exchange_weak(
                current_min,
                latency_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_min = x,
            }
        }
        
        // Update max (CAS loop)
        let mut current_max = self.max_latency_ns.load(Ordering::Relaxed);
        while latency_ns > current_max {
            match self.max_latency_ns.compare_exchange_weak(
                current_max,
                latency_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => current_max = x,
            }
        }
        
        // Update sum (fetch_add is sufficient)
        self.sum_latency_ns.fetch_add(latency_ns, Ordering::Relaxed);
    }

    /// Process pending reports (call from consumer thread).
    pub fn process_pending<F>(&self, mut handler: F) -> usize
    where
        F: FnMut(&ExecutionReport),
    {
        let mut count = 0;
        while let Ok(report) = self.report_receiver.try_recv() {
            handler(&report);
            count += 1;
        }
        count
    }

    /// Get real-time fill rate (fills / total reports).
    pub fn get_fill_rate(&self) -> f64 {
        let total = self.total_reports_received.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.total_fills.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Get rejection rate.
    pub fn get_reject_rate(&self) -> f64 {
        let total = self.total_reports_received.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.total_rejects.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Get average latency in microseconds.
    pub fn get_avg_latency_us(&self) -> f64 {
        let total = self.total_fills.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        (self.sum_latency_ns.load(Ordering::Relaxed) as f64 / total as f64) / 1000.0
    }

    /// Get min latency in microseconds.
    pub fn get_min_latency_us(&self) -> f64 {
        let min = self.min_latency_ns.load(Ordering::Relaxed);
        if min == u64::MAX {
            return 0.0;
        }
        min as f64 / 1000.0
    }

    /// Get max latency in microseconds.
    pub fn get_max_latency_us(&self) -> f64 {
        self.max_latency_ns.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Get completion rate (fully filled orders / tracked orders).
    pub fn get_completion_rate(&self) -> f64 {
        let tracked = self.orders_tracked.load(Ordering::Relaxed);
        if tracked == 0 {
            return 0.0;
        }
        self.completed_orders.load(Ordering::Relaxed) as f64 / tracked as f64
    }

    /// Get summary statistics.
    pub fn get_summary(&self) -> ExecutionSummary {
        ExecutionSummary {
            total_reports: self.total_reports_received.load(Ordering::Relaxed),
            total_fills: self.total_fills.load(Ordering::Relaxed),
            total_rejects: self.total_rejects.load(Ordering::Relaxed),
            total_cancels: self.total_cancels.load(Ordering::Relaxed),
            fill_rate: self.get_fill_rate(),
            reject_rate: self.get_reject_rate(),
            avg_latency_us: self.get_avg_latency_us(),
            min_latency_us: self.get_min_latency_us(),
            max_latency_us: self.get_max_latency_us(),
            completed_orders: self.completed_orders.load(Ordering::Relaxed) as u64,
        }
    }

    pub fn shutdown(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

/// Summary snapshot of execution metrics.
#[derive(Debug, Default)]
pub struct ExecutionSummary {
    pub total_reports: u64,
    pub total_fills: u64,
    pub total_rejects: u64,
    pub total_cancels: u64,
    pub fill_rate: f64,
    pub reject_rate: f64,
    pub avg_latency_us: f64,
    pub min_latency_us: f64,
    pub max_latency_us: f64,
    pub completed_orders: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_aggregation() {
        let agg = ExecutionReportAggregator::new();
        
        // Simulate a fill
        let report = ExecutionReport::new(
            12345,
            "client_oid_001",
            "BTCUSDT",
            0, // Buy
            0, // Limit
            FillStatus::Filled,
            50000.0,
            1.0,
            1.0,
            50000.0,
            0.001,
            "BTC",
            1000000000,
            1,
        );
        
        assert!(agg.submit_report(report).is_ok());
        
        // Check counters
        assert_eq!(agg.total_reports_received.load(Ordering::Relaxed), 1);
        assert_eq!(agg.total_fills.load(Ordering::Relaxed), 1);
        assert_eq!(agg.completed_orders.load(Ordering::Relaxed), 1);
        
        let summary = agg.get_summary();
        assert_eq!(summary.total_fills, 1);
        assert!((summary.fill_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_latency_tracking() {
        let agg = ExecutionReportAggregator::new();
        
        // Submit reports with different latencies
        for i in 1..=5 {
            let mut report = ExecutionReport::new(
                i,
                &format!("oid_{}", i),
                "ETHUSDT",
                0,
                0,
                FillStatus::Filled,
                3000.0,
                10.0,
                10.0,
                3000.0,
                0.01,
                "ETH",
                1000000000,
                1,
            );
            report.latency_ns = i * 1000; // 1us, 2us, ..., 5us
            
            agg.submit_report(report).unwrap();
        }
        
        assert!((agg.get_avg_latency_us() - 3.0).abs() < 0.001);
        assert!((agg.get_min_latency_us() - 1.0).abs() < 0.001);
        assert!((agg.get_max_latency_us() - 5.0).abs() < 0.001);
    }
}
