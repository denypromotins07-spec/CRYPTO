//! twap_vwap_executor.rs
//! =====================
//! High-frequency, mathematically precise implementation of TWAP, VWAP,
//! and Implementation Shortfall execution algorithms.
//!
//! Optimized for AMD Ryzen AI 5 with microsecond-level precision.

use std::time::{Duration, Instant};
use parking_lot::RwLock;
use log::{info, debug, warn};

/// Execution algorithm type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgoType {
    Twap,
    Vwap,
    ImplementationShortfall,
}

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// Configuration for TWAP execution
#[derive(Debug, Clone)]
pub struct TwapConfig {
    /// Total duration in seconds
    pub duration_secs: u64,
    /// Number of slices (child orders)
    pub num_slices: usize,
    /// Randomization factor (0.0-1.0) for timing jitter
    pub randomization: f64,
}

impl Default for TwapConfig {
    fn default() -> Self {
        Self {
            duration_secs: 3600, // 1 hour
            num_slices: 60,      // 1 slice per minute
            randomization: 0.1,  // 10% timing jitter
        }
    }
}

/// Configuration for VWAP execution
#[derive(Debug, Clone)]
pub struct VwapConfig {
    /// Historical volume profile (hourly buckets, 24 hours)
    pub volume_profile: Vec<f64>,
    /// Participation rate limit (max % of market volume)
    pub max_participation: f64,
    /// Minimum child order size
    pub min_child_qty: f64,
}

impl Default for VwapConfig {
    fn default() -> Self {
        // Default flat volume profile
        Self {
            volume_profile: vec![1.0; 24],
            max_participation: 0.15, // Max 15% of volume
            min_child_qty: 0.001,
        }
    }
}

/// Configuration for Implementation Shortfall
#[derive(Debug, Clone)]
pub struct IsConfig {
    /// Urgency parameter (0.0-1.0): higher = more aggressive
    pub urgency: f64,
    /// Risk aversion parameter
    pub risk_aversion: f64,
    /// Price impact model coefficient
    pub impact_coefficient: f64,
    /// Maximum execution time in seconds
    pub max_duration_secs: u64,
}

impl Default for IsConfig {
    fn default() -> Self {
        Self {
            urgency: 0.5,
            risk_aversion: 0.5,
            impact_coefficient: 0.1,
            max_duration_secs: 1800, // 30 minutes
        }
    }
}

/// State of an active execution algorithm
#[derive(Debug)]
pub struct ExecutionState {
    pub algo_id: String,
    pub algo_type: AlgoType,
    pub symbol: String,
    pub side: Side,
    pub total_quantity: f64,
    pub executed_quantity: f64,
    pub avg_executed_price: f64,
    pub start_time: Instant,
    pub end_time: Instant,
    pub is_complete: bool,
    /// Scheduled slices with their target quantities
    pub schedule: Vec<Slice>,
    pub current_slice_idx: usize,
}

/// A single slice in the execution schedule
#[derive(Debug, Clone)]
pub struct Slice {
    pub slice_id: usize,
    pub target_time: Instant,
    pub target_quantity: f64,
    pub executed_quantity: f64,
    pub avg_price: f64,
    pub status: SliceStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceStatus {
    Pending,
    Active,
    Completed,
    Skipped,
}

impl ExecutionState {
    pub fn remaining_qty(&self) -> f64 {
        self.total_quantity - self.executed_quantity
    }

    pub fn progress(&self) -> f64 {
        if self.total_quantity <= 0.0 {
            return 0.0;
        }
        self.executed_quantity / self.total_quantity
    }

    pub fn time_progress(&self) -> f64 {
        let now = Instant::now();
        if now < self.start_time {
            return 0.0;
        }
        if now >= self.end_time {
            return 1.0;
        }
        let elapsed = now.duration_since(self.start_time).as_secs_f64();
        let total = self.end_time.duration_since(self.start_time).as_secs_f64();
        if total <= 0.0 {
            return 1.0;
        }
        (elapsed / total).min(1.0)
    }
}

/// Statistics for execution algorithm performance
#[derive(Debug, Default)]
pub struct ExecutionStatistics {
    pub total_algos_run: u64,
    pub twap_count: u64,
    pub vwap_count: u64,
    pub is_count: u64,
    pub avg_slippage_bps: f64,
    pub avg_completion_rate: f64,
}

/// High-frequency Execution Algorithm Engine
pub struct AlgoExecutor {
    statistics: RwLock<ExecutionStatistics>,
    active_algos: RwLock<std::collections::HashMap<String, ExecutionState>>,
}

impl AlgoExecutor {
    /// Create a new algorithm executor
    pub fn new() -> Self {
        info!("Initializing AlgoExecutor");
        Self {
            statistics: RwLock::new(ExecutionStatistics::default()),
            active_algos: RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Start a TWAP execution
    pub fn start_twap(
        &self,
        algo_id: String,
        symbol: String,
        side: Side,
        quantity: f64,
        config: TwapConfig,
    ) -> ExecutionState {
        let now = Instant::now();
        let end_time = now + Duration::from_secs(config.duration_secs);

        // Calculate slice quantities
        let base_qty = quantity / config.num_slices as f64;
        let mut schedule = Vec::with_capacity(config.num_slices);

        for i in 0..config.num_slices {
            // Add randomization to timing
            let jitter = if config.randomization > 0.0 {
                let random_offset = (i as f64 * 0.17).sin() * config.randomization;
                random_offset * config.duration_secs as f64 / config.num_slices as f64
            } else {
                0.0
            };

            let target_time = now
                + Duration::from_secs_f64(
                    (i as f64 * config.duration_secs as f64 / config.num_slices as f64) + jitter,
                );

            // Vary slice sizes slightly to avoid detection
            let slice_qty = base_qty * (1.0 + (i as f64 * 0.13).sin() * 0.1);

            schedule.push(Slice {
                slice_id: i,
                target_time,
                target_quantity: slice_qty.max(config.min_child_qty()),
                executed_quantity: 0.0,
                avg_price: 0.0,
                status: SliceStatus::Pending,
            });
        }

        let state = ExecutionState {
            algo_id: algo_id.clone(),
            algo_type: AlgoType::Twap,
            symbol,
            side,
            total_quantity: quantity,
            executed_quantity: 0.0,
            avg_executed_price: 0.0,
            start_time: now,
            end_time,
            is_complete: false,
            schedule,
            current_slice_idx: 0,
        };

        // Register the algorithm
        {
            let mut algos = self.active_algos.write();
            algos.insert(algo_id, state.clone());
        }

        // Update statistics
        {
            let mut stats = self.statistics.write();
            stats.total_algos_run += 1;
            stats.twap_count += 1;
        }

        info!(
            "Started TWAP {} | Symbol: {} | Side: {:?} | Qty: {} | Slices: {}",
            algo_id, state.symbol, side, quantity, config.num_slices
        );

        state
    }

    /// Start a VWAP execution
    pub fn start_vwap(
        &self,
        algo_id: String,
        symbol: String,
        side: Side,
        quantity: f64,
        config: VwapConfig,
    ) -> ExecutionState {
        let now = Instant::now();
        let duration_secs = 24 * 3600; // 24 hours for full VWAP
        let end_time = now + Duration::from_secs(duration_secs);

        // Normalize volume profile
        let total_volume: f64 = config.volume_profile.iter().sum();
        let normalized_profile: Vec<f64> = config
            .volume_profile
            .iter()
            .map(|v| if total_volume > 0.0 { v / total_volume } else { 1.0 / 24.0 })
            .collect();

        // Create hourly slices based on volume profile
        let mut schedule = Vec::with_capacity(24);

        for i in 0..24 {
            let target_time = now + Duration::from_secs(i as u64 * 3600);
            let target_quantity = quantity * normalized_profile[i];

            if target_quantity >= config.min_child_qty {
                schedule.push(Slice {
                    slice_id: i,
                    target_time,
                    target_quantity,
                    executed_quantity: 0.0,
                    avg_price: 0.0,
                    status: SliceStatus::Pending,
                });
            }
        }

        let state = ExecutionState {
            algo_id: algo_id.clone(),
            algo_type: AlgoType::Vwap,
            symbol,
            side,
            total_quantity: quantity,
            executed_quantity: 0.0,
            avg_executed_price: 0.0,
            start_time: now,
            end_time,
            is_complete: false,
            schedule,
            current_slice_idx: 0,
        };

        {
            let mut algos = self.active_algos.write();
            algos.insert(algo_id, state.clone());
        }

        {
            let mut stats = self.statistics.write();
            stats.total_algos_run += 1;
            stats.vwap_count += 1;
        }

        info!(
            "Started VWAP {} | Symbol: {} | Side: {:?} | Qty: {}",
            algo_id, state.symbol, side, quantity
        );

        state
    }

    /// Start an Implementation Shortfall execution
    pub fn start_implementation_shortfall(
        &self,
        algo_id: String,
        symbol: String,
        side: Side,
        quantity: f64,
        arrival_price: f64,
        config: IsConfig,
    ) -> ExecutionState {
        let now = Instant::now();
        let end_time = now + Duration::from_secs(config.max_duration_secs);

        // Calculate optimal trading trajectory using Almgren-Chriss model
        // Simplified version: more aggressive = more front-loaded
        let num_slices = 30; // 30 slices for IS
        let mut schedule = Vec::with_capacity(num_slices);

        for i in 0..num_slices {
            let time_fraction = i as f64 / num_slices as f64;

            // Optimal trajectory: exponential decay based on urgency
            // Higher urgency = more trades early
            let cumulative_fraction = if config.urgency > 0.5 {
                // Front-loaded
                1.0 - (-3.0 * config.urgency * time_fraction).exp()
            } else {
                // More linear
                time_fraction.powf(1.0 / (1.0 - config.urgency + 0.01))
            };

            let prev_fraction = if i == 0 {
                0.0
            } else {
                let prev_time = (i - 1) as f64 / num_slices as f64;
                if config.urgency > 0.5 {
                    1.0 - (-3.0 * config.urgency * prev_time).exp()
                } else {
                    prev_time.powf(1.0 / (1.0 - config.urgency + 0.01))
                }
            };

            let slice_qty = quantity * (cumulative_fraction - prev_fraction);
            let target_time =
                now + Duration::from_secs((time_fraction * config.max_duration_secs as f64) as u64);

            schedule.push(Slice {
                slice_id: i,
                target_time,
                target_quantity: slice_qty,
                executed_quantity: 0.0,
                avg_price: 0.0,
                status: SliceStatus::Pending,
            });
        }

        let state = ExecutionState {
            algo_id: algo_id.clone(),
            algo_type: AlgoType::ImplementationShortfall,
            symbol,
            side,
            total_quantity: quantity,
            executed_quantity: 0.0,
            avg_executed_price: 0.0,
            start_time: now,
            end_time,
            is_complete: false,
            schedule,
            current_slice_idx: 0,
        };

        {
            let mut algos = self.active_algos.write();
            algos.insert(algo_id, state.clone());
        }

        {
            let mut stats = self.statistics.write();
            stats.total_algos_run += 1;
            stats.is_count += 1;
        }

        info!(
            "Started IS {} | Symbol: {} | Side: {:?} | Qty: {} | Arrival: {}",
            algo_id, state.symbol, side, quantity, arrival_price
        );

        state
    }

    /// Get the next slice to execute (if any)
    pub fn get_next_slice(&self, algo_id: &str) -> Option<Slice> {
        let algos = self.active_algos.read();
        let state = algos.get(algo_id)?;

        if state.is_complete {
            return None;
        }

        let now = Instant::now();

        // Find the next pending slice that's due
        for slice in &state.schedule[state.current_slice_idx..] {
            if slice.status == SliceStatus::Pending && now >= slice.target_time {
                return Some(slice.clone());
            }
        }

        None
    }

    /// Update execution with a fill
    pub fn update_fill(
        &self,
        algo_id: &str,
        slice_id: usize,
        quantity: f64,
        price: f64,
    ) -> Option<ExecutionState> {
        let mut algos = self.active_algos.write();
        let state = algos.get_mut(algo_id)?;

        // Update slice
        if slice_id < state.schedule.len() {
            let slice = &mut state.schedule[slice_id];
            slice.executed_quantity += quantity;
            
            // Update average price
            if slice.avg_price > 0.0 {
                let total_value = slice.avg_price * slice.executed_quantity
                    + price * quantity;
                slice.executed_quantity += quantity;
                slice.avg_price = total_value / slice.executed_quantity;
            } else {
                slice.avg_price = price;
            }
            
            if slice.executed_quantity >= slice.target_quantity * 0.99 {
                slice.status = SliceStatus::Completed;
            }
        }

        // Update overall state
        state.executed_quantity += quantity;
        
        // Update average executed price
        if state.avg_executed_price > 0.0 {
            let total_value = state.avg_executed_price * (state.executed_quantity - quantity)
                + price * quantity;
            state.avg_executed_price = total_value / state.executed_quantity;
        } else {
            state.avg_executed_price = price;
        }

        // Check completion
        if state.executed_quantity >= state.total_quantity * 0.99
            || Instant::now() >= state.end_time
        {
            state.is_complete = true;
            info!("Algorithm {} completed | Executed: {} / {} | Avg Price: {}", 
                  algo_id, state.executed_quantity, state.total_quantity, state.avg_executed_price);
        }

        Some(state.clone())
    }

    /// Get active algorithm state
    pub fn get_algo_state(&self, algo_id: &str) -> Option<ExecutionState> {
        let algos = self.active_algos.read();
        algos.get(algo_id).cloned()
    }

    /// Remove completed algorithm
    pub fn remove_completed(&self, algo_id: &str) -> bool {
        let mut algos = self.active_algos.write();
        if let Some(state) = algos.get(algo_id) {
            if state.is_complete {
                algos.remove(algo_id);
                return true;
            }
        }
        false
    }

    /// Get execution statistics
    pub fn get_statistics(&self) -> ExecutionStatistics {
        self.statistics.read().clone()
    }

    /// Calculate implementation shortfall (slippage)
    pub fn calculate_shortfall(
        &self,
        algo_id: &str,
        arrival_price: f64,
    ) -> Option<f64> {
        let algos = self.active_algos.read();
        let state = algos.get(algo_id)?;

        if !state.is_complete || state.avg_executed_price <= 0.0 {
            return None;
        }

        let shortfall = match state.side {
            Side::Buy => state.avg_executed_price - arrival_price,
            Side::Sell => arrival_price - state.avg_executed_price,
        };

        // Return in basis points
        Some(shortfall / arrival_price * 10000.0)
    }
}

impl Default for AlgoExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twap_creation() {
        let executor = AlgoExecutor::new();
        let config = TwapConfig {
            duration_secs: 60,
            num_slices: 10,
            randomization: 0.0,
        };

        let state = executor.start_twap(
            "test_twap".to_string(),
            "BTCUSDT".to_string(),
            Side::Buy,
            1.0,
            config,
        );

        assert_eq!(state.algo_type, AlgoType::Twap);
        assert_eq!(state.schedule.len(), 10);
        assert!(!state.is_complete);
    }

    #[test]
    fn test_execution_progress() {
        let executor = AlgoExecutor::new();
        let config = TwapConfig::default();

        let state = executor.start_twap(
            "test_progress".to_string(),
            "ETHUSDT".to_string(),
            Side::Sell,
            10.0,
            config,
        );

        assert!((state.progress() - 0.0).abs() < 0.001);
        assert!((state.time_progress() - 0.0).abs() < 0.001);
    }
}
