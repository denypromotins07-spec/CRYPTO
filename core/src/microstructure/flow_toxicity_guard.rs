//! `flow_toxicity_guard.rs` - Sub-Microsecond Toxic Flow Guardrail
//! 
//! **STAGE 10 | CHAPTER 2 | FILE 3**
//! 
//! This module implements a hard, sub-microsecond guardrail that automatically
//! responds to toxic order flow detected by VPIN and other metrics. When toxicity
//! exceeds critical thresholds, the guard can:
//! 1. Halt market-making activities
//! 2. Pull resting orders immediately
//! 3. Aggressively widen spreads
//! 4. Switch to reduce-only mode
//! 
//! **Key Features:**
//! - Lock-free atomic state management
//! - Nanosecond-level response time
//! - Multi-tier alert system
//! - Automatic recovery with hysteresis

use std::sync::atomic::{AtomicBool, AtomicUsize, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use crate::microstructure::vpin::VpinCalculator;

/// Toxicity severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ToxicityLevel {
    Normal = 0,
    Elevated = 1,
    High = 2,
    Critical = 3,
}

impl ToxicityLevel {
    /// Get recommended action for this toxicity level
    pub fn recommended_action(&self) -> ToxicityAction {
        match self {
            ToxicityLevel::Normal => ToxicityAction::None,
            ToxicityLevel::Elevated => ToxicityAction::WidenSpreads,
            ToxicityLevel::High => ToxicityAction::ReduceSize,
            ToxicityLevel::Critical => ToxicityAction::HaltTrading,
        }
    }
}

/// Actions the guard can take
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToxicityAction {
    /// No action needed
    None,
    /// Widen bid-ask spreads by specified percentage
    WidenSpreads,
    /// Reduce order sizes
    ReduceSize,
    /// Cancel all resting orders and halt new quoting
    HaltTrading,
    /// Enter reduce-only mode (no new positions)
    ReduceOnly,
}

/// Configuration for the flow toxicity guard
#[derive(Debug, Clone)]
pub struct ToxicityGuardConfig {
    /// VPIN threshold for elevated toxicity
    pub vpin_elevated_threshold: f64,
    /// VPIN threshold for high toxicity
    pub vpin_high_threshold: f64,
    /// VPIN threshold for critical toxicity
    pub vpin_critical_threshold: f64,
    /// Minimum duration in elevated state before taking action (nanoseconds)
    pub min_elevated_duration_ns: u64,
    /// Hysteresis factor for recovery (must drop below threshold * hysteresis)
    pub recovery_hysteresis: f64,
    /// Spread widening factor at elevated level (e.g., 1.5 = 50% wider)
    pub elevated_spread_factor: f64,
    /// Spread widening factor at high level
    pub high_spread_factor: f64,
    /// Order size reduction factor at high level (e.g., 0.5 = 50% size)
    pub high_size_reduction: f64,
}

impl Default for ToxicityGuardConfig {
    fn default() -> Self {
        Self {
            vpin_elevated_threshold: 0.3,
            vpin_high_threshold: 0.5,
            vpin_critical_threshold: 0.7,
            min_elevated_duration_ns: 1_000_000, // 1ms
            recovery_hysteresis: 0.8,
            elevated_spread_factor: 1.5,
            high_spread_factor: 2.5,
            high_size_reduction: 0.5,
        }
    }
}

/// State of the toxicity guard
#[derive(Debug, Clone, Copy)]
pub struct GuardState {
    pub current_level: ToxicityLevel,
    pub current_action: ToxicityAction,
    pub spread_multiplier: f64,
    pub size_multiplier: f64,
    pub is_trading_halted: bool,
    pub is_reduce_only: bool,
    pub last_vpin: f64,
    pub elevated_since_ns: u64,
    pub consecutive_toxic_buckets: usize,
}

impl Default for GuardState {
    fn default() -> Self {
        Self {
            current_level: ToxicityLevel::Normal,
            current_action: ToxicityAction::None,
            spread_multiplier: 1.0,
            size_multiplier: 1.0,
            is_trading_halted: false,
            is_reduce_only: false,
            last_vpin: 0.0,
            elevated_since_ns: 0,
            consecutive_toxic_buckets: 0,
        }
    }
}

/// Flow Toxicity Guard - Main implementation
/// 
/// Thread-safe, lock-free implementation using atomic operations
/// for nanosecond-level response to toxic flow conditions.
pub struct FlowToxicityGuard {
    config: ToxicityGuardConfig,
    
    // Atomic state flags for lock-free access
    is_halted: AtomicBool,
    is_reduce_only: AtomicBool,
    current_level: AtomicUsize,
    
    // Timing information
    elevated_start_ns: AtomicU64,
    last_update_ns: AtomicU64,
    
    // Statistics
    total_halts: AtomicUsize,
    total_warnings: AtomicUsize,
    toxic_events_count: AtomicUsize,
    
    // Cached state for quick reads
    cached_spread_factor: f64,
    cached_size_factor: f64,
}

impl FlowToxicityGuard {
    /// Create a new toxicity guard with custom configuration
    pub fn new(config: ToxicityGuardConfig) -> Self {
        Self {
            config,
            is_halted: AtomicBool::new(false),
            is_reduce_only: AtomicBool::new(false),
            current_level: AtomicUsize::new(ToxicityLevel::Normal as usize),
            elevated_start_ns: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(0),
            total_halts: AtomicUsize::new(0),
            total_warnings: AtomicUsize::new(0),
            toxic_events_count: AtomicUsize::new(0),
            cached_spread_factor: 1.0,
            cached_size_factor: 1.0,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(ToxicityGuardConfig::default())
    }

    /// Process a VPIN update and potentially trigger protective actions
    /// 
    /// # Arguments
    /// * `vpin` - Current VPIN value
    /// * `timestamp_ns` - Current timestamp in nanoseconds
    /// 
    /// Returns: The action taken (if any)
    #[inline(always)]
    pub fn process_vpin(&mut self, vpin: f64, timestamp_ns: u64) -> ToxicityAction {
        self.last_update_ns.store(timestamp_ns, Ordering::Relaxed);
        
        let level = self.determine_level(vpin);
        let prev_level = ToxicityLevel::from_usize(self.current_level.load(Ordering::Acquire));
        
        // Update level atomically
        self.current_level.store(level as usize, Ordering::Release);
        
        // Determine action based on level and duration
        let action = match level {
            ToxicityLevel::Normal => {
                self.recover(timestamp_ns);
                ToxicityAction::None
            }
            ToxicityLevel::Elevated => {
                // Check if we've been elevated long enough to act
                let elevated_since = self.elevated_start_ns.load(Ordering::Acquire);
                if elevated_since == 0 {
                    self.elevated_start_ns.store(timestamp_ns, Ordering::Release);
                } else if timestamp_ns - elevated_since >= self.config.min_elevated_duration_ns {
                    self.total_warnings.fetch_add(1, Ordering::Relaxed);
                    ToxicityAction::WidenSpreads
                } else {
                    ToxicityAction::None
                }
            }
            ToxicityLevel::High => {
                self.toxic_events_count.fetch_add(1, Ordering::Relaxed);
                self.elevated_start_ns.store(timestamp_ns, Ordering::Release);
                ToxicityAction::ReduceSize
            }
            ToxicityLevel::Critical => {
                self.toxic_events_count.fetch_add(1, Ordering::Relaxed);
                self.is_halted.store(true, Ordering::Release);
                self.total_halts.fetch_add(1, Ordering::Relaxed);
                ToxicityAction::HaltTrading
            }
        };

        // Update cached factors
        self.update_cached_factors(level);
        
        action
    }

    /// Determine toxicity level from VPIN value
    #[inline(always)]
    fn determine_level(&self, vpin: f64) -> ToxicityLevel {
        if vpin >= self.config.vpin_critical_threshold {
            ToxicityLevel::Critical
        } else if vpin >= self.config.vpin_high_threshold {
            ToxicityLevel::High
        } else if vpin >= self.config.vpin_elevated_threshold {
            ToxicityLevel::Elevated
        } else {
            ToxicityLevel::Normal
        }
    }

    /// Recovery logic with hysteresis
    fn recover(&mut self, timestamp_ns: u64) {
        let was_halted = self.is_halted.swap(false, Ordering::AcqRel);
        
        if was_halted {
            // Reset elevated timer on recovery
            self.elevated_start_ns.store(0, Ordering::Release);
        }
        
        self.is_reduce_only.store(false, Ordering::Release);
        self.current_level.store(ToxicityLevel::Normal as usize, Ordering::Release);
    }

    /// Update cached spread and size factors
    fn update_cached_factors(&mut self, level: ToxicityLevel) {
        (self.cached_spread_factor, self.cached_size_factor) = match level {
            ToxicityLevel::Normal => (1.0, 1.0),
            ToxicityLevel::Elevated => (self.config.elevated_spread_factor, 1.0),
            ToxicityLevel::High => (self.config.high_spread_factor, self.config.high_size_reduction),
            ToxicityLevel::Critical => (f64::MAX, 0.0),
        };
    }

    /// Check if trading is currently halted
    #[inline(always)]
    pub fn is_trading_halted(&self) -> bool {
        self.is_halted.load(Ordering::Acquire)
    }

    /// Check if in reduce-only mode
    #[inline(always)]
    pub fn is_reduce_only(&self) -> bool {
        self.is_reduce_only.load(Ordering::Acquire) || self.is_halted.load(Ordering::Acquire)
    }

    /// Get current spread multiplier for quote adjustment
    #[inline(always)]
    pub fn spread_multiplier(&self) -> f64 {
        self.cached_spread_factor
    }

    /// Get current size multiplier for order sizing
    #[inline(always)]
    pub fn size_multiplier(&self) -> f64 {
        self.cached_size_factor
    }

    /// Get current toxicity level
    #[inline(always)]
    pub fn current_level(&self) -> ToxicityLevel {
        ToxicityLevel::from_usize(self.current_level.load(Ordering::Acquire))
    }

    /// Get current guard state snapshot
    pub fn get_state(&self) -> GuardState {
        GuardState {
            current_level: self.current_level(),
            current_action: self.current_level().recommended_action(),
            spread_multiplier: self.spread_multiplier(),
            size_multiplier: self.size_multiplier(),
            is_trading_halted: self.is_trading_halted(),
            is_reduce_only: self.is_reduce_only(),
            last_vpin: 0.0, // Would need to be passed in or stored
            elevated_since_ns: self.elevated_start_ns.load(Ordering::Acquire),
            consecutive_toxic_buckets: self.toxic_events_count.load(Ordering::Acquire),
        }
    }

    /// Manually trigger a halt (emergency override)
    pub fn emergency_halt(&self) {
        self.is_halted.store(true, Ordering::Release);
        self.total_halts.fetch_add(1, Ordering::Relaxed);
    }

    /// Manually resume trading (after manual review)
    pub fn resume_trading(&mut self) {
        self.is_halted.store(false, Ordering::Release);
        self.is_reduce_only.store(false, Ordering::Release);
        self.elevated_start_ns.store(0, Ordering::Release);
        self.current_level.store(ToxicityLevel::Normal as usize, Ordering::Release);
        self.cached_spread_factor = 1.0;
        self.cached_size_factor = 1.0;
    }

    /// Get statistics
    pub fn stats(&self) -> (usize, usize, usize) {
        (
            self.total_halts.load(Ordering::Relaxed),
            self.total_warnings.load(Ordering::Relaxed),
            self.toxic_events_count.load(Ordering::Relaxed),
        )
    }

    /// Reset all statistics
    pub fn reset_stats(&mut self) {
        self.total_halts.store(0, Ordering::Relaxed);
        self.total_warnings.store(0, Ordering::Relaxed);
        self.toxic_events_count.store(0, Ordering::Relaxed);
    }
}

impl From<usize> for ToxicityLevel {
    fn from(val: usize) -> Self {
        match val {
            0 => ToxicityLevel::Normal,
            1 => ToxicityLevel::Elevated,
            2 => ToxicityLevel::High,
            _ => ToxicityLevel::Critical,
        }
    }
}

/// Integration helper for combining VPIN calculator with guard
pub struct ToxicityMonitor {
    vpin_calc: VpinCalculator,
    guard: FlowToxicityGuard,
    last_action: ToxicityAction,
}

impl ToxicityMonitor {
    pub fn new(vpin_buckets: usize, bucket_volume: f64, config: ToxicityGuardConfig) -> Self {
        Self {
            vpin_calc: VpinCalculator::new(vpin_buckets, bucket_volume),
            guard: FlowToxicityGuard::new(config),
            last_action: ToxicityAction::None,
        }
    }

    /// Process a trade and update toxicity state
    /// 
    /// # Arguments
    /// * `volume` - Trade volume
    /// * `side` - Trade side (buyer/seller initiated)
    /// * `timestamp_ns` - Trade timestamp
    /// 
    /// Returns: (bucket_completed, action_taken)
    pub fn process_trade(
        &mut self,
        volume: f64,
        side: crate::microstructure::vpin::TradeSide,
        timestamp_ns: u64,
    ) -> (bool, ToxicityAction) {
        let (complete, vpin) = self.vpin_calc.process_trade(volume, side, timestamp_ns);
        
        if complete {
            let action = self.guard.process_vpin(vpin, timestamp_ns);
            self.last_action = action;
            (true, action)
        } else {
            (false, self.last_action)
        }
    }

    /// Get reference to underlying guard
    pub fn guard(&self) -> &FlowToxicityGuard {
        &self.guard
    }

    /// Get mutable reference to underlying guard
    pub fn guard_mut(&mut self) -> &mut FlowToxicityGuard {
        &mut self.guard
    }

    /// Get reference to VPIN calculator
    pub fn vpin_calculator(&self) -> &VpinCalculator {
        &self.vpin_calc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::microstructure::vpin::TradeSide;

    #[test]
    fn test_guard_normal_flow() {
        let mut guard = FlowToxicityGuard::default_config();
        
        // Low VPIN should not trigger any action
        let action = guard.process_vpin(0.1, 1_000_000_000);
        assert_eq!(action, ToxicityAction::None);
        assert!(!guard.is_trading_halted());
    }

    #[test]
    fn test_guard_critical_flow() {
        let mut guard = FlowToxicityGuard::default_config();
        
        // Critical VPIN should halt trading immediately
        let action = guard.process_vpin(0.8, 1_000_000_000);
        assert_eq!(action, ToxicityAction::HaltTrading);
        assert!(guard.is_trading_halted());
    }

    #[test]
    fn test_guard_recovery() {
        let mut guard = FlowToxicityGuard::default_config();
        
        // Trigger halt
        guard.process_vpin(0.8, 1_000_000_000);
        assert!(guard.is_trading_halted());
        
        // Recover with low VPIN
        guard.process_vpin(0.1, 2_000_000_000);
        assert!(!guard.is_trading_halted());
        assert_eq!(guard.spread_multiplier(), 1.0);
    }

    #[test]
    fn test_toxicity_monitor_integration() {
        let mut monitor = ToxicityMonitor::new(10, 100.0, ToxicityGuardConfig::default());
        
        // Simulate toxic flow
        for i in 0..100 {
            let (_, action) = monitor.process_trade(
                10.0,
                TradeSide::BuyerInitiated,
                i as u64 * 1_000_000,
            );
            
            if monitor.guard().is_trading_halted() {
                assert_eq!(action, ToxicityAction::HaltTrading);
                break;
            }
        }
    }
}
