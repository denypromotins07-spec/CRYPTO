// core/src/risk/drawdown_control.rs
// =============================================================================
// DRAWDOWN CONTROL & CIRCUIT BREAKERS
// =============================================================================
// Purpose: Hard-coded, ultra-fast circuit breakers that enforce strict limits
// on daily loss, maximum drawdown, gross exposure, and single-asset concentration.
// Must execute in sub-microsecond time to prevent catastrophic loss.
//
// Features:
// - Daily loss limit with auto-trading halt
// - Maximum drawdown from peak equity
// - Gross exposure limits (long + short)
// - Per-asset concentration limits
// - Lock-free state checks for hot path

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Configuration for drawdown controls
#[derive(Debug, Clone)]
pub struct DrawdownConfig {
    /// Maximum daily loss as % of starting equity
    pub max_daily_loss_pct: f64,
    /// Maximum drawdown from peak as % of peak
    pub max_drawdown_pct: f64,
    /// Maximum gross exposure (sum of all positions) as % of equity
    pub max_gross_exposure_pct: f64,
    /// Maximum single asset exposure as % of equity
    pub max_single_asset_pct: f64,
    /// Cooldown period after circuit breaker trip (seconds)
    pub cooldown_seconds: u64,
}

impl Default for DrawdownConfig {
    fn default() -> Self {
        Self {
            max_daily_loss_pct: 0.03,      // 3% daily loss limit
            max_drawdown_pct: 0.10,         // 10% max drawdown
            max_gross_exposure_pct: 2.0,    // 200% gross (with leverage)
            max_single_asset_pct: 0.25,     // 25% max in one asset
            cooldown_seconds: 3600,         // 1 hour cooldown
        }
    }
}

/// State of circuit breaker
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    Active,
    Tripped,
    Cooldown,
}

/// Result of risk check
#[derive(Debug, Clone)]
pub struct RiskCheckResult {
    pub allowed: bool,
    pub circuit_state: CircuitState,
    pub violation_reason: Option<&'static str>,
    pub current_metrics: RiskMetrics,
}

/// Current risk metrics
#[derive(Debug, Clone)]
pub struct RiskMetrics {
    pub daily_pnl: f64,
    pub daily_pnl_pct: f64,
    pub peak_equity: f64,
    pub current_equity: f64,
    pub drawdown_pct: f64,
    pub gross_exposure: f64,
    pub gross_exposure_pct: f64,
}

/// Ultra-fast drawdown controller using atomic operations
pub struct DrawdownController {
    config: DrawdownConfig,
    
    // Atomic state for lock-free reads in hot path
    /// Starting equity at beginning of day (stored as u64 fixed point)
    daily_start_equity: AtomicU64,
    /// Peak equity ever reached (fixed point)
    peak_equity: AtomicU64,
    /// Current equity (fixed point)
    current_equity: AtomicU64,
    /// Current gross exposure (fixed point)
    gross_exposure: AtomicU64,
    /// Circuit breaker state flag
    circuit_tripped: AtomicBool,
    /// Timestamp when circuit was tripped
    trip_timestamp: AtomicU64,
    
    // Non-atomic for less frequent updates
    max_single_asset_exposure: f64,
}

// Fixed point conversion: multiply by 1e9 for nanodollars
const FIXED_POINT_SCALE: u64 = 1_000_000_000;

fn to_fixed(value: f64) -> u64 {
    (value * FIXED_POINT_SCALE as f64) as u64
}

fn from_fixed(value: u64) -> f64 {
    value as f64 / FIXED_POINT_SCALE as f64
}

impl DrawdownController {
    pub fn new(config: DrawdownConfig, initial_equity: f64) -> Self {
        let equity_fixed = to_fixed(initial_equity);
        
        Self {
            config,
            daily_start_equity: AtomicU64::new(equity_fixed),
            peak_equity: AtomicU64::new(equity_fixed),
            current_equity: AtomicU64::new(equity_fixed),
            gross_exposure: AtomicU64::new(0),
            circuit_tripped: AtomicBool::new(false),
            trip_timestamp: AtomicU64::new(0),
            max_single_asset_exposure: 0.0,
        }
    }

    /// Update current equity - called on every PnL change
    #[inline]
    pub fn update_equity(&self, equity: f64) {
        self.current_equity.store(to_fixed(equity), Ordering::Relaxed);
        
        // Update peak if higher
        let equity_fixed = to_fixed(equity);
        let mut peak = self.peak_equity.load(Ordering::Relaxed);
        while equity_fixed > peak {
            match self.peak_equity.compare_exchange_weak(
                peak, equity_fixed, Ordering::Relaxed, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }

    /// Update gross exposure
    #[inline]
    pub fn update_gross_exposure(&self, exposure: f64) {
        self.gross_exposure.store(to_fixed(exposure), Ordering::Relaxed);
    }

    /// Update max single asset exposure
    pub fn update_single_asset_exposure(&mut self, exposure_pct: f64) {
        self.max_single_asset_exposure = exposure_pct;
    }

    /// Reset daily PnL at start of trading day
    pub fn reset_daily(&self, new_start_equity: f64) {
        self.daily_start_equity.store(to_fixed(new_start_equity), Ordering::Relaxed);
    }

    /// CRITICAL: Fast-path risk check for pre-order validation
    /// This must be extremely fast - no allocations, minimal branching
    #[inline]
    pub fn check_pre_order(&self, order_notional: f64, current_equity: f64) -> RiskCheckResult {
        let equity = current_equity;
        let peak = from_fixed(self.peak_equity.load(Ordering::Relaxed));
        let start = from_fixed(self.daily_start_equity.load(Ordering::Relaxed));
        let gross = from_fixed(self.gross_exposure.load(Ordering::Relaxed));
        
        // Calculate current metrics
        let daily_pnl = equity - start;
        let daily_pnl_pct = if start > 0.0 { daily_pnl / start } else { 0.0 };
        
        let drawdown = if peak > 0.0 { (peak - equity) / peak } else { 0.0 };
        
        let new_gross = gross + order_notional;
        let new_gross_pct = new_gross / equity;
        
        // Check circuit breaker first (fastest path)
        if self.circuit_tripped.load(Ordering::Acquire) {
            let trip_ts = self.trip_timestamp.load(Ordering::Relaxed);
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            
            if now - trip_ts >= self.config.cooldown_seconds {
                // Cooldown expired, allow reset attempt
                return RiskCheckResult {
                    allowed: false,
                    circuit_state: CircuitState::Cooldown,
                    violation_reason: Some("circuit_breaker_cooldown"),
                    current_metrics: RiskMetrics {
                        daily_pnl,
                        daily_pnl_pct,
                        peak_equity: peak,
                        current_equity: equity,
                        drawdown_pct: drawdown,
                        gross_exposure: gross,
                        gross_exposure_pct: gross / equity,
                    },
                };
            }
            
            return RiskCheckResult {
                allowed: false,
                circuit_state: CircuitState::Tripped,
                violation_reason: Some("circuit_breaker_active"),
                current_metrics: RiskMetrics {
                    daily_pnl,
                    daily_pnl_pct,
                    peak_equity: peak,
                    current_equity: equity,
                    drawdown_pct: drawdown,
                    gross_exposure: gross,
                    gross_exposure_pct: gross / equity,
                },
            };
        }
        
        // Check daily loss limit
        if daily_pnl_pct < -self.config.max_daily_loss_pct {
            self.trigger_circuit();
            return RiskCheckResult {
                allowed: false,
                circuit_state: CircuitState::Tripped,
                violation_reason: Some("daily_loss_limit"),
                current_metrics: RiskMetrics {
                    daily_pnl,
                    daily_pnl_pct,
                    peak_equity: peak,
                    current_equity: equity,
                    drawdown_pct: drawdown,
                    gross_exposure: gross,
                    gross_exposure_pct: gross / equity,
                },
            };
        }
        
        // Check max drawdown
        if drawdown > self.config.max_drawdown_pct {
            self.trigger_circuit();
            return RiskCheckResult {
                allowed: false,
                circuit_state: CircuitState::Tripped,
                violation_reason: Some("max_drawdown"),
                current_metrics: RiskMetrics {
                    daily_pnl,
                    daily_pnl_pct,
                    peak_equity: peak,
                    current_equity: equity,
                    drawdown_pct: drawdown,
                    gross_exposure: gross,
                    gross_exposure_pct: gross / equity,
                },
            };
        }
        
        // Check gross exposure
        if new_gross_pct > self.config.max_gross_exposure_pct {
            return RiskCheckResult {
                allowed: false,
                circuit_state: CircuitState::Active,
                violation_reason: Some("gross_exposure_limit"),
                current_metrics: RiskMetrics {
                    daily_pnl,
                    daily_pnl_pct,
                    peak_equity: peak,
                    current_equity: equity,
                    drawdown_pct: drawdown,
                    gross_exposure: gross,
                    gross_exposure_pct: gross / equity,
                },
            };
        }
        
        // Check single asset concentration
        if self.max_single_asset_exposure > self.config.max_single_asset_pct {
            return RiskCheckResult {
                allowed: false,
                circuit_state: CircuitState::Active,
                violation_reason: Some("single_asset_concentration"),
                current_metrics: RiskMetrics {
                    daily_pnl,
                    daily_pnl_pct,
                    peak_equity: peak,
                    current_equity: equity,
                    drawdown_pct: drawdown,
                    gross_exposure: gross,
                    gross_exposure_pct: gross / equity,
                },
            };
        }
        
        // All checks passed
        RiskCheckResult {
            allowed: true,
            circuit_state: CircuitState::Active,
            violation_reason: None,
            current_metrics: RiskMetrics {
                daily_pnl,
                daily_pnl_pct,
                peak_equity: peak,
                current_equity: equity,
                drawdown_pct: drawdown,
                gross_exposure: gross,
                gross_exposure_pct: gross / equity,
            },
        }
    }

    /// Trigger the circuit breaker
    fn trigger_circuit(&self) {
        self.circuit_tripped.store(true, Ordering::Release);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.trip_timestamp.store(now, Ordering::Relaxed);
    }

    /// Manually reset circuit breaker (after review)
    pub fn reset_circuit(&self) {
        self.circuit_tripped.store(false, Ordering::Release);
        self.trip_timestamp.store(0, Ordering::Relaxed);
    }

    /// Get current circuit state
    #[inline]
    pub fn get_circuit_state(&self) -> CircuitState {
        if !self.circuit_tripped.load(Ordering::Acquire) {
            CircuitState::Active
        } else {
            let trip_ts = self.trip_timestamp.load(Ordering::Relaxed);
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            
            if now - trip_ts >= self.config.cooldown_seconds {
                CircuitState::Cooldown
            } else {
                CircuitState::Tripped
            }
        }
    }

    /// Get current drawdown percentage
    #[inline]
    pub fn current_drawdown_pct(&self) -> f64 {
        let equity = from_fixed(self.current_equity.load(Ordering::Relaxed));
        let peak = from_fixed(self.peak_equity.load(Ordering::Relaxed));
        if peak > 0.0 {
            (peak - equity) / peak
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daily_loss_limit() {
        let config = DrawdownConfig::default();
        let controller = DrawdownController::new(config, 100_000.0);
        
        // Simulate 4% daily loss
        controller.update_equity(96_000.0);
        
        let result = controller.check_pre_order(1_000.0, 96_000.0);
        assert!(!result.allowed);
        assert_eq!(result.violation_reason, Some("daily_loss_limit"));
    }

    #[test]
    fn test_gross_exposure_limit() {
        let config = DrawdownConfig::default();
        let controller = DrawdownController::new(config, 100_000.0);
        
        // Set existing gross exposure
        controller.update_gross_exposure(180_000.0);
        
        // Try to add 50k more (would exceed 200% limit)
        let result = controller.check_pre_order(50_000.0, 100_000.0);
        assert!(!result.allowed);
        assert_eq!(result.violation_reason, Some("gross_exposure_limit"));
    }
}
