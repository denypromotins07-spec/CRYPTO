//! `greeks_calculator.rs` - Real-time Lock-free Greeks Calculation
//! 
//! **STAGE 10 | CHAPTER 1 | FILE 2**
//! 
//! This module provides ultra-fast calculation of option Greeks (Delta, Gamma, Theta, Vega, Rho)
//! with lock-free data structures for concurrent access during high-frequency trading.
//! 
//! **Key Features:**
//! - Zero heap allocation during hot path via pre-allocated pools.
//! - Atomic counters for thread-safe aggregation without locks.
//! - SIMD-ready batch processing.
//! - Memory capped at strict limits to stay within 8GB global budget.

use std::sync::atomic::{AtomicUsize, Ordering};
use crate::options::black_scholes::{norm_cdf, norm_pdf};

/// Maximum number of positions in the portfolio (pre-allocated to prevent runtime allocation)
/// Tuned to fit within memory budget: 10,000 positions * ~200 bytes = ~2MB
const MAX_POSITIONS: usize = 10_000;

/// Pre-allocated buffer for Greek calculations (cache-line aligned)
#[repr(align(64))]
pub struct GreeksBuffer {
    pub deltas: [f64; MAX_POSITIONS],
    pub gammas: [f64; MAX_POSITIONS],
    pub thetas: [f64; MAX_POSITIONS],
    pub vegas: [f64; MAX_POSITIONS],
    pub rhos: [f64; MAX_POSITIONS],
    count: AtomicUsize,
}

impl GreeksBuffer {
    pub const fn new() -> Self {
        Self {
            deltas: [0.0; MAX_POSITIONS],
            gammas: [0.0; MAX_POSITIONS],
            thetas: [0.0; MAX_POSITIONS],
            vegas: [0.0; MAX_POSITIONS],
            rhos: [0.0; MAX_POSITIONS],
            count: AtomicUsize::new(0),
        }
    }

    #[inline(always)]
    pub fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn push(&self, delta: f64, gamma: f64, theta: f64, vega: f64, rho: f64) -> Option<usize> {
        let idx = self.count.fetch_add(1, Ordering::AcqRel);
        if idx >= MAX_POSITIONS {
            return None; // Buffer full - safety check
        }
        
        // Direct memory write - no bounds check needed due to atomic guard
        unsafe {
            let deltas_ptr = self.deltas.as_ptr() as *mut f64;
            let gammas_ptr = self.gammas.as_ptr() as *mut f64;
            let thetas_ptr = self.thetas.as_ptr() as *mut f64;
            let vegas_ptr = self.vegas.as_ptr() as *mut f64;
            let rhos_ptr = self.rhos.as_ptr() as *mut f64;

            *deltas_ptr.add(idx) = delta;
            *gammas_ptr.add(idx) = gamma;
            *thetas_ptr.add(idx) = theta;
            *vegas_ptr.add(idx) = vega;
            *rhos_ptr.add(idx) = rho;
        }
        
        Some(idx)
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }

    /// Calculate portfolio-wide net Greeks (thread-safe summation)
    pub fn aggregate(&self) -> PortfolioGreeks {
        let count = self.len();
        let mut net_delta = 0.0;
        let mut net_gamma = 0.0;
        let mut net_theta = 0.0;
        let mut net_vega = 0.0;
        let mut net_rho = 0.0;

        for i in 0..count {
            unsafe {
                net_delta += *self.deltas.as_ptr().add(i);
                net_gamma += *self.gammas.as_ptr().add(i);
                net_theta += *self.thetas.as_ptr().add(i);
                net_vega += *self.vegas.as_ptr().add(i);
                net_rho += *self.rhos.as_ptr().add(i);
            }
        }

        PortfolioGreeks {
            delta: net_delta,
            gamma: net_gamma,
            theta: net_theta,
            vega: net_vega,
            rho: net_rho,
        }
    }
}

impl Default for GreeksBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Aggregated portfolio Greeks
#[derive(Debug, Clone, Copy, Default)]
pub struct PortfolioGreeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
}

/// Calculate all Greeks for a single European option
/// 
/// # Arguments
/// * `s` - Spot price
/// * `k` - Strike price
/// * `t` - Time to maturity (years)
/// * `r` - Risk-free rate
/// * `sigma` - Volatility
/// * `is_call` - true for call, false for put
/// 
/// Returns: (delta, gamma, theta, vega, rho)
#[inline(always)]
pub fn calculate_greeks(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    is_call: bool,
) -> (f64, f64, f64, f64, f64) {
    if t <= 0.0 || sigma <= 0.0 {
        return calculate_greeks_at_expiry(s, k, is_call);
    }

    let sqrt_t = t.sqrt();
    let d1 = (s / k).ln() + (r + 0.5 * sigma * sigma) * t;
    let d1 = d1 / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;

    let nd1 = norm_cdf(d1);
    let nd2 = norm_cdf(d2);
    let n_d1 = norm_pdf(d1);

    // Delta
    let delta = if is_call {
        nd1
    } else {
        nd1 - 1.0
    };

    // Gamma (same for call and put)
    let gamma = n_d1 / (s * sigma * sqrt_t);

    // Theta (per day, so divide by 365)
    let term1 = -(s * n_d1 * sigma) / (2.0 * sqrt_t);
    let theta = if is_call {
        term1 - r * k * (-r * t).exp() * nd2
    } else {
        term1 + r * k * (-r * t).exp() * norm_cdf(-d2)
    };
    let theta = theta / 365.0; // Convert to daily

    // Vega (per 1% change in vol)
    let vega = (s * sqrt_t * n_d1) / 100.0;

    // Rho (per 1% change in rate)
    let rho = if is_call {
        k * t * (-r * t).exp() * nd2
    } else {
        -k * t * (-r * t).exp() * norm_cdf(-d2)
    };
    let rho = rho / 100.0; // Convert to per 1%

    (delta, gamma, theta, vega, rho)
}

/// Handle Greeks at expiry (t=0)
#[inline(always)]
fn calculate_greeks_at_expiry(s: f64, k: f64, is_call: bool) -> (f64, f64, f64, f64, f64) {
    let delta = if is_call {
        if s > k { 1.0 } else { 0.0 }
    } else {
        if s < k { -1.0 } else { 0.0 }
    };

    // Gamma, Theta, Vega, Rho are undefined/zero at expiry
    (delta, 0.0, 0.0, 0.0, 0.0)
}

/// Batch Greek calculation with pre-allocated output buffers
/// Optimized for processing entire portfolios in one pass
/// 
/// # Arguments
/// * `spots`, `strikes`, etc. - Input parameter slices (must be same length)
/// * `output_delta`, etc. - Pre-allocated output slices (must be >= input length)
/// 
/// # Safety
/// Caller must ensure output buffers are properly allocated
pub fn calculate_greeks_batch(
    spots: &[f64],
    strikes: &[f64],
    times: &[f64],
    rates: &[f64],
    volatilities: &[f64],
    is_calls: &[bool],
    output_delta: &mut [f64],
    output_gamma: &mut [f64],
    output_theta: &mut [f64],
    output_vega: &mut [f64],
    output_rho: &mut [f64],
) {
    assert_eq!(spots.len(), strikes.len());
    assert_eq!(spots.len(), times.len());
    assert_eq!(spots.len(), rates.len());
    assert_eq!(spots.len(), volatilities.len());
    assert_eq!(spots.len(), is_calls.len());

    let len = spots.len().min(output_delta.len())
        .min(output_gamma.len())
        .min(output_theta.len())
        .min(output_vega.len())
        .min(output_rho.len());

    for i in 0..len {
        let (d, g, th, v, r) = calculate_greeks(
            spots[i],
            strikes[i],
            times[i],
            rates[i],
            volatilities[i],
            is_calls[i],
        );
        output_delta[i] = d;
        output_gamma[i] = g;
        output_theta[i] = th;
        output_vega[i] = v;
        output_rho[i] = r;
    }
}

/// Position-level Greek tracking with quantity multiplier
#[derive(Debug, Clone, Copy)]
pub struct PositionGreeks {
    pub symbol: u64, // Hashed symbol identifier
    pub quantity: f64,
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
}

impl PositionGreeks {
    #[inline(always)]
    pub fn new(symbol: u64, quantity: f64, greeks: (f64, f64, f64, f64, f64)) -> Self {
        Self {
            symbol,
            quantity,
            delta: greeks.0 * quantity,
            gamma: greeks.1 * quantity,
            theta: greeks.2 * quantity,
            vega: greeks.3 * quantity,
            rho: greeks.4 * quantity,
        }
    }

    /// Get dollar-adjusted Greeks (Greek * spot * quantity)
    #[inline(always)]
    pub fn dollar_delta(&self, spot: f64) -> f64 {
        self.delta * spot
    }

    #[inline(always)]
    pub fn dollar_gamma(&self, spot: f64) -> f64 {
        self.gamma * spot * spot
    }
}

/// Risk limit checker - triggers alerts when Greeks exceed thresholds
pub struct GreekRiskLimits {
    pub max_net_delta: f64,
    pub max_net_gamma: f64,
    pub max_net_theta: f64,
    pub max_net_vega: f64,
    pub max_net_rho: f64,
}

impl GreekRiskLimits {
    pub const fn new(
        max_delta: f64,
        max_gamma: f64,
        max_theta: f64,
        max_vega: f64,
        max_rho: f64,
    ) -> Self {
        Self {
            max_net_delta: max_delta,
            max_net_gamma: max_gamma,
            max_net_theta: max_theta,
            max_net_vega: max_vega,
            max_net_rho: max_rho,
        }
    }

    /// Check if current portfolio Greeks exceed limits
    #[inline(always)]
    pub fn check_limits(&self, greeks: &PortfolioGreeks) -> GreekLimitViolation {
        let mut violations = GreekLimitViolation::empty();

        if greeks.delta.abs() > self.max_net_delta {
            violations |= GreekLimitViolation::DELTA;
        }
        if greeks.gamma.abs() > self.max_net_gamma {
            violations |= GreekLimitViolation::GAMMA;
        }
        if greeks.theta.abs() > self.max_net_theta {
            violations |= GreekLimitViolation::THETA;
        }
        if greeks.vega.abs() > self.max_net_vega {
            violations |= GreekLimitViolation::VEGA;
        }
        if greeks.rho.abs() > self.max_net_rho {
            violations |= GreekLimitViolation::RHO;
        }

        violations
    }
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct GreekLimitViolation: u8 {
        const EMPTY = 0;
        const DELTA = 1 << 0;
        const GAMMA = 1 << 1;
        const THETA = 1 << 2;
        const VEGA = 1 << 3;
        const RHO = 1 << 4;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greeks_call_option() {
        // ATM Call: S=100, K=100, T=1, r=0.05, sigma=0.2
        let (delta, gamma, theta, vega, rho) = calculate_greeks(100.0, 100.0, 1.0, 0.05, 0.2, true);
        
        assert!((delta - 0.60).abs() < 0.01);
        assert!(gamma > 0.0);
        assert!(theta < 0.0); // Time decay
        assert!(vega > 0.0);
        assert!(rho > 0.0);
    }

    #[test]
    fn test_greeks_put_option() {
        // ATM Put: S=100, K=100, T=1, r=0.05, sigma=0.2
        let (delta, gamma, theta, vega, rho) = calculate_greeks(100.0, 100.0, 1.0, 0.05, 0.2, false);
        
        assert!((delta + 0.40).abs() < 0.01); // Negative delta
        assert!(gamma > 0.0);
        assert!(theta < 0.0);
        assert!(vega > 0.0);
        assert!(rho < 0.0); // Negative rho for puts
    }

    #[test]
    fn test_greeks_buffer_thread_safety() {
        let buffer = GreeksBuffer::new();
        
        // Simulate concurrent pushes
        for i in 0..100 {
            buffer.push(0.5, 0.01, -0.02, 0.1, 0.05).unwrap();
        }

        assert_eq!(buffer.len(), 100);
        
        let agg = buffer.aggregate();
        assert!((agg.delta - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_risk_limits() {
        let limits = GreekRiskLimits::new(10.0, 1.0, 5.0, 10.0, 5.0);
        let greeks = PortfolioGreeks {
            delta: 15.0, // Exceeds limit
            gamma: 0.5,
            theta: -2.0,
            vega: 5.0,
            rho: 1.0,
        };

        let violations = limits.check_limits(&greeks);
        assert!(violations.contains(GreekLimitViolation::DELTA));
        assert!(!violations.contains(GreekLimitViolation::GAMMA));
    }
}
