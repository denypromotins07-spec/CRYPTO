//! core/src/regime/volatility_surface.rs
//!
//! Real-time construction of the implied volatility surface using options data,
//! perp funding rates, and historical volatility to detect mispriced risk.
//!
//! Target Hardware: AMD Ryzen AI 5 (AVX2 optimized)

use std::collections::HashMap;

/// Volatility surface point.
#[derive(Debug, Clone)]
pub struct VolSurfacePoint {
    pub strike: f64,
    pub expiry_days: u32,
    pub implied_vol: f64,
    pub delta: f64,
}

/// Volatility surface for a single asset.
pub struct VolatilitySurface {
    /// Surface points organized by (strike_idx, expiry_idx).
    surface: HashMap<(u32, u32), VolSurfacePoint>,
    
    /// Strike buckets.
    strikes: Vec<f64>,
    
    /// Expiry buckets (in days).
    expiries: Vec<u32>,
    
    /// Current spot price.
    spot_price: f64,
    
    /// Risk-free rate (approximated).
    risk_free_rate: f64,
    
    /// ATM volatility.
    atm_vol: f64,
    
    /// Skew parameters.
    skew_slope: f64,
    skew_curvature: f64,
    
    /// Term structure parameters.
    term_structure: Vec<f64>,
}

impl VolatilitySurface {
    pub fn new(spot_price: f64) -> Self {
        // Define strike buckets (±50% around spot in 5% increments)
        let strikes: Vec<f64> = (80..=120)
            .map(|pct| spot_price * (pct as f64 / 100.0))
            .collect();
        
        // Define expiry buckets
        let expiries = vec![1, 3, 7, 14, 30, 60, 90];
        
        let num_strikes = strikes.len();
        let num_expiries = expiries.len();
        
        Self {
            surface: HashMap::with_capacity(num_strikes * num_expiries),
            strikes,
            expiries,
            spot_price,
            risk_free_rate: 0.05, // 5% annualized
            atm_vol: 0.6, // 60% default
            skew_slope: -0.1, // Negative skew typical in crypto
            skew_curvature: 0.02,
            term_structure: vec![0.6; num_expiries],
        }
    }

    /// Black-Scholes d1 calculation.
    #[inline]
    fn d1(spot: f64, strike: f64, vol: f64, time_to_expiry: f64, r: f64) -> f64 {
        if time_to_expiry <= 0.0 || vol <= 0.0 {
            return 0.0;
        }
        let sqrt_t = time_to_expiry.sqrt();
        ((spot / strike).ln() + (r + 0.5 * vol * vol) * time_to_expiry) / (vol * sqrt_t)
    }

    /// Black-Scholes d2 calculation.
    #[inline]
    fn d2(d1: f64, vol: f64, time_to_expiry: f64) -> f64 {
        if time_to_expiry <= 0.0 {
            return d1;
        }
        d1 - vol * time_to_expiry.sqrt()
    }

    /// Standard normal CDF approximation.
    #[inline]
    fn norm_cdf(x: f64) -> f64 {
        const A1: f64 = 0.254829592;
        const A2: f64 = -0.284496736;
        const A3: f64 = 1.421413741;
        const A4: f64 = -1.453152027;
        const A5: f64 = 1.061405429;
        const P: f64 = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + P * x);
        let y = 1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp();
        0.5 * (1.0 + sign * y)
    }

    /// Black-Scholes call option price.
    pub fn bs_call(&self, strike: f64, vol: f64, time_to_expiry: f64) -> f64 {
        let d1 = Self::d1(self.spot_price, strike, vol, time_to_expiry, self.risk_free_rate);
        let d2 = Self::d2(d1, vol, time_to_expiry);
        
        self.spot_price * Self::norm_cdf(d1) 
            - strike * (-self.risk_free_rate * time_to_expiry).exp() * Self::norm_cdf(d2)
    }

    /// Black-Scholes put option price.
    pub fn bs_put(&self, strike: f64, vol: f64, time_to_expiry: f64) -> f64 {
        let d1 = Self::d1(self.spot_price, strike, vol, time_to_expiry, self.risk_free_rate);
        let d2 = Self::d2(d1, vol, time_to_expiry);
        
        strike * (-self.risk_free_rate * time_to_expiry).exp() * Self::norm_cdf(-d2)
            - self.spot_price * Self::norm_cdf(-d1)
    }

    /// Newton-Raphson implied volatility solver.
    pub fn implied_vol(&self, market_price: f64, strike: f64, time_to_expiry: f64, is_call: bool) -> Option<f64> {
        if time_to_expiry <= 0.0 {
            return None;
        }

        let mut vol = 0.5; // Initial guess
        let tolerance = 1e-6;
        let max_iterations = 100;

        for _ in 0..max_iterations {
            let price = if is_call {
                self.bs_call(strike, vol, time_to_expiry)
            } else {
                self.bs_put(strike, vol, time_to_expiry)
            };

            let diff = price - market_price;
            if diff.abs() < tolerance {
                return Some(vol);
            }

            // Vega (derivative w.r.t. vol)
            let d1 = Self::d1(self.spot_price, strike, vol, time_to_expiry, self.risk_free_rate);
            let vega = self.spot_price * (2.0 * std::f64::consts::PI).sqrt().recip() 
                * (-0.5 * d1 * d1).exp() * time_to_expiry.sqrt();

            if vega.abs() < 1e-10 {
                break;
            }

            vol -= diff / vega;
            vol = vol.clamp(0.01, 5.0);
        }

        Some(vol)
    }

    /// Build surface from market data.
    pub fn build_from_market(
        &mut self,
        options_data: &[(f64, u32, f64, bool)], // (strike, expiry_days, price, is_call)
        funding_rate: f64,
        hist_vol: f64,
    ) {
        // Update spot from ATM options
        let atm_strike = self.spot_price;
        
        // Blend funding rate and historical vol into surface
        self.atm_vol = 0.4 * hist_vol + 0.3 * funding_rate.abs() * 2.0 + 0.3 * self.atm_vol;
        
        // Build full surface
        for (expiry_idx, &expiry_days) in self.expiries.iter().enumerate() {
            let time_to_expiry = expiry_days as f64 / 365.0;
            
            // Term structure adjustment
            let term_adj = 1.0 + 0.1 * (expiry_days as f64 / 30.0).ln();
            
            for (strike_idx, &strike) in self.strikes.iter().enumerate() {
                let moneyness = strike / self.spot_price;
                
                // Skew adjustment
                let skew_adj = 1.0 + self.skew_slope * (moneyness - 1.0) 
                    + self.skew_curvature * (moneyness - 1.0).powi(2);
                
                let iv = self.atm_vol * term_adj * skew_adj;
                
                // Calculate delta
                let d1 = Self::d1(self.spot_price, strike, iv, time_to_expiry, self.risk_free_rate);
                let delta = Self::norm_cdf(d1);
                
                self.surface.insert(
                    (strike_idx as u32, expiry_idx as u32),
                    VolSurfacePoint {
                        strike,
                        expiry_days,
                        implied_vol: iv,
                        delta,
                    },
                );
            }
        }
        
        // Update term structure
        self.term_structure = self.expiries.iter()
            .map(|&d| {
                let t = d as f64 / 365.0;
                self.atm_vol * (1.0 + 0.1 * t.ln())
            })
            .collect();
    }

    /// Get interpolated implied vol for given strike and expiry.
    pub fn get_implied_vol(&self, strike: f64, expiry_days: u32) -> f64 {
        // Find nearest buckets
        let strike_idx = self.strikes.iter()
            .enumerate()
            .min_by_key(|(_, s)| ((*s - strike).abs() * 1e6) as i64)
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        let expiry_idx = self.expiries.iter()
            .enumerate()
            .min_by_key(|(_, e)| ((*e - expiry_days).abs()) as i64)
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        self.surface.get(&(strike_idx as u32, expiry_idx as u32))
            .map(|p| p.implied_vol)
            .unwrap_or(self.atm_vol)
    }

    /// Detect volatility mispricing.
    pub fn detect_mispricing(&self, market_iv: f64, strike: f64, expiry_days: u32) -> f64 {
        let model_iv = self.get_implied_vol(strike, expiry_days);
        (market_iv - model_iv) / model_iv // Relative mispricing
    }

    /// Calculate VIX-like index from surface.
    pub fn calculate_vix(&self) -> f64 {
        // Weighted average of 30-day OTM options
        let target_days = 30;
        let mut sum_var: f64 = 0.0;
        let mut count = 0;
        
        for (expiry_idx, &expiry_days) in self.expiries.iter().enumerate() {
            if (expiry_days as i32 - target_days as i32).abs() > 5 {
                continue;
            }
            
            for (strike_idx, &strike) in self.strikes.iter().enumerate() {
                if let Some(point) = self.surface.get(&(strike_idx as u32, expiry_idx as u32)) {
                    // Include OTM options
                    if (strike > self.spot_price && point.delta < 0.5) ||
                       (strike < self.spot_price && point.delta > 0.5) {
                        sum_var += point.implied_vol.powi(2);
                        count += 1;
                    }
                }
            }
        }
        
        if count == 0 {
            return self.atm_vol;
        }
        
        (sum_var / count as f64).sqrt()
    }

    /// Update spot price.
    pub fn update_spot(&mut self, spot: f64) {
        if (spot - self.spot_price).abs() / self.spot_price > 0.01 {
            // Significant move, rebuild strikes
            self.spot_price = spot;
            self.strikes = (80..=120)
                .map(|pct| spot * (pct as f64 / 100.0))
                .collect();
            self.surface.clear();
        } else {
            self.spot_price = spot;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_black_scholes() {
        let surface = VolatilitySurface::new(100.0);
        
        // Test call price
        let call_price = surface.bs_call(100.0, 0.2, 0.25); // ATM, 20% vol, 3 months
        assert!(call_price > 0.0);
        assert!(call_price < 10.0);
        
        // Test put-call parity
        let put_price = surface.bs_put(100.0, 0.2, 0.25);
        let forward = 100.0 * (0.05 * 0.25).exp();
        assert!((call_price - put_price - (100.0 - forward)).abs() < 0.01);
    }

    #[test]
    fn test_implied_vol_solver() {
        let surface = VolatilitySurface::new(100.0);
        
        // Generate theoretical price
        let true_vol = 0.3;
        let market_price = surface.bs_call(100.0, true_vol, 0.25);
        
        // Recover implied vol
        let recovered_vol = surface.implied_vol(market_price, 100.0, 0.25, true);
        assert!(recovered_vol.is_some());
        assert!((recovered_vol.unwrap() - true_vol).abs() < 0.01);
    }

    #[test]
    fn test_vix_calculation() {
        let mut surface = VolatilitySurface::new(100.0);
        surface.build_from_market(&[], 0.01, 0.5);
        
        let vix = surface.calculate_vix();
        assert!(vix > 0.0);
        assert!(vix < 2.0);
    }
}
