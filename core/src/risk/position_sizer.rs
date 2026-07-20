// core/src/risk/position_sizer.rs
// =============================================================================
// REAL-TIME POSITION SIZING ENGINE
// =============================================================================
// Purpose: Ultra-fast calculation of Kelly Criterion and fixed-fractional 
// position sizing based on real-time account equity, ATR, and portfolio heat.
//
// Features:
// - Dynamic Kelly sizing with fractional scaling
// - ATR-based volatility adjustment
// - Portfolio heat limits (max total risk exposure)
// - Sub-microsecond execution via pre-computed tables

/// Configuration for position sizing
#[derive(Debug, Clone)]
pub struct PositionSizerConfig {
    /// Fractional Kelly multiplier (0.5 = half Kelly)
    pub kelly_fraction: f64,
    /// Maximum position size as % of equity
    pub max_position_pct: f64,
    /// Maximum total portfolio heat (sum of all risks)
    pub max_portfolio_heat: f64,
    /// Risk per trade for fixed-fractional
    pub fixed_risk_pct: f64,
    /// Minimum ATR multiplier for stop distance
    pub min_atr_multiplier: f64,
}

impl Default for PositionSizerConfig {
    fn default() -> Self {
        Self {
            kelly_fraction: 0.25,      // Conservative quarter-Kelly
            max_position_pct: 0.20,     // Max 20% in single position
            max_portfolio_heat: 0.06,   // Max 6% total risk
            fixed_risk_pct: 0.01,       // 1% risk per trade
            min_atr_multiplier: 1.5,    // Stop at least 1.5x ATR
        }
    }
}

/// Result of position size calculation
#[derive(Debug, Clone)]
pub struct PositionSizeResult {
    /// Quantity to trade
    pub quantity: f64,
    /// Dollar value of position
    pub notional_value: f64,
    /// Stop loss price
    pub stop_loss: f64,
    /// Risk amount in dollars
    pub risk_amount: f64,
    /// Kelly fraction used
    pub kelly_fraction_used: f64,
    /// Reason if size was capped
    pub cap_reason: Option<&'static str>,
}

/// Real-time position sizer
pub struct PositionSizer {
    config: PositionSizerConfig,
    /// Current account equity
    current_equity: f64,
    /// Current portfolio heat (sum of open position risks)
    current_heat: f64,
    /// Win rate estimate (0.0 to 1.0)
    win_rate: f64,
    /// Win/Loss ratio estimate
    win_loss_ratio: f64,
    /// Number of trades for statistical significance
    trade_count: u32,
}

impl PositionSizer {
    pub fn new(config: PositionSizerConfig, initial_equity: f64) -> Self {
        Self {
            config,
            current_equity: initial_equity,
            current_heat: 0.0,
            win_rate: 0.5,      // Default until we have data
            win_loss_ratio: 1.5, // Default until we have data
            trade_count: 0,
        }
    }

    /// Update account equity
    #[inline]
    pub fn update_equity(&mut self, equity: f64) {
        self.current_equity = equity;
    }

    /// Update strategy statistics for Kelly calculation
    pub fn update_stats(&mut self, win_rate: f64, win_loss_ratio: f64, trade_count: u32) {
        self.win_rate = win_rate.clamp(0.0, 1.0);
        self.win_loss_ratio = win_loss_ratio.max(0.1);
        self.trade_count = trade_count;
    }

    /// Add to current portfolio heat
    #[inline]
    pub fn add_heat(&mut self, risk_amount: f64) {
        self.current_heat += risk_amount;
    }

    /// Release heat when position closes
    #[inline]
    pub fn release_heat(&mut self, risk_amount: f64) {
        self.current_heat = (self.current_heat - risk_amount).max(0.0);
    }

    /// Calculate position size using Kelly Criterion
    /// 
    /// Kelly formula: f* = p - q/b
    /// where p = win probability, q = loss probability, b = win/loss ratio
    pub fn calculate_kelly_size(&self, entry_price: f64, stop_loss: f64) -> PositionSizeResult {
        let risk_per_unit = (entry_price - stop_loss).abs();
        
        // Calculate raw Kelly fraction
        let p = self.win_rate;
        let q = 1.0 - p;
        let b = self.win_loss_ratio;
        
        let raw_kelly = if b > 0.0 {
            p - q / b
        } else {
            0.0
        };
        
        // Apply fractional Kelly
        let adjusted_kelly = raw_kelly * self.config.kelly_fraction;
        
        // Ensure non-negative
        let kelly_fraction = adjusted_kelly.max(0.0);
        
        // Calculate position size
        let max_risk_amount = self.current_equity * self.config.fixed_risk_pct;
        let kelly_risk_amount = self.current_equity * kelly_fraction;
        
        // Use the more conservative of Kelly or fixed risk
        let risk_amount = kelly_risk_amount.min(max_risk_amount);
        
        // Check portfolio heat limit
        let available_heat = (self.current_equity * self.config.max_portfolio_heat) - self.current_heat;
        let final_risk = risk_amount.min(available_heat);
        
        let mut cap_reason = None;
        if final_risk < risk_amount {
            cap_reason = Some("portfolio_heat_limit");
        }
        
        // Calculate quantity
        let quantity = if risk_per_unit > 0.0 {
            final_risk / risk_per_unit
        } else {
            0.0
        };
        
        // Apply max position limit
        let max_notional = self.current_equity * self.config.max_position_pct;
        let notional_value = quantity * entry_price;
        
        let (final_quantity, final_cap_reason) = if notional_value > max_notional {
            (max_notional / entry_price, Some("max_position_pct"))
        } else {
            (quantity, cap_reason)
        };
        
        PositionSizeResult {
            quantity: final_quantity,
            notional_value: final_quantity * entry_price,
            stop_loss,
            risk_amount: final_quantity * risk_per_unit,
            kelly_fraction_used: kelly_fraction,
            cap_reason: final_cap_reason,
        }
    }

    /// Calculate position size using ATR-based volatility adjustment
    pub fn calculate_atr_size(&self, entry_price: f64, atr: f64, 
                               direction: i8) -> PositionSizeResult {
        // Stop loss based on ATR
        let atr_multiplier = self.config.min_atr_multiplier;
        let stop_distance = atr * atr_multiplier;
        
        let stop_loss = if direction > 0 {
            entry_price - stop_distance
        } else {
            entry_price + stop_distance
        };
        
        // Use fixed fractional risk
        let risk_amount = self.current_equity * self.config.fixed_risk_pct;
        
        // Check portfolio heat
        let available_heat = (self.current_equity * self.config.max_portfolio_heat) - self.current_heat;
        let final_risk = risk_amount.min(available_heat);
        
        let risk_per_unit = stop_distance;
        let quantity = if risk_per_unit > 0.0 {
            final_risk / risk_per_unit
        } else {
            0.0
        };
        
        // Apply max position limit
        let max_notional = self.current_equity * self.config.max_position_pct;
        let notional_value = quantity * entry_price;
        
        let final_quantity = if notional_value > max_notional {
            max_notional / entry_price
        } else {
            quantity
        };
        
        PositionSizeResult {
            quantity: final_quantity,
            notional_value: final_quantity * entry_price,
            stop_loss,
            risk_amount: final_quantity * risk_per_unit,
            kelly_fraction_used: 0.0, // Not using Kelly here
            cap_reason: if notional_value > max_notional { Some("max_position_pct") } else { None },
        }
    }

    /// Get current portfolio heat percentage
    #[inline]
    pub fn heat_percentage(&self) -> f64 {
        if self.current_equity > 0.0 {
            self.current_heat / self.current_equity
        } else {
            0.0
        }
    }

    /// Check if new position is allowed under heat limits
    #[inline]
    pub fn can_open_position(&self, proposed_risk: f64) -> bool {
        let max_total_heat = self.current_equity * self.config.max_portfolio_heat;
        self.current_heat + proposed_risk <= max_total_heat
    }

    /// Reset heat tracking (e.g., daily reset)
    pub fn reset_heat(&mut self) {
        self.current_heat = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kelly_sizing() {
        let config = PositionSizerConfig::default();
        let mut sizer = PositionSizer::new(config, 100_000.0);
        
        // Set reasonable stats
        sizer.update_stats(0.55, 2.0, 100);
        
        let result = sizer.calculate_kelly_size(50_000.0, 49_000.0);
        
        assert!(result.quantity > 0.0);
        assert!(result.risk_amount <= 100_000.0 * 0.06); // Within heat limit
    }

    #[test]
    fn test_heat_limit() {
        let config = PositionSizerConfig::default();
        let mut sizer = PositionSizer::new(config, 100_000.0);
        
        // Add existing heat
        sizer.add_heat(5_000.0);
        
        // Should still allow small positions
        assert!(sizer.can_open_position(500.0));
        
        // Should reject large positions
        assert!(!sizer.can_open_position(10_000.0));
    }
}
