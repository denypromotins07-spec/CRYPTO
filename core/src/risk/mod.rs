// core/src/risk/mod.rs
// =============================================================================
// REAL-TIME RISK MANAGEMENT MODULE
// =============================================================================
// Aggregates position sizing, drawdown control, and slippage monitoring
// into a unified ultra-fast risk guardrail system.

pub mod position_sizer;
pub mod drawdown_control;
pub mod slippage_monitor;

pub use position_sizer::{PositionSizer, PositionSizerConfig, PositionSizeResult};
pub use drawdown_control::{DrawdownController, DrawdownConfig, RiskCheckResult, CircuitState, RiskMetrics};
pub use slippage_monitor::{SlippageMonitor, SlippageConfig, SlippageAnalysis};

/// Unified risk management engine
pub struct RiskEngine {
    pub position_sizer: PositionSizer,
    pub drawdown_controller: DrawdownController,
    pub slippage_monitor: SlippageMonitor,
}

impl RiskEngine {
    pub fn new(initial_equity: f64) -> Self {
        Self {
            position_sizer: PositionSizer::new(PositionSizerConfig::default(), initial_equity),
            drawdown_controller: DrawdownController::new(DrawdownConfig::default(), initial_equity),
            slippage_monitor: SlippageMonitor::new(SlippageConfig::default(), 2.5, 5.0),
        }
    }

    /// Pre-trade risk check - combines all risk controls
    #[inline]
    pub fn pre_trade_check(&self, order_notional: f64, current_equity: f64) -> RiskCheckResult {
        self.drawdown_controller.check_pre_order(order_notional, current_equity)
    }

    /// Calculate approved position size with all risk constraints
    pub fn calculate_position(&mut self, entry_price: f64, stop_loss: f64, 
                               current_equity: f64) -> PositionSizeResult {
        // First check if trade is allowed at all
        let risk_per_unit = (entry_price - stop_loss).abs();
        let preliminary_size = self.position_sizer.calculate_kelly_size(entry_price, stop_loss);
        
        // Verify against drawdown limits
        let check = self.drawdown_controller.check_pre_order(
            preliminary_size.notional_value, 
            current_equity
        );
        
        if !check.allowed {
            return PositionSizeResult {
                quantity: 0.0,
                notional_value: 0.0,
                stop_loss,
                risk_amount: 0.0,
                kelly_fraction_used: 0.0,
                cap_reason: check.violation_reason,
            };
        }
        
        preliminary_size
    }

    /// Record a fill for TCA analysis
    pub fn record_fill(&mut self, symbol: &str, side: bool, expected_price: f64,
                       fill_price: f64, quantity: f64, daily_volume: f64) {
        self.slippage_monitor.record_fill(symbol, side, expected_price, fill_price, quantity, daily_volume);
    }

    /// Analyze slippage for proposed order
    pub fn analyze_slippage(&self, symbol: &str, side: bool, quantity: f64,
                            price: f64, daily_volume: f64, edge_bps: f64) -> SlippageAnalysis {
        self.slippage_monitor.analyze_order(symbol, side, quantity, price, daily_volume, edge_bps)
    }

    /// Update equity across all risk components
    pub fn update_equity(&mut self, equity: f64) {
        self.position_sizer.update_equity(equity);
        self.drawdown_controller.update_equity(equity);
    }

    /// Update gross exposure
    pub fn update_gross_exposure(&self, exposure: f64) {
        self.drawdown_controller.update_gross_exposure(exposure);
    }

    /// Get current risk metrics summary
    pub fn get_risk_summary(&self) -> RiskSummary {
        let dd_metrics = self.drawdown_controller.check_pre_order(0.0, 
            crate::risk::from_fixed(self.drawdown_controller.current_equity.load(std::sync::atomic::Ordering::Relaxed)));
        
        RiskSummary {
            current_equity: dd_metrics.current_metrics.current_equity,
            peak_equity: dd_metrics.current_metrics.peak_equity,
            daily_pnl_pct: dd_metrics.current_metrics.daily_pnl_pct,
            drawdown_pct: dd_metrics.current_metrics.drawdown_pct,
            gross_exposure_pct: dd_metrics.current_metrics.gross_exposure_pct,
            heat_pct: self.position_sizer.heat_percentage(),
            circuit_state: dd_metrics.circuit_state,
        }
    }

    /// Reset daily limits (called at start of trading day)
    pub fn reset_daily(&self, new_start_equity: f64) {
        self.drawdown_controller.reset_daily(new_start_equity);
        self.position_sizer.reset_heat();
    }
}

/// Summary of current risk state
#[derive(Debug, Clone)]
pub struct RiskSummary {
    pub current_equity: f64,
    pub peak_equity: f64,
    pub daily_pnl_pct: f64,
    pub drawdown_pct: f64,
    pub gross_exposure_pct: f64,
    pub heat_pct: f64,
    pub circuit_state: CircuitState,
}

// Re-export from_fixed for external use
pub use drawdown_control::from_fixed;
