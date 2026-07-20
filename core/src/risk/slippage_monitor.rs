// core/src/risk/slippage_monitor.rs
// =============================================================================
// SLIPPAGE MONITOR & TRANSACTION COST ANALYSIS (TCA)
// =============================================================================
// Purpose: Real-time Transaction Cost Analysis and slippage modeling.
// Calculates expected edge vs expected slippage and automatically rejects
// or re-routes orders if the risk/reward ratio falls below threshold.
//
// Features:
// - Real-time slippage tracking per order
// - Market impact estimation using volume participation
// - Expected vs actual fill price analysis
// - Automatic order rejection based on cost/edge ratio

use std::collections::HashMap;

/// Configuration for slippage monitoring
#[derive(Debug, Clone)]
pub struct SlippageConfig {
    /// Maximum acceptable slippage as % of price
    pub max_slippage_pct: f64,
    /// Maximum market impact as % of daily volume
    pub max_volume_participation: f64,
    /// Minimum edge/cost ratio to allow trade
    pub min_edge_cost_ratio: f64,
    /// Lookback window for slippage statistics (number of trades)
    pub lookback_trades: usize,
}

impl Default for SlippageConfig {
    fn default() -> Self {
        Self {
            max_slippage_pct: 0.001,          // 10 bps max slippage
            max_volume_participation: 0.05,   // Max 5% of daily volume
            min_edge_cost_ratio: 2.0,         // Edge must be 2x cost
            lookback_trades: 50,
        }
    }
}

/// Result of a slippage analysis
#[derive(Debug, Clone)]
pub struct SlippageAnalysis {
    /// Expected slippage in bps
    pub expected_slippage_bps: f64,
    /// Estimated market impact in bps
    pub market_impact_bps: f64,
    /// Total estimated cost (slippage + fees + impact)
    pub total_cost_bps: f64,
    /// Expected edge from strategy signal
    pub expected_edge_bps: f64,
    /// Edge to cost ratio
    pub edge_cost_ratio: f64,
    /// Whether order should be allowed
    pub allowed: bool,
    /// Reason if rejected
    pub rejection_reason: Option<&'static str>,
}

/// Historical slippage record
#[derive(Clone)]
struct SlippageRecord {
    symbol: String,
    side: bool, // true = buy, false = sell
    expected_price: f64,
    fill_price: f64,
    quantity: f64,
    volume_participation: f64,
}

/// Real-time slippage monitor
pub struct SlippageMonitor {
    config: SlippageConfig,
    /// Historical slippage records per symbol
    history: HashMap<String, Vec<SlippageRecord>>,
    /// Running average slippage per symbol
    avg_slippage_bps: HashMap<String, f64>,
    /// Fee structure (bps)
    maker_fee_bps: f64,
    taker_fee_bps: f64,
}

impl SlippageMonitor {
    pub fn new(config: SlippageConfig, maker_fee_bps: f64, taker_fee_bps: f64) -> Self {
        Self {
            config,
            history: HashMap::new(),
            avg_slippage_bps: HashMap::new(),
            maker_fee_bps,
            taker_fee_bps,
        }
    }

    /// Record a completed fill for TCA analysis
    pub fn record_fill(&mut self, symbol: &str, side: bool, expected_price: f64,
                       fill_price: f64, quantity: f64, daily_volume: f64) {
        let volume_participation = if daily_volume > 0.0 {
            quantity / daily_volume
        } else {
            0.0
        };

        let record = SlippageRecord {
            symbol: symbol.to_string(),
            side,
            expected_price,
            fill_price,
            quantity,
            volume_participation,
        };

        // Add to history
        let history = self.history.entry(symbol.to_string()).or_insert_with(Vec::new);
        if history.len() >= self.config.lookback_trades {
            history.remove(0);
        }
        history.push(record);

        // Update running average
        self.update_average_slippage(symbol);
    }

    /// Update average slippage for a symbol
    fn update_average_slippage(&mut self, symbol: &str) {
        if let Some(history) = self.history.get(symbol) {
            if history.is_empty() {
                return;
            }

            let mut total_slippage_bps = 0.0;
            for record in history {
                let slippage = if record.side {
                    // Buy: positive slippage means paid more than expected
                    (record.fill_price - record.expected_price) / record.expected_price
                } else {
                    // Sell: positive slippage means received less than expected
                    (record.expected_price - record.fill_price) / record.expected_price
                };
                total_slippage_bps += slippage * 10_000.0; // Convert to bps
            }

            let avg = total_slippage_bps / history.len() as f64;
            self.avg_slippage_bps.insert(symbol.to_string(), avg);
        }
    }

    /// Analyze proposed order for slippage and cost
    pub fn analyze_order(&self, symbol: &str, side: bool, quantity: f64,
                         current_price: f64, daily_volume: f64,
                         expected_edge_bps: f64) -> SlippageAnalysis {
        // Get historical slippage for this symbol
        let hist_slippage = self.avg_slippage_bps.get(symbol).copied().unwrap_or(5.0); // Default 5 bps

        // Calculate volume participation
        let volume_participation = if daily_volume > 0.0 {
            quantity / daily_volume
        } else {
            1.0 // Assume 100% if no volume data
        };

        // Estimate market impact using square-root law
        // Impact ≈ spread/2 + alpha * sqrt(participation)
        let spread_estimate = 10.0; // Assume 10 bps spread default
        let impact_alpha = 50.0; // Calibration factor
        let market_impact_bps = spread_estimate / 2.0 + impact_alpha * volume_participation.sqrt();

        // Expected slippage is historical + impact
        let expected_slippage_bps = hist_slippage + market_impact_bps;

        // Determine fee based on order type assumption (taker for market orders)
        let fee_bps = self.taker_fee_bps;

        // Total cost
        let total_cost_bps = expected_slippage_bps + fee_bps;

        // Edge to cost ratio
        let edge_cost_ratio = if total_cost_bps > 0.0 {
            expected_edge_bps / total_cost_bps
        } else {
            f64::INFINITY
        };

        // Check if order should be allowed
        let mut allowed = true;
        let mut rejection_reason = None;

        // Check slippage limit
        if expected_slippage_bps / 10_000.0 > self.config.max_slippage_pct {
            allowed = false;
            rejection_reason = Some("expected_slippage_exceeds_limit");
        }

        // Check volume participation
        if volume_participation > self.config.max_volume_participation {
            allowed = false;
            rejection_reason = Some("volume_participation_too_high");
        }

        // Check edge/cost ratio
        if edge_cost_ratio < self.config.min_edge_cost_ratio {
            allowed = false;
            rejection_reason = Some("insufficient_edge_vs_cost");
        }

        SlippageAnalysis {
            expected_slippage_bps,
            market_impact_bps,
            total_cost_bps,
            expected_edge_bps,
            edge_cost_ratio,
            allowed,
            rejection_reason,
        }
    }

    /// Get optimal order type recommendation
    pub fn recommend_order_type(&self, symbol: &str, quantity: f64, 
                                 daily_volume: f64) -> &'static str {
        let participation = if daily_volume > 0.0 {
            quantity / daily_volume
        } else {
            1.0
        };

        // Decision tree for order type
        if participation < 0.001 {
            // Very small order: market order acceptable
            "MARKET"
        } else if participation < 0.01 {
            // Small order: limit order at mid
            "LIMIT_MID"
        } else if participation < 0.05 {
            // Medium order: passive limit with time
            "LIMIT_PASSIVE"
        } else {
            // Large order: use TWAP/Iceberg
            "ALGO_TWAP"
        }
    }

    /// Get recent slippage statistics for a symbol
    pub fn get_slippage_stats(&self, symbol: &str) -> Option<(f64, f64, usize)> {
        self.history.get(symbol).map(|records| {
            if records.is_empty() {
                return (0.0, 0.0, 0);
            }

            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for record in records {
                let slippage = if record.side {
                    (record.fill_price - record.expected_price) / record.expected_price * 10_000.0
                } else {
                    (record.expected_price - record.fill_price) / record.expected_price * 10_000.0
                };
                sum += slippage;
                sum_sq += slippage * slippage;
            }

            let n = records.len() as f64;
            let mean = sum / n;
            let variance = (sum_sq / n) - (mean * mean);
            let std_dev = variance.sqrt();

            (mean, std_dev, records.len())
        })
    }

    /// Clear old history (memory management)
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.avg_slippage_bps.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slippage_analysis() {
        let config = SlippageConfig::default();
        let mut monitor = SlippageMonitor::new(config, 2.5, 5.0); // 2.5 bps maker, 5 bps taker

        // Record some historical fills
        monitor.record_fill("BTC/USDT", true, 50_000.0, 50_020.0, 0.1, 100_000.0);
        monitor.record_fill("BTC/USDT", true, 50_000.0, 50_015.0, 0.1, 100_000.0);
        monitor.record_fill("BTC/USDT", false, 50_000.0, 49_985.0, 0.1, 100_000.0);

        // Analyze a new order
        let analysis = monitor.analyze_order("BTC/USDT", true, 0.5, 50_000.0, 100_000.0, 30.0);

        assert!(analysis.expected_slippage_bps > 0.0);
        assert!(analysis.total_cost_bps > 0.0);
        
        // With 30 bps edge and ~10-15 bps cost, ratio should pass
        if analysis.edge_cost_ratio >= 2.0 {
            assert!(analysis.allowed);
        }
    }

    #[test]
    fn test_large_order_rejection() {
        let config = SlippageConfig::default();
        let monitor = SlippageMonitor::new(config, 2.5, 5.0);

        // Large order relative to volume
        let analysis = monitor.analyze_order("BTC/USDT", true, 10_000.0, 50_000.0, 100_000.0, 30.0);

        // Should reject due to high volume participation
        assert!(!analysis.allowed);
        assert_eq!(analysis.rejection_reason, Some("volume_participation_too_high"));
    }
}
