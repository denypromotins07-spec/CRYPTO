//! Transaction Cost Analysis (TCA) Simulator
//! 
//! Models order book depth, market impact, slippage, and smart order routing
//! to calculate realistic net PnL for high-frequency trading strategies.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::engine::{Price, Quantity, Side, OrderBook, TimestampNs};

// ============================================================================
// Market Impact Models
// ============================================================================

/// Market impact model types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ImpactModel {
    /// Linear impact: cost proportional to trade size
    Linear,
    /// Square-root impact: cost proportional to sqrt(trade size / volume)
    SquareRoot,
    /// Almgren-Chriss model with temporary and permanent impact
    AlmgrenChriss,
    /// Log-linear impact for large trades
    LogLinear,
}

impl Default for ImpactModel {
    fn default() -> Self {
        Self::SquareRoot
    }
}

/// Parameters for market impact calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactParameters {
    /// Model type
    pub model: ImpactModel,
    /// Temporary impact coefficient (basis points)
    pub temporary_impact_bps: f64,
    /// Permanent impact coefficient (basis points)
    pub permanent_impact_bps: f64,
    /// Daily volume reference (for normalization)
    pub daily_volume_ref: f64,
    /// Volatility adjustment factor
    pub volatility_factor: f64,
    /// Spread sensitivity
    pub spread_sensitivity: f64,
}

impl Default for ImpactParameters {
    fn default() -> Self {
        Self {
            model: ImpactModel::SquareRoot,
            temporary_impact_bps: 5.0,    // 5 bps temporary impact
            permanent_impact_bps: 2.0,    // 2 bps permanent impact
            daily_volume_ref: 1e9,        // $1B daily volume reference
            volatility_factor: 1.0,
            spread_sensitivity: 0.5,
        }
    }
}

impl ImpactParameters {
    /// Calculate market impact in basis points
    pub fn calculate_impact_bps(
        &self,
        trade_value_usd: f64,
        spread_bps: f64,
        volatility: f64,
    ) -> f64 {
        let vol_adj = self.volatility_factor * (volatility / 0.02); // Normalize to 2% vol
        let participation_rate = trade_value_usd / self.daily_volume_ref;

        match self.model {
            ImpactModel::Linear => {
                // Linear model: impact = k * participation_rate
                self.temporary_impact_bps * participation_rate * vol_adj
            }
            ImpactModel::SquareRoot => {
                // Square-root model: impact = k * sqrt(participation_rate)
                // This is the most commonly used model in practice
                self.temporary_impact_bps * participation_rate.sqrt() * vol_adj
                    + self.spread_sensitivity * spread_bps
            }
            ImpactModel::AlmgrenChriss => {
                // Almgren-Chriss model with both temporary and permanent components
                let temp = self.temporary_impact_bps * participation_rate.sqrt();
                let perm = self.permanent_impact_bps * participation_rate;
                (temp + perm) * vol_adj + self.spread_sensitivity * spread_bps
            }
            ImpactModel::LogLinear => {
                // Log-linear for very large trades
                if participation_rate > 0.01 {
                    // Large trade (>1% of daily volume)
                    self.temporary_impact_bps * (1.0 + participation_rate.ln()) * vol_adj
                } else {
                    self.temporary_impact_bps * participation_rate * vol_adj
                }
            }
        }
    }

    /// Calculate expected slippage for a trade
    pub fn calculate_slippage(
        &self,
        trade_value_usd: f64,
        spread_bps: f64,
        volatility: f64,
    ) -> SlippageEstimate {
        let impact_bps = self.calculate_impact_bps(trade_value_usd, spread_bps, volatility);
        
        // Half-spread cost (assuming we cross half the spread on average)
        let spread_cost_bps = spread_bps / 2.0;
        
        // Total expected cost
        let total_cost_bps = spread_cost_bps + impact_bps;
        
        SlippageEstimate {
            spread_cost_bps,
            impact_cost_bps: impact_bps,
            total_cost_bps,
            confidence_95: total_cost_bps * 1.645, // Assuming normal distribution
            confidence_99: total_cost_bps * 2.326,
        }
    }
}

/// Slippage estimation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageEstimate {
    /// Cost from bid-ask spread (bps)
    pub spread_cost_bps: f64,
    /// Cost from market impact (bps)
    pub impact_cost_bps: f64,
    /// Total expected cost (bps)
    pub total_cost_bps: f64,
    /// 95% confidence upper bound (bps)
    pub confidence_95: f64,
    /// 99% confidence upper bound (bps)
    pub confidence_99: f64,
}

// ============================================================================
// Order Book Depth Analysis
// ============================================================================

/// Order book depth at various price levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthProfile {
    /// Price levels from mid price
    pub levels: Vec<DepthLevel>,
    /// Total liquidity within 1% of mid
    pub liquidity_1pct: Quantity,
    /// Total liquidity within 5% of mid
    pub liquidity_5pct: Quantity,
    /// Imbalance ratio (bid/ask)
    pub imbalance_ratio: f64,
}

/// Single depth level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthLevel {
    pub offset_bps: f64,
    pub bid_quantity: Quantity,
    pub ask_quantity: Quantity,
    pub cumulative_bid: Quantity,
    pub cumulative_ask: Quantity,
}

impl DepthProfile {
    /// Analyze order book depth
    pub fn analyze(book: &OrderBook, mid_price: Price) -> Self {
        let mut levels = Vec::new();
        let mut liquidity_1pct = 0u64;
        let mut liquidity_5pct = 0u64;
        let mut total_bid = 0u64;
        let mut total_ask = 0u64;

        // Analyze bids
        let one_pct = (mid_price as f64 * 0.01) as Price;
        let five_pct = (mid_price as f64 * 0.05) as Price;

        for (_, level) in book.bids.levels.iter().rev() {
            let offset = ((mid_price - level.price) as f64 / mid_price as f64) * 10000.0; // in bps
            
            total_bid += level.quantity;
            
            if (level.price as f64 - mid_price as f64).abs() <= one_pct as f64 {
                liquidity_1pct += level.quantity;
            }
            if (level.price as f64 - mid_price as f64).abs() <= five_pct as f64 {
                liquidity_5pct += level.quantity;
            }

            levels.push(DepthLevel {
                offset_bps: offset,
                bid_quantity: level.quantity,
                ask_quantity: 0,
                cumulative_bid: total_bid,
                cumulative_ask: 0,
            });
        }

        // Analyze asks
        for (_, level) in book.asks.levels.iter() {
            let offset = ((level.price - mid_price) as f64 / mid_price as f64) * 10000.0;
            
            total_ask += level.quantity;
            
            if (level.price as f64 - mid_price as f64).abs() <= one_pct as f64 {
                liquidity_1pct += level.quantity;
            }
            if (level.price as f64 - mid_price as f64).abs() <= five_pct as f64 {
                liquidity_5pct += level.quantity;
            }

            // Update existing level or add new
            let offset_idx = levels.iter().position(|l| (l.offset_bps - offset).abs() < 1.0);
            if let Some(idx) = offset_idx {
                levels[idx].ask_quantity = level.quantity;
                levels[idx].cumulative_ask = total_ask;
            } else {
                levels.push(DepthLevel {
                    offset_bps: offset,
                    bid_quantity: 0,
                    ask_quantity: level.quantity,
                    cumulative_bid: 0,
                    cumulative_ask: total_ask,
                });
            }
        }

        // Sort by offset
        levels.sort_by(|a, b| a.offset_bps.partial_cmp(&b.offset_bps).unwrap());

        let imbalance_ratio = if total_ask == 0 {
            f64::INFINITY
        } else {
            total_bid as f64 / total_ask as f64
        };

        Self {
            levels,
            liquidity_1pct,
            liquidity_5pct,
            imbalance_ratio,
        }
    }

    /// Get executable quantity at maximum slippage
    pub fn get_executable_quantity(&self, side: Side, max_slippage_bps: f64, mid_price: Price) -> Quantity {
        let mut total = 0u64;

        for level in &self.levels {
            if level.offset_bps > max_slippage_bps {
                break;
            }

            let qty = match side {
                Side::Buy => level.ask_quantity,
                Side::Sell => level.bid_quantity,
            };
            total += qty;
        }

        total
    }
}

// ============================================================================
// Smart Order Router
// ============================================================================

/// Smart order routing decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// Target venue/exchange
    pub venue: String,
    /// Order quantity
    pub quantity: Quantity,
    /// Expected price
    pub expected_price: Price,
    /// Expected slippage (bps)
    pub expected_slippage_bps: f64,
    /// Priority (lower = higher priority)
    pub priority: u32,
}

/// Smart Order Router configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Available venues
    pub venues: Vec<VenueInfo>,
    /// Minimum quantity per child order
    pub min_child_quantity: Quantity,
    /// Maximum number of child orders
    pub max_child_orders: usize,
    /// Prefer maker rebates
    pub prefer_maker: bool,
    /// Dark pool participation
    pub use_dark_pools: bool,
}

/// Venue information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueInfo {
    pub name: String,
    pub maker_fee_bps: f64,
    pub taker_fee_bps: f64,
    pub avg_spread_bps: f64,
    pub avg_daily_volume: f64,
    pub latency_ms: u64,
    pub reliability_score: f64, // 0.0 to 1.0
}

/// Smart Order Router implementation
pub struct SmartOrderRouter {
    config: RouterConfig,
    impact_params: ImpactParameters,
    orders_routed: AtomicU64,
    total_value_routed: AtomicU64,
}

impl SmartOrderRouter {
    pub fn new(config: RouterConfig, impact_params: ImpactParameters) -> Self {
        Self {
            config,
            impact_params,
            orders_routed: AtomicU64::new(0),
            total_value_routed: AtomicU64::new(0),
        }
    }

    /// Split large order into optimal child orders across venues
    pub fn route_order(
        &self,
        symbol: &str,
        side: Side,
        total_quantity: Quantity,
        current_price: Price,
        order_books: &HashMap<String, OrderBook>,
    ) -> Vec<RoutingDecision> {
        let mut decisions = Vec::new();
        let mut remaining_qty = total_quantity;

        // Score each venue
        let mut venue_scores: Vec<(String, f64)> = self.config.venues
            .iter()
            .map(|v| {
                let score = self.score_venue(v, side, current_price);
                (v.name.clone(), score)
            })
            .collect();

        // Sort by score (higher = better)
        venue_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Allocate quantity to venues
        for (venue_name, _score) in venue_scores {
            if remaining_qty == 0 {
                break;
            }

            // Get venue info
            let venue = self.config.venues.iter().find(|v| v.name == venue_name).unwrap();
            
            // Calculate optimal allocation based on volume and fees
            let volume_share = venue.avg_daily_volume 
                / self.config.venues.iter().map(|v| v.avg_daily_volume).sum::<f64>();
            
            let allocated_qty = (total_quantity as f64 * volume_share) as Quantity;
            let actual_qty = allocated_qty.min(remaining_qty);
            
            if actual_qty >= self.config.min_child_quantity {
                // Estimate execution price
                let fee_bps = if self.config.prefer_maker {
                    venue.maker_fee_bps
                } else {
                    venue.taker_fee_bps
                };
                
                let expected_price = match side {
                    Side::Buy => (current_price as f64 * (1.0 + fee_bps / 10000.0)) as Price,
                    Side::Sell => (current_price as f64 * (1.0 - fee_bps / 10000.0)) as Price,
                };

                decisions.push(RoutingDecision {
                    venue: venue_name,
                    quantity: actual_qty,
                    expected_price,
                    expected_slippage_bps: fee_bps,
                    priority: decisions.len() as u32,
                });

                remaining_qty -= actual_qty;
                self.orders_routed.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Handle any remaining quantity
        if remaining_qty > 0 && !decisions.is_empty() {
            // Add to the best venue
            decisions[0].quantity += remaining_qty;
        }

        self.total_value_routed.fetch_add(
            (total_quantity as u128 * current_price as u128) as u64,
            Ordering::Relaxed,
        );

        decisions
    }

    /// Score a venue based on multiple factors
    fn score_venue(&self, venue: &VenueInfo, side: Side, price: Price) -> f64 {
        let fee_score = if self.config.prefer_maker {
            1.0 - venue.maker_fee_bps / 100.0 // Lower fees = higher score
        } else {
            1.0 - venue.taker_fee_bps / 100.0
        };

        let spread_score = 1.0 - venue.avg_spread_bps / 100.0;
        let reliability_score = venue.reliability_score;
        let latency_score = 1.0 / (1.0 + venue.latency_ms as f64 / 100.0);

        // Weighted combination
        fee_score * 0.4 + spread_score * 0.3 + reliability_score * 0.2 + latency_score * 0.1
    }

    /// Get routing statistics
    pub fn get_stats(&self) -> HashMap<String, u64> {
        let mut stats = HashMap::new();
        stats.insert("orders_routed".to_string(), self.orders_routed.load(Ordering::Relaxed));
        stats.insert("total_value_routed".to_string(), self.total_value_routed.load(Ordering::Relaxed));
        stats
    }
}

// ============================================================================
// TCA Engine
// ============================================================================

/// Transaction Cost Analysis result for a single trade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCAResult {
    /// Trade identifier
    pub trade_id: u64,
    /// Symbol
    pub symbol: String,
    /// Side
    pub side: Side,
    /// Executed quantity
    pub quantity: Quantity,
    /// Execution price
    pub execution_price: Price,
    /// Arrival price (price when order was initiated)
    pub arrival_price: Price,
    /// VWAP during execution window
    pub vwap_price: Price,
    /// Implementation shortfall (bps)
    pub implementation_shortfall_bps: f64,
    /// Market timing cost (bps)
    pub timing_cost_bps: f64,
    /// Spread cost (bps)
    pub spread_cost_bps: f64,
    /// Market impact cost (bps)
    pub impact_cost_bps: f64,
    /// Fees paid (bps)
    pub fee_cost_bps: f64,
    /// Total transaction cost (bps)
    pub total_cost_bps: f64,
    /// Expected cost from model (bps)
    pub expected_cost_bps: f64,
    /// Cost vs expected difference (bps)
    pub cost_surprise_bps: f64,
    /// Execution timestamp
    pub timestamp_ns: TimestampNs,
}

impl TCAResult {
    /// Calculate implementation shortfall
    pub fn calculate_shortfall(&mut self, arrival_price: Price) {
        self.arrival_price = arrival_price;
        
        match self.side {
            Side::Buy => {
                self.implementation_shortfall_bps = 
                    ((self.execution_price - arrival_price) as f64 / arrival_price as f64) * 10000.0;
            }
            Side::Sell => {
                self.implementation_shortfall_bps = 
                    ((arrival_price - self.execution_price) as f64 / arrival_price as f64) * 10000.0;
            }
        }
    }

    /// Decompose costs into components
    pub fn decompose_costs(&mut self, spread_bps: f64, fee_bps: f64) {
        self.spread_cost_bps = spread_bps / 2.0;
        self.fee_cost_bps = fee_bps;
        
        // Residual is attributed to market impact and timing
        let residual = self.implementation_shortfall_bps - self.spread_cost_bps - self.fee_cost_bps;
        self.impact_cost_bps = residual * 0.7; // Assume 70% is impact
        self.timing_cost_bps = residual * 0.3; // Assume 30% is timing
        
        self.total_cost_bps = self.spread_cost_bps + self.impact_cost_bps 
            + self.timing_cost_bps + self.fee_cost_bps;
    }
}

/// Aggregate TCA statistics over multiple trades
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCAStatistics {
    /// Total number of trades
    pub trade_count: u64,
    /// Total volume traded
    pub total_volume: Quantity,
    /// Total value traded (USD)
    pub total_value_usd: f64,
    /// Average implementation shortfall (bps)
    pub avg_shortfall_bps: f64,
    /// Standard deviation of shortfall (bps)
    pub shortfall_std_bps: f64,
    /// Average spread cost (bps)
    pub avg_spread_cost_bps: f64,
    /// Average impact cost (bps)
    pub avg_impact_cost_bps: f64,
    /// Average timing cost (bps)
    pub avg_timing_cost_bps: f64,
    /// Average fee cost (bps)
    pub avg_fee_cost_bps: f64,
    /// Average total cost (bps)
    pub avg_total_cost_bps: f64,
    /// Average expected cost (bps)
    pub avg_expected_cost_bps: f64,
    /// Average cost surprise (bps)
    pub avg_surprise_bps: f64,
    /// Best (lowest) cost observed (bps)
    pub best_cost_bps: f64,
    /// Worst (highest) cost observed (bps)
    pub worst_cost_bps: f64,
    /// Percentage of trades that beat expectations
    pub beat_rate: f64,
}

/// Main TCA engine
pub struct TCAEngine {
    impact_params: ImpactParameters,
    router: SmartOrderRouter,
    trade_results: parking_lot::Mutex<Vec<TCAResult>>,
    current_volatility: parking_lot::RwLock<f64>,
    current_spreads: parking_lot::RwLock<HashMap<String, f64>>,
}

impl TCAEngine {
    pub fn new(impact_params: ImpactParameters, router_config: RouterConfig) -> Self {
        let router = SmartOrderRouter::new(router_config, impact_params.clone());
        
        Self {
            impact_params,
            router,
            trade_results: parking_lot::Mutex::new(Vec::new()),
            current_volatility: parking_lot::RwLock::new(0.02), // Default 2%
            current_spreads: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Update current volatility estimate
    pub fn update_volatility(&self, vol: f64) {
        *self.current_volatility.write() = vol;
    }

    /// Update spread for a symbol
    pub fn update_spread(&self, symbol: String, spread_bps: f64) {
        self.current_spreads.write().insert(symbol, spread_bps);
    }

    /// Record and analyze a trade
    pub fn record_trade(
        &self,
        trade_id: u64,
        symbol: String,
        side: Side,
        quantity: Quantity,
        execution_price: Price,
        arrival_price: Price,
        vwap_price: Price,
        fee_bps: f64,
    ) -> TCAResult {
        let spread_bps = self.current_spreads.read().get(&symbol).copied().unwrap_or(10.0);
        let volatility = *self.current_volatility.read();
        
        let mut result = TCAResult {
            trade_id,
            symbol,
            side,
            quantity,
            execution_price,
            arrival_price,
            vwap_price,
            implementation_shortfall_bps: 0.0,
            timing_cost_bps: 0.0,
            spread_cost_bps: 0.0,
            impact_cost_bps: 0.0,
            fee_cost_bps,
            total_cost_bps: 0.0,
            expected_cost_bps: 0.0,
            cost_surprise_bps: 0.0,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as TimestampNs,
        };

        // Calculate implementation shortfall
        result.calculate_shortfall(arrival_price);

        // Decompose costs
        result.decompose_costs(spread_bps, fee_bps);

        // Calculate expected cost
        let trade_value = quantity as f64 * execution_price as f64;
        let slippage = self.impact_params.calculate_slippage(trade_value, spread_bps, volatility);
        result.expected_cost_bps = slippage.total_cost_bps + fee_bps;

        // Calculate surprise
        result.cost_surprise_bps = result.total_cost_bps - result.expected_cost_bps;

        // Store result
        self.trade_results.lock().push(result.clone());

        result
    }

    /// Calculate aggregate statistics
    pub fn calculate_statistics(&self, symbol: Option<&str>) -> TCAStatistics {
        let results = self.trade_results.lock();
        
        let filtered: Vec<&TCAResult> = if let Some(sym) = symbol {
            results.iter().filter(|r| r.symbol == sym).collect()
        } else {
            results.iter().collect()
        };

        if filtered.is_empty() {
            return TCAStatistics {
                trade_count: 0,
                total_volume: 0,
                total_value_usd: 0.0,
                avg_shortfall_bps: 0.0,
                shortfall_std_bps: 0.0,
                avg_spread_cost_bps: 0.0,
                avg_impact_cost_bps: 0.0,
                avg_timing_cost_bps: 0.0,
                avg_fee_cost_bps: 0.0,
                avg_total_cost_bps: 0.0,
                avg_expected_cost_bps: 0.0,
                avg_surprise_bps: 0.0,
                best_cost_bps: 0.0,
                worst_cost_bps: 0.0,
                beat_rate: 0.0,
            };
        }

        let n = filtered.len() as f64;
        
        let total_volume: Quantity = filtered.iter().map(|r| r.quantity).sum();
        let total_value: f64 = filtered.iter()
            .map(|r| r.quantity as f64 * r.execution_price as f64)
            .sum();

        let shortfalls: Vec<f64> = filtered.iter().map(|r| r.implementation_shortfall_bps).collect();
        let avg_shortfall = shortfalls.iter().sum::<f64>() / n;
        let shortfall_variance = shortfalls.iter()
            .map(|s| (s - avg_shortfall).powi(2))
            .sum::<f64>() / (n - 1.0);
        let shortfall_std = shortfall_variance.sqrt();

        let avg_spread = filtered.iter().map(|r| r.spread_cost_bps).sum::<f64>() / n;
        let avg_impact = filtered.iter().map(|r| r.impact_cost_bps).sum::<f64>() / n;
        let avg_timing = filtered.iter().map(|r| r.timing_cost_bps).sum::<f64>() / n;
        let avg_fees = filtered.iter().map(|r| r.fee_cost_bps).sum::<f64>() / n;
        let avg_total = filtered.iter().map(|r| r.total_cost_bps).sum::<f64>() / n;
        let avg_expected = filtered.iter().map(|r| r.expected_cost_bps).sum::<f64>() / n;
        let avg_surprise = filtered.iter().map(|r| r.cost_surprise_bps).sum::<f64>() / n;

        let costs: Vec<f64> = filtered.iter().map(|r| r.total_cost_bps).collect();
        let best_cost = costs.iter().cloned().fold(f64::INFINITY, f64::min);
        let worst_cost = costs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let beat_count = filtered.iter().filter(|r| r.cost_surprise_bps <= 0.0).count();
        let beat_rate = beat_count as f64 / n;

        TCAStatistics {
            trade_count: filtered.len() as u64,
            total_volume,
            total_value_usd: total_value,
            avg_shortfall_bps: avg_shortfall,
            shortfall_std_bps: shortfall_std,
            avg_spread_cost_bps: avg_spread,
            avg_impact_cost_bps: avg_impact,
            avg_timing_cost_bps: avg_timing,
            avg_fee_cost_bps: avg_fees,
            avg_total_cost_bps: avg_total,
            avg_expected_cost_bps: avg_expected,
            avg_surprise_bps: avg_surprise,
            best_cost_bps: best_cost,
            worst_cost_bps: worst_cost,
            beat_rate,
        }
    }

    /// Get recent trade results
    pub fn get_recent_trades(&self, limit: usize) -> Vec<TCAResult> {
        let results = self.trade_results.lock();
        results.iter().rev().take(limit).cloned().collect()
    }

    /// Clear historical data
    pub fn clear_history(&self) {
        self.trade_results.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_impact_calculation() {
        let params = ImpactParameters::default();
        
        // Small trade (0.01% of daily volume)
        let impact_small = params.calculate_impact_bps(100_000.0, 10.0, 0.02);
        assert!(impact_small > 0.0);
        assert!(impact_small < 5.0); // Should be small

        // Large trade (1% of daily volume)
        let impact_large = params.calculate_impact_bps(10_000_000.0, 10.0, 0.02);
        assert!(impact_large > impact_small);
    }

    #[test]
    fn test_slippage_estimate() {
        let params = ImpactParameters::default();
        
        let estimate = params.calculate_slippage(1_000_000.0, 10.0, 0.02);
        
        assert!(estimate.total_cost_bps > 0.0);
        assert!(estimate.confidence_99 > estimate.confidence_95);
        assert!(estimate.confidence_95 > estimate.total_cost_bps);
    }

    #[test]
    fn test_tca_engine() {
        let impact_params = ImpactParameters::default();
        let router_config = RouterConfig {
            venues: vec![
                VenueInfo {
                    name: "BINANCE".to_string(),
                    maker_fee_bps: 1.0,
                    taker_fee_bps: 1.0,
                    avg_spread_bps: 5.0,
                    avg_daily_volume: 1e9,
                    latency_ms: 50,
                    reliability_score: 0.99,
                },
            ],
            min_child_quantity: 100,
            max_child_orders: 10,
            prefer_maker: true,
            use_dark_pools: false,
        };

        let engine = TCAEngine::new(impact_params, router_config);
        engine.update_spread("BTCUSD".to_string(), 5.0);

        // Record a trade
        let result = engine.record_trade(
            1,
            "BTCUSD".to_string(),
            Side::Buy,
            1000,
            50000,
            50050, // Arrival price slightly lower (we paid more)
            50025,
            1.0,
        );

        assert!(result.implementation_shortfall_bps > 0.0); // We underperformed arrival
        assert!(result.total_cost_bps > 0.0);
    }
}
