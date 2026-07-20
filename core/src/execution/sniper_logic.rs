//! Ultra-Low Latency Sniper Logic for Liquidity Detection
//! 
//! Detects fleeting liquidity or large hidden orders and fires marketable limit orders 
//! instantly to capture the spread or front-run large players. Optimized for microsecond 
//! execution on AMD Ryzen AI 5 hardware.
//!
//! Target Hardware: AMD Ryzen AI 5 + AMD Radeon GPU + 16GB RAM (8GB cap)

use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::{RwLock, Mutex};
use log::{info, warn, debug, error};
use serde::{Serialize, Deserialize};

use crate::types::{Symbol, Side, Price, Quantity, OrderId, Timestamp};
use crate::execution::order_book::OrderBook;

/// Configuration for sniper logic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SniperConfig {
    /// Minimum spread threshold in basis points to trigger sniping
    pub min_spread_bps: u32,
    /// Maximum order size to snipe (avoid large adverse selection)
    pub max_snipe_size: Quantity,
    /// Minimum order size worth sniping
    pub min_snipe_size: Quantity,
    /// Latency threshold for stale quote detection (microseconds)
    pub stale_quote_latency_us: u64,
    /// Aggression level (0.0 - 1.0): higher = more aggressive
    pub aggression_level: f64,
    /// Maximum position delta allowed from sniping
    pub max_position_delta: Quantity,
    /// Cooldown period after snipe (milliseconds)
    pub cooldown_ms: u64,
    /// Enable hidden order detection
    pub detect_hidden_orders: bool,
}

impl Default for SniperConfig {
    fn default() -> Self {
        Self {
            min_spread_bps: 5,           // 5 bps minimum spread
            max_snipe_size: 1.0,         // Max 1 unit
            min_snipe_size: 0.01,        // Min 0.01 units
            stale_quote_latency_us: 100, // 100 microseconds
            aggression_level: 0.7,       // Moderately aggressive
            max_position_delta: 5.0,     // Max 5 units delta
            cooldown_ms: 50,             // 50ms cooldown
            detect_hidden_orders: true,
        }
    }
}

/// Detected sniping opportunity
#[derive(Debug, Clone)]
pub struct SnipeOpportunity {
    pub symbol: Symbol,
    pub side: Side,
    pub price: Price,
    pub quantity: Quantity,
    pub expected_slippage_bps: f64,
    pub confidence: f64,
    pub detected_at: Instant,
    pub opportunity_type: OpportunityType,
}

/// Type of sniping opportunity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpportunityType {
    /// Wide spread arbitrage
    SpreadArb,
    /// Stale quote detection
    StaleQuote,
    /// Hidden order detected
    HiddenOrder,
    /// Large trade imbalance
    Imbalance,
    /// Cross-exchange arb (if multi-venue)
    CrossVenue,
}

/// Statistics for a sniped order
#[derive(Debug, Clone)]
pub struct SnipeStats {
    pub total_snipes: u64,
    pub successful_snipes: u64,
    pub failed_snipes: u64,
    pub total_profit: f64,
    pub avg_profit_per_snipe: f64,
    pub best_snipe_profit: f64,
    pub worst_snipe_loss: f64,
    pub hit_rate: f64,
}

/// Main sniper engine
pub struct SniperEngine {
    config: SniperConfig,
    /// Reference to live order book
    order_book: Arc<RwLock<OrderBook>>,
    /// Current position tracking
    current_position: RwLock<f64>,
    /// Last snipe time per symbol
    last_snipe_time: RwLock<std::collections::HashMap<Symbol, Instant>>,
    /// Active opportunities
    active_opportunities: RwLock<Vec<SnipeOpportunity>>,
    /// Statistics
    stats: RwLock<SnipeStats>,
    /// Enabled flag
    enabled: RwLock<bool>,
}

impl SniperEngine {
    /// Create new sniper engine
    pub fn new(config: SniperConfig, order_book: Arc<RwLock<OrderBook>>) -> Self {
        Self {
            config,
            order_book,
            current_position: RwLock::new(0.0),
            last_snipe_time: RwLock::new(std::collections::HashMap::new()),
            active_opportunities: RwLock::new(Vec::new()),
            stats: RwLock::new(SnipeStats {
                total_snipes: 0,
                successful_snipes: 0,
                failed_snipes: 0,
                total_profit: 0.0,
                avg_profit_per_snipe: 0.0,
                best_snipe_profit: 0.0,
                worst_snipe_loss: 0.0,
                hit_rate: 0.0,
            }),
            enabled: RwLock::new(true),
        }
    }

    /// Scan for sniping opportunities
    pub fn scan(&self) -> Vec<SnipeOpportunity> {
        if !*self.enabled.read() {
            return Vec::new();
        }

        let ob = self.order_book.read();
        let mut opportunities = Vec::new();

        // Check spread arbitrage
        if let Some((spread_bps, bid_price, ask_price, bid_qty, ask_qty)) = self.check_spread(&ob) {
            if spread_bps >= self.config.min_spread_bps as f64 {
                opportunities.push(SnipeOpportunity {
                    symbol: ob.symbol(),
                    side: Side::Buy,
                    price: bid_price,
                    quantity: bid_qty.min(self.config.max_snipe_size),
                    expected_slippage_bps: spread_bps * 0.5,
                    confidence: self.calculate_confidence(spread_bps),
                    detected_at: Instant::now(),
                    opportunity_type: OpportunityType::SpreadArb,
                });
                
                opportunities.push(SnipeOpportunity {
                    symbol: ob.symbol(),
                    side: Side::Sell,
                    price: ask_price,
                    quantity: ask_qty.min(self.config.max_snipe_size),
                    expected_slippage_bps: spread_bps * 0.5,
                    confidence: self.calculate_confidence(spread_bps),
                    detected_at: Instant::now(),
                    opportunity_type: OpportunityType::SpreadArb,
                });
            }
        }

        // Check for stale quotes
        if let Some(opportunity) = self.check_stale_quotes(&ob) {
            opportunities.push(opportunity);
        }

        // Check for hidden orders
        if self.config.detect_hidden_orders {
            if let Some(opportunity) = self.check_hidden_orders(&ob) {
                opportunities.push(opportunity);
            }
        }

        // Filter by cooldown
        let now = Instant::now();
        let last_snipe = self.last_snipe_time.read();
        opportunities.retain(|opp| {
            if let Some(last_time) = last_snipe.get(&opp.symbol) {
                now.duration_since(*last_time).as_millis() as u64 >= self.config.cooldown_ms
            } else {
                true
            }
        });

        // Sort by confidence
        opportunities.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

        *self.active_opportunities.write() = opportunities.clone();
        opportunities
    }

    /// Check spread for arbitrage opportunity
    fn check_spread(&self, ob: &OrderBook) -> Option<(f64, Price, Price, Quantity, Quantity)> {
        let best_bid = ob.best_bid()?;
        let best_ask = ob.best_ask()?;
        
        if best_bid <= 0.0 || best_ask <= 0.0 {
            return None;
        }

        let spread = best_ask - best_bid;
        let spread_bps = (spread / best_bid) * 10000.0;
        
        let bid_qty = ob.best_bid_qty().unwrap_or(0.0);
        let ask_qty = ob.best_ask_qty().unwrap_or(0.0);

        // Ensure quantities are within limits
        if bid_qty < self.config.min_snipe_size || ask_qty < self.config.min_snipe_size {
            return None;
        }

        Some((spread_bps, best_bid, best_ask, bid_qty, ask_qty))
    }

    /// Check for stale quotes (quotes that haven't updated in a while)
    fn check_stale_quotes(&self, ob: &OrderBook) -> Option<SnipeOpportunity> {
        let now = Instant::now();
        let last_update = ob.last_update_time()?;
        let elapsed_us = now.duration_since(last_update).as_micros() as u64;

        if elapsed_us > self.config.stale_quote_latency_us {
            // Quote is stale - opportunity to snipe
            let best_bid = ob.best_bid()?;
            let best_ask = ob.best_ask()?;
            
            // Confidence increases with staleness
            let confidence = ((elapsed_us - self.config.stale_quote_latency_us) as f64 
                / self.config.stale_quote_latency_us as f64).min(1.0);

            return Some(SnipeOpportunity {
                symbol: ob.symbol(),
                side: Side::Buy,
                price: best_bid,
                quantity: ob.best_bid_qty().unwrap_or(0.0),
                expected_slippage_bps: 2.0,
                confidence,
                detected_at: now,
                opportunity_type: OpportunityType::StaleQuote,
            });
        }

        None
    }

    /// Detect hidden orders through order flow analysis
    fn check_hidden_orders(&self, ob: &OrderBook) -> Option<SnipeOpportunity> {
        // Look for signs of hidden liquidity:
        // 1. Large trades that don't move the book
        // 2. Consistent replenishment at a price level
        // 3. Unusual order-to-trade ratio
        
        let recent_trades = ob.recent_trades();
        if recent_trades.is_empty() {
            return None;
        }

        // Calculate order flow imbalance
        let buy_volume: Quantity = recent_trades.iter()
            .filter(|t| t.side == Side::Buy)
            .map(|t| t.quantity)
            .sum();
        
        let sell_volume: Quantity = recent_trades.iter()
            .filter(|t| t.side == Side::Sell)
            .map(|t| t.quantity)
            .sum();

        let total_volume = buy_volume + sell_volume;
        if total_volume < self.config.min_snipe_size {
            return None;
        }

        let imbalance = (buy_volume - sell_volume) / total_volume;

        // Significant imbalance suggests hidden order
        if imbalance.abs() > 0.7 {
            let side = if imbalance > 0.0 { Side::Buy } else { Side::Sell };
            let price = if side == Side::Buy {
                ob.best_ask()?
            } else {
                ob.best_bid()?
            };

            return Some(SnipeOpportunity {
                symbol: ob.symbol(),
                side,
                price,
                quantity: total_volume * 0.1, // Snipe 10% of volume
                expected_slippage_bps: 3.0,
                confidence: imbalance.abs(),
                detected_at: Instant::now(),
                opportunity_type: OpportunityType::HiddenOrder,
            });
        }

        None
    }

    /// Calculate confidence score for an opportunity
    fn calculate_confidence(&self, spread_bps: f64) -> f64 {
        // Base confidence from spread
        let spread_conf = (spread_bps / 100.0).min(1.0);
        
        // Adjust by aggression level
        spread_conf * self.config.aggression_level
    }

    /// Execute a snipe
    pub fn execute_snipe(&self, opportunity: &SnipeOpportunity) -> Result<OrderId, SniperError> {
        // Check position limits
        let current_pos = *self.current_position.read();
        let proposed_delta = match opportunity.side {
            Side::Buy => opportunity.quantity,
            Side::Sell => -opportunity.quantity,
        };

        if (current_pos + proposed_delta).abs() > self.config.max_position_delta {
            return Err(SniperError::PositionLimitExceeded);
        }

        // Check cooldown
        let now = Instant::now();
        let mut last_snipe = self.last_snipe_time.write();
        if let Some(last_time) = last_snipe.get(&opportunity.symbol) {
            if now.duration_since(*last_time).as_millis() as u64 < self.config.cooldown_ms {
                return Err(SniperError::CooldownActive);
            }
        }

        // Generate order ID
        let order_id = OrderId::new(random_order_id());

        // Update tracking
        last_snipe.insert(opportunity.symbol, now);
        *self.current_position.write() += proposed_delta;

        info!(
            "SNIPED: {} {} @ {:.8} (type: {:?}, confidence: {:.2})",
            opportunity.side,
            opportunity.quantity,
            opportunity.price,
            opportunity.opportunity_type,
            opportunity.confidence
        );

        Ok(order_id)
    }

    /// Record snipe result
    pub fn record_result(&self, order_id: OrderId, profit: f64, success: bool) {
        let mut stats = self.stats.write();
        
        stats.total_snipes += 1;
        if success {
            stats.successful_snipes += 1;
            stats.total_profit += profit;
            
            if profit > stats.best_snipe_profit {
                stats.best_snipe_profit = profit;
            }
            if profit < stats.worst_snipe_loss {
                stats.worst_snipe_loss = profit;
            }
        } else {
            stats.failed_snipes += 1;
        }

        stats.hit_rate = stats.successful_snipes as f64 / stats.total_snipes as f64;
        stats.avg_profit_per_snipe = stats.total_profit / stats.total_snipes as f64;
    }

    /// Get current statistics
    pub fn get_stats(&self) -> SnipeStats {
        self.stats.read().clone()
    }

    /// Enable/disable sniping
    pub fn set_enabled(&self, enabled: bool) {
        info!("Sniper engine {}", if enabled { "enabled" } else { "disabled" });
        *self.enabled.write() = enabled;
    }

    /// Reset position tracking
    pub fn reset_position(&self) {
        *self.current_position.write() = 0.0;
    }

    /// Get active opportunities
    pub fn get_active_opportunities(&self) -> Vec<SnipeOpportunity> {
        self.active_opportunities.read().clone()
    }
}

/// Sniper errors
#[derive(Debug, Clone)]
pub enum SniperError {
    PositionLimitExceeded,
    CooldownActive,
    NoLiquidity,
    InvalidPrice,
    OrderRejected,
}

impl std::fmt::Display for SniperError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PositionLimitExceeded => write!(f, "Position limit exceeded"),
            Self::CooldownActive => write!(f, "Cooldown period active"),
            Self::NoLiquidity => write!(f, "No liquidity available"),
            Self::InvalidPrice => write!(f, "Invalid price"),
            Self::OrderRejected => write!(f, "Order rejected"),
        }
    }
}

/// Generate random order ID
fn random_order_id() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = SniperConfig::default();
        assert_eq!(config.min_spread_bps, 5);
        assert_eq!(config.aggression_level, 0.7);
    }

    #[test]
    fn test_opportunity_creation() {
        let opp = SnipeOpportunity {
            symbol: Symbol::BTCUSDT,
            side: Side::Buy,
            price: 50000.0,
            quantity: 0.1,
            expected_slippage_bps: 2.0,
            confidence: 0.8,
            detected_at: Instant::now(),
            opportunity_type: OpportunityType::SpreadArb,
        };

        assert_eq!(opp.side, Side::Buy);
        assert!(opp.confidence > 0.0);
    }

    #[test]
    fn test_stats_tracking() {
        let stats = SnipeStats {
            total_snipes: 10,
            successful_snipes: 7,
            failed_snipes: 3,
            total_profit: 100.0,
            avg_profit_per_snipe: 10.0,
            best_snipe_profit: 50.0,
            worst_snipe_loss: -20.0,
            hit_rate: 0.7,
        };

        assert_eq!(stats.hit_rate, 0.7);
        assert_eq!(stats.total_snipes, stats.successful_snipes + stats.failed_snipes);
    }
}
