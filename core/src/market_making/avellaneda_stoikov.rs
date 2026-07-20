//! Avellaneda-Stoikov Market Making Model
//! =========================================
//! Chapter 4, File 1: Rust Advanced Market Making & Inventory Risk
//!
//! Implementation of the Avellaneda-Stoikov market making model. 
//! Calculates the optimal reservation price and bid-ask spread dynamically 
//! based on current inventory, volatility, and time horizon.
//!
//! Reference: Avellaneda, M., & Stoikov, S. (2008). High-frequency trading 
//! in a limit order book. Quantitative Finance.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;
use chrono::{DateTime, Utc};

/// Market making quote with bid/ask prices and sizes
#[derive(Debug, Clone)]
pub struct Quote {
    /// Bid price
    pub bid_price: f64,
    /// Ask price
    pub ask_price: f64,
    /// Bid size (quantity)
    pub bid_size: f64,
    /// Ask size (quantity)
    pub ask_size: f64,
    /// Reservation price (fair value)
    pub reservation_price: f64,
    /// Spread in basis points
    pub spread_bps: f64,
    /// Timestamp
    pub timestamp_ns: u64,
}

impl Quote {
    /// Create a new quote
    pub fn new(
        bid_price: f64,
        ask_price: f64,
        bid_size: f64,
        ask_size: f64,
        reservation_price: f64,
    ) -> Self {
        let mid = (bid_price + ask_price) / 2.0;
        let spread_bps = if mid > 0.0 {
            (ask_price - bid_price) / mid * 10000.0
        } else {
            0.0
        };
        
        Self {
            bid_price,
            ask_price,
            bid_size,
            ask_size,
            reservation_price,
            spread_bps,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }
    
    /// Get mid price
    pub fn mid_price(&self) -> f64 {
        (self.bid_price + self.ask_price) / 2.0
    }
    
    /// Get half spread in bps
    pub fn half_spread_bps(&self) -> f64 {
        self.spread_bps / 2.0
    }
}

/// Parameters for Avellaneda-Stoikov model
#[derive(Debug, Clone)]
pub struct ASModelParams {
    /// Risk aversion coefficient (gamma)
    /// Higher values = more conservative quoting
    pub risk_aversion: f64,
    
    /// Order book liquidity parameter (kappa)
    /// Measures sensitivity of order arrival to spread
    pub kappa: f64,
    
    /// Order arrival intensity at zero spread (lambda_0)
    pub lambda_zero: f64,
    
    /// Volatility estimate (annualized)
    pub volatility: f64,
    
    /// Time horizon in seconds
    pub time_horizon_s: f64,
    
    /// Maximum inventory position
    pub max_inventory: f64,
    
    /// Minimum spread (bps)
    pub min_spread_bps: f64,
    
    /// Maximum spread (bps)
    pub max_spread_bps: f64,
}

impl Default for ASModelParams {
    fn default() -> Self {
        Self {
            risk_aversion: 0.1,         // Moderate risk aversion
            kappa: 5.0,                 // Typical liquidity parameter
            lambda_zero: 1.0,           // Base arrival rate
            volatility: 0.02,           // 2% daily vol
            time_horizon_s: 300.0,      // 5 minute horizon
            max_inventory: 100.0,       // Max 100 units
            min_spread_bps: 2.0,        // Min 2 bps
            max_spread_bps: 50.0,       // Max 50 bps
        }
    }
}

/// Current state for market making
#[derive(Debug, Clone)]
pub struct MarketState {
    /// Current mid price
    pub mid_price: f64,
    /// Current inventory (positive = long)
    pub inventory: f64,
    /// Realized P&L
    pub realized_pnl: f64,
    /// Unrealized P&L
    pub unrealized_pnl: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl MarketState {
    pub fn new(mid_price: f64, inventory: f64) -> Self {
        Self {
            mid_price,
            inventory,
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            timestamp: Utc::now(),
        }
    }
}

/// Avellaneda-Stoikov Market Maker
pub struct AvellanedaStoikovMM {
    /// Model parameters
    params: RwLock<ASModelParams>,
    /// Current market state
    state: RwLock<MarketState>,
    /// Current quote
    current_quote: RwLock<Option<Quote>>,
    /// Trade counter
    trade_count: AtomicU64,
    /// Quote update counter
    quote_updates: AtomicU64,
}

impl AvellanedaStoikovMM {
    /// Create new market maker with default parameters
    pub fn new(initial_price: f64, max_inventory: f64) -> Self {
        let mut params = ASModelParams::default();
        params.max_inventory = max_inventory;
        
        Self {
            params: RwLock::new(params),
            state: RwLock::new(MarketState::new(initial_price, 0.0)),
            current_quote: RwLock::new(None),
            trade_count: AtomicU64::new(0),
            quote_updates: AtomicU64::new(0),
        }
    }
    
    /// Create with custom parameters
    pub fn with_params(params: ASModelParams, initial_price: f64) -> Self {
        Self {
            params: RwLock::new(params),
            state: RwLock::new(MarketState::new(initial_price, 0.0)),
            current_quote: RwLock::new(None),
            trade_count: AtomicU64::new(0),
            quote_updates: AtomicU64::new(0),
        }
    }
    
    /// Update model parameters
    pub fn update_params(&self, params: ASModelParams) {
        *self.params.write() = params;
    }
    
    /// Get current parameters
    pub fn get_params(&self) -> ASModelParams {
        self.params.read().clone()
    }
    
    /// Update market state
    pub fn update_state(&self, mid_price: f64, inventory: f64) {
        let mut state = self.state.write();
        state.mid_price = mid_price;
        state.inventory = inventory;
        state.timestamp = Utc::now();
        
        // Update unrealized P&L
        state.unrealized_pnl = inventory * (mid_price - state.mid_price);
    }
    
    /// Calculate reservation price (Avellaneda-Stoikov formula)
    /// 
    /// r = s - q * gamma * sigma^2 * (T - t)
    /// 
    /// Where:
    /// - r: reservation price
    /// - s: mid price
    /// - q: inventory
    /// - gamma: risk aversion
    /// - sigma: volatility
    /// - T - t: remaining time horizon
    pub fn calculate_reservation_price(&self) -> f64 {
        let params = self.params.read();
        let state = self.state.read();
        
        let mid = state.mid_price;
        let q = state.inventory;
        let gamma = params.risk_aversion;
        let sigma = params.volatility;
        let tau = params.time_horizon_s;
        
        // Reservation price adjustment for inventory risk
        let adjustment = q * gamma * sigma * sigma * tau;
        
        // Skew reservation price away from inventory
        let reservation = mid - adjustment;
        
        reservation
    }
    
    /// Calculate optimal spread (Avellaneda-Stoikov formula)
    /// 
    /// delta = 1/gamma + gamma * sigma^2 * (T - t) / 2
    /// 
    /// Returns half-spread in price units
    pub fn calculate_optimal_spread(&self) -> f64 {
        let params = self.params.read();
        let state = self.state.read();
        
        let gamma = params.risk_aversion;
        let sigma = params.volatility;
        let tau = params.time_horizon_s;
        let mid = state.mid_price;
        
        // Optimal half-spread in absolute terms
        // delta = 1/gamma + (gamma * sigma^2 * tau) / 2
        let base_spread = 1.0 / gamma;
        let risk_spread = gamma * sigma * sigma * tau / 2.0;
        
        let half_spread = (base_spread + risk_spread) * mid / 10000.0;
        
        // Convert to bps for limits checking
        let half_spread_bps = if mid > 0.0 {
            half_spread / mid * 10000.0
        } else {
            0.0
        };
        
        // Apply spread limits
        let min_half = params.min_spread_bps / 2.0;
        let max_half = params.max_spread_bps / 2.0;
        
        let clamped_bps = half_spread_bps.clamp(min_half, max_half);
        clamped_bps / 10000.0 * mid
    }
    
    /// Generate optimal quote
    pub fn generate_quote(&self) -> Quote {
        let reservation = self.calculate_reservation_price();
        let half_spread = self.calculate_optimal_spread();
        
        let bid_price = reservation - half_spread;
        let ask_price = reservation + half_spread;
        
        // Size based on inventory (reduce size when near limits)
        let params = self.params.read();
        let state = self.state.read();
        
        let inventory_ratio = (state.inventory / params.max_inventory).abs();
        let size_factor = (1.0 - inventory_ratio).max(0.2); // At least 20% size
        
        let base_size = params.max_inventory * 0.1; // 10% of max per quote
        let bid_size = base_size * size_factor;
        let ask_size = base_size * size_factor;
        
        drop(params);
        drop(state);
        
        let quote = Quote::new(bid_price, ask_price, bid_size, ask_size, reservation);
        
        // Store current quote
        *self.current_quote.write() = Some(quote.clone());
        self.quote_updates.fetch_add(1, Ordering::Relaxed);
        
        quote
    }
    
    /// Process a fill event
    pub fn process_fill(
        &self,
        side: Side,
        quantity: f64,
        price: f64,
    ) -> FillResult {
        let mut state = self.state.write();
        
        // Update inventory
        match side {
            Side::Buy => {
                state.inventory += quantity;
            }
            Side::Sell => {
                state.inventory -= quantity;
            }
        }
        
        // Update realized P&L (simplified)
        // In production, track cost basis properly
        state.realized_pnl += match side {
            Side::Buy => -quantity * price,
            Side::Sell => quantity * price,
        };
        
        self.trade_count.fetch_add(1, Ordering::Relaxed);
        
        FillResult {
            side,
            quantity,
            price,
            new_inventory: state.inventory,
            new_unrealized_pnl: state.unrealized_pnl,
        }
    }
    
    /// Get current inventory
    pub fn get_inventory(&self) -> f64 {
        self.state.read().inventory
    }
    
    /// Get current P&L
    pub fn get_pnl(&self) -> (f64, f64) {
        let state = self.state.read();
        (state.realized_pnl, state.unrealized_pnl)
    }
    
    /// Get statistics
    pub fn get_statistics(&self) -> MMStatistics {
        let state = self.state.read();
        let params = self.params.read();
        let quote = self.current_quote.read();
        
        MMStatistics {
            inventory: state.inventory,
            realized_pnl: state.realized_pnl,
            unrealized_pnl: state.unrealized_pnl,
            total_pnl: state.realized_pnl + state.unrealized_pnl,
            mid_price: state.mid_price,
            reservation_price: quote.as_ref().map(|q| q.reservation_price).unwrap_or(0.0),
            current_spread_bps: quote.as_ref().map(|q| q.spread_bps).unwrap_or(0.0),
            trade_count: self.trade_count.load(Ordering::Relaxed),
            quote_updates: self.quote_updates.load(Ordering::Relaxed),
            risk_aversion: params.risk_aversion,
            volatility: params.volatility,
        }
    }
    
    /// Reset market maker state
    pub fn reset(&self, initial_price: f64) {
        *self.state.write() = MarketState::new(initial_price, 0.0);
        *self.current_quote.write() = None;
        self.trade_count.store(0, Ordering::Relaxed);
    }
}

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// Result of a fill
#[derive(Debug, Clone)]
pub struct FillResult {
    pub side: Side,
    pub quantity: f64,
    pub price: f64,
    pub new_inventory: f64,
    pub new_unrealized_pnl: f64,
}

/// Market maker statistics
#[derive(Debug, Clone)]
pub struct MMStatistics {
    pub inventory: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_pnl: f64,
    pub mid_price: f64,
    pub reservation_price: f64,
    pub current_spread_bps: f64,
    pub trade_count: u64,
    pub quote_updates: u64,
    pub risk_aversion: f64,
    pub volatility: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_reservation_price_skew() {
        let params = ASModelParams {
            risk_aversion: 0.5,
            volatility: 0.02,
            time_horizon_s: 300.0,
            ..Default::default()
        };
        
        let mm = AvellanedaStoikovMM::with_params(params, 100.0);
        
        // Test with no inventory
        mm.update_state(100.0, 0.0);
        let r_no_inv = mm.calculate_reservation_price();
        assert!((r_no_inv - 100.0).abs() < 0.01);
        
        // Test with positive inventory (should skew down)
        mm.update_state(100.0, 10.0);
        let r_long = mm.calculate_reservation_price();
        assert!(r_long < 100.0);
        
        // Test with negative inventory (should skew up)
        mm.update_state(100.0, -10.0);
        let r_short = mm.calculate_reservation_price();
        assert!(r_short > 100.0);
        
        println!("No inventory: {}", r_no_inv);
        println!("Long inventory: {}", r_long);
        println!("Short inventory: {}", r_short);
    }
    
    #[test]
    fn test_spread_calculation() {
        let params = ASModelParams {
            risk_aversion: 0.1,
            volatility: 0.02,
            time_horizon_s: 300.0,
            min_spread_bps: 2.0,
            max_spread_bps: 50.0,
            ..Default::default()
        };
        
        let mm = AvellanedaStoikovMM::with_params(params, 100.0);
        mm.update_state(100.0, 0.0);
        
        let spread = mm.calculate_optimal_spread();
        let spread_bps = spread / 100.0 * 10000.0;
        
        println!("Optimal half-spread: {:.2} bps", spread_bps);
        assert!(spread_bps >= 1.0); // At least 1 bps half spread
        assert!(spread_bps <= 25.0); // Not more than 25 bps half spread
    }
    
    #[test]
    fn test_quote_generation() {
        let mm = AvellanedaStoikovMM::new(100.0, 100.0);
        mm.update_state(100.0, 5.0);
        
        let quote = mm.generate_quote();
        
        println!("Bid: {:.4}, Ask: {:.4}, Mid: {:.4}", 
                 quote.bid_price, quote.ask_price, quote.mid_price());
        println!("Reservation: {:.4}, Spread: {:.2} bps", 
                 quote.reservation_price, quote.spread_bps);
        
        // With positive inventory, reservation should be below mid
        assert!(quote.reservation_price < 100.0);
        
        // Bid should be below reservation, ask above
        assert!(quote.bid_price < quote.reservation_price);
        assert!(quote.ask_price > quote.reservation_price);
    }
}
