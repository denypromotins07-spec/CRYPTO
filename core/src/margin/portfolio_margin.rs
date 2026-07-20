//! Portfolio Margin Engine
//! 
//! Calculates real-time portfolio margin requirements and cross-collateral
//! optimization, ensuring the bot utilizes minimum capital for maximum leverage
//! without triggering liquidations.
//! 
//! Hardware Target: AMD Ryzen AI 5 with optimized calculations
//! Memory Constraint: Bounded position tracking, pre-allocated structures

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use dashmap::DashMap;

/// Position in a single asset
#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: u64,
    pub side: PositionSide,
    pub size: u64, // Fixed-point (size * 1e8)
    pub entry_price: u64,
    pub current_price: u64,
    pub unrealized_pnl: i64, // Fixed-point (pnl * 1e8)
    pub leverage: u32,
    pub liquidation_price: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionSide {
    Long,
    Short,
    None,
}

/// Margin requirements for a position
#[derive(Debug, Clone)]
pub struct MarginRequirement {
    pub symbol: u64,
    pub initial_margin: u64,
    pub maintenance_margin: u64,
    pub available_margin: u64,
    pub margin_ratio: f64,
    pub liquidation_risk: f32,
}

/// Portfolio-wide margin summary
#[derive(Debug, Clone)]
pub struct PortfolioMargin {
    pub total_collateral: u64,
    pub total_initial_margin: u64,
    pub total_maintenance_margin: u64,
    pub available_margin: u64,
    pub portfolio_margin_ratio: f64,
    pub max_withdrawable: u64,
    pub liquidation_risk_score: f32,
    pub timestamp_ns: u64,
}

/// Collateral optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub action: OptimizationAction,
    pub symbol: u64,
    pub amount: u64,
    pub expected_improvement: f64,
    pub priority: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationAction {
    AddCollateral,
    RemoveCollateral,
    ReducePosition,
    Rebalance,
}

/// Risk tier for margin calculations
#[derive(Debug, Clone, Copy)]
pub struct RiskTier {
    pub min_notional: u64,
    pub max_notional: u64,
    pub initial_margin_rate: f64,
    pub maintenance_margin_rate: f64,
}

/// Main portfolio margin engine
pub struct PortfolioMarginEngine {
    /// Active positions
    positions: DashMap<u64, Position>,
    
    /// Collateral balances per asset
    collateral: DashMap<u64, CollateralBalance>,
    
    /// Risk tiers per symbol (exchange-defined)
    risk_tiers: DashMap<u64, Vec<RiskTier>>,
    
    /// Current prices
    prices: DashMap<u64, u64>,
    
    /// Portfolio configuration
    max_leverage: u32,
    default_margin_rate: f64,
    
    /// Statistics
    margin_calls: AtomicU64,
    liquidations_avoided: AtomicU64,
    
    /// Running flag
    is_running: AtomicBool,
}

/// Collateral balance for an asset
#[derive(Debug, Clone)]
pub struct CollateralBalance {
    pub asset: String,
    pub free_balance: u64,
    pub locked_balance: u64,
    pub total_balance: u64,
    pub usd_value: u64,
    pub collateral_weight: f64, // 0.0 to 1.0
}

impl PortfolioMarginEngine {
    pub fn new(max_leverage: u32, default_margin_rate: f64) -> Self {
        Self {
            positions: DashMap::with_capacity(256),
            collateral: DashMap::with_capacity(64),
            risk_tiers: DashMap::with_capacity(256),
            prices: DashMap::with_capacity(256),
            max_leverage,
            default_margin_rate,
            margin_calls: AtomicU64::new(0),
            liquidations_avoided: AtomicU64::new(0),
            is_running: AtomicBool::new(false),
        }
    }

    /// Update or add a position
    #[inline]
    pub fn update_position(&self, position: Position) {
        self.positions.insert(position.symbol, position);
    }

    /// Update collateral balance
    #[inline]
    pub fn update_collateral(&self, asset: &str, balance: CollateralBalance) {
        // Use hash of asset name as key
        let key = hash_asset(asset);
        self.collateral.insert(key, balance);
    }

    /// Update price for a symbol
    #[inline]
    pub fn update_price(&self, symbol: u64, price: u64) {
        self.prices.insert(symbol, price);
        
        // Update position PnL
        if let Some(mut pos) = self.positions.get_mut(&symbol) {
            pos.current_price = price;
            pos.unrealized_pnl = calculate_pnl(&pos, price);
        }
    }

    /// Set risk tiers for a symbol
    pub fn set_risk_tiers(&self, symbol: u64, tiers: Vec<RiskTier>) {
        self.risk_tiers.insert(symbol, tiers);
    }

    /// Calculate margin requirement for a specific position
    pub fn calculate_position_margin(&self, symbol: u64) -> Option<MarginRequirement> {
        let position = self.positions.get(&symbol)?;
        let price = self.prices.get(&symbol).copied().unwrap_or(position.current_price);
        
        // Calculate notional value
        let notional = (position.size as f64 * price as f64 / 1e8) as u64;
        
        // Get applicable margin rate from risk tiers
        let margin_rate = self.get_margin_rate(symbol, notional);
        
        // Initial and maintenance margin
        let initial_margin = (notional as f64 * margin_rate.initial_margin_rate) as u64;
        let maintenance_margin = (notional as f64 * margin_rate.maintenance_margin_rate) as u64;
        
        // Available margin
        let collateral_value = self.get_total_collateral_value();
        let used_margin = self.calculate_total_used_margin();
        let available_margin = collateral_value.saturating_sub(used_margin);
        
        // Margin ratio
        let margin_ratio = if initial_margin > 0 {
            available_margin as f64 / initial_margin as f64
        } else {
            1.0
        };
        
        // Liquidation risk based on distance to liquidation price
        let liq_risk = calculate_liquidation_risk(&position, price);
        
        Some(MarginRequirement {
            symbol,
            initial_margin,
            maintenance_margin,
            available_margin,
            margin_ratio,
            liquidation_risk: liq_risk,
        })
    }

    /// Calculate portfolio-wide margin
    pub fn calculate_portfolio_margin(&self) -> PortfolioMargin {
        let now_ns = get_timestamp_ns();
        
        // Sum up all collateral
        let total_collateral: u64 = self.collateral.iter()
            .map(|e| e.value().usd_value)
            .sum();
        
        // Calculate total margin requirements
        let mut total_initial = 0u64;
        let mut total_maintenance = 0u64;
        let mut max_liq_risk = 0.0f32;
        
        for entry in self.positions.iter() {
            let symbol = *entry.key();
            if let Some(req) = self.calculate_position_margin(symbol) {
                total_initial += req.initial_margin;
                total_maintenance += req.maintenance_margin;
                max_liq_risk = max_liq_risk.max(req.liquidation_risk);
            }
        }
        
        let available_margin = total_collateral.saturating_sub(total_initial);
        
        // Portfolio margin ratio
        let portfolio_ratio = if total_initial > 0 {
            total_collateral as f64 / total_initial as f64
        } else {
            1.0
        };
        
        // Max withdrawable (available minus buffer)
        let buffer = (total_maintenance as f64 * 0.2) as u64; // 20% buffer
        let max_withdrawable = available_margin.saturating_sub(buffer);
        
        // Overall liquidation risk score
        let liq_risk_score = if portfolio_ratio > 1.0 {
            ((portfolio_ratio - 1.0) * 10.0).min(1.0) as f32
        } else {
            1.0
        };
        
        PortfolioMargin {
            total_collateral,
            total_initial_margin: total_initial,
            total_maintenance_margin: total_maintenance,
            available_margin,
            portfolio_margin_ratio: portfolio_ratio,
            max_withdrawable,
            liquidation_risk_score: liq_risk_score,
            timestamp_ns: now_ns,
        }
    }

    /// Generate optimization recommendations
    pub fn generate_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();
        let portfolio = self.calculate_portfolio_margin();
        
        // Check if we need more collateral
        if portfolio.portfolio_margin_ratio < 1.5 {
            recommendations.push(OptimizationRecommendation {
                action: OptimizationAction::AddCollateral,
                symbol: 0,
                amount: (portfolio.total_maintenance_margin as f64 * 0.5) as u64,
                expected_improvement: 0.5,
                priority: 1,
            });
        }
        
        // Check for positions that should be reduced
        for entry in self.positions.iter() {
            let symbol = *entry.key();
            if let Some(req) = self.calculate_position_margin(symbol) {
                if req.liquidation_risk > 0.8 {
                    // High risk position - recommend reduction
                    let position = self.positions.get(&symbol).unwrap();
                    recommendations.push(OptimizationRecommendation {
                        action: OptimizationAction::ReducePosition,
                        symbol,
                        amount: position.size / 2,
                        expected_improvement: 0.3,
                        priority: 2,
                    });
                }
            }
        }
        
        // Sort by priority
        recommendations.sort_by_key(|r| r.priority);
        
        recommendations
    }

    /// Get margin rate from risk tiers
    fn get_margin_rate(&self, symbol: u64, notional: u64) -> RiskTier {
        if let Some(tiers) = self.risk_tiers.get(&symbol) {
            for tier in tiers.iter() {
                if notional >= tier.min_notional && notional <= tier.max_notional {
                    return *tier;
                }
            }
        }
        
        // Default tier
        RiskTier {
            min_notional: 0,
            max_notional: u64::MAX,
            initial_margin_rate: 1.0 / self.max_leverage as f64,
            maintenance_margin_rate: self.default_margin_rate,
        }
    }

    /// Get total collateral value in USD
    fn get_total_collateral_value(&self) -> u64 {
        self.collateral.iter()
            .map(|e| e.value().usd_value)
            .sum()
    }

    /// Calculate total used margin
    fn calculate_total_used_margin(&self) -> u64 {
        let mut total = 0u64;
        for entry in self.positions.iter() {
            if let Some(req) = self.calculate_position_margin(*entry.key()) {
                total += req.initial_margin;
            }
        }
        total
    }

    /// Check for margin call conditions
    pub fn check_margin_call(&self) -> bool {
        let portfolio = self.calculate_portfolio_margin();
        
        if portfolio.portfolio_margin_ratio < 1.0 {
            self.margin_calls.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        
        false
    }

    /// Record avoided liquidation
    pub fn record_liquidation_avoided(&self) {
        self.liquidations_avoided.fetch_add(1, Ordering::Relaxed);
    }

    /// Get engine statistics
    pub fn get_stats(&self) -> MarginStats {
        MarginStats {
            active_positions: self.positions.len(),
            collateral_assets: self.collateral.len(),
            margin_calls: self.margin_calls.load(Ordering::Relaxed),
            liquidations_avoided: self.liquidations_avoided.load(Ordering::Relaxed),
        }
    }
}

/// Calculate PnL for a position
#[inline]
fn calculate_pnl(position: &Position, current_price: u64) -> i64 {
    match position.side {
        PositionSide::Long => {
            ((current_price as i64 - position.entry_price as i64) * position.size as i64 / 1e8 as i64)
        }
        PositionSide::Short => {
            ((position.entry_price as i64 - current_price as i64) * position.size as i64 / 1e8 as i64)
        }
        PositionSide::None => 0,
    }
}

/// Calculate liquidation risk score
fn calculate_liquidation_risk(position: &Position, current_price: u64) -> f32 {
    if position.liquidation_price == 0 {
        return 0.0;
    }
    
    match position.side {
        PositionSide::Long => {
            // Risk increases as price approaches liquidation from above
            if current_price <= position.liquidation_price {
                1.0
            } else {
                let distance = (current_price - position.liquidation_price) as f64;
                let price = current_price as f64;
                (1.0 - (distance / price).min(1.0)) as f32
            }
        }
        PositionSide::Short => {
            // Risk increases as price approaches liquidation from below
            if current_price >= position.liquidation_price {
                1.0
            } else {
                let distance = (position.liquidation_price - current_price) as f64;
                let price = current_price as f64;
                (1.0 - (distance / price).min(1.0)) as f32
            }
        }
        PositionSide::None => 0.0,
    }
}

/// Hash asset name to u64
fn hash_asset(name: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    name.hash(&mut hasher);
    hasher.finish()
}

#[derive(Debug)]
pub struct MarginStats {
    pub active_positions: usize,
    pub collateral_assets: usize,
    pub margin_calls: u64,
    pub liquidations_avoided: u64,
}

/// Get current timestamp in nanoseconds
#[inline(always)]
fn get_timestamp_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_margin_calculation() {
        let engine = PortfolioMarginEngine::new(10, 0.005);
        
        // Add collateral
        engine.update_collateral("USDT", CollateralBalance {
            asset: "USDT".to_string(),
            free_balance: 100000_000000,
            locked_balance: 0,
            total_balance: 100000_000000,
            usd_value: 100000_000000,
            collateral_weight: 1.0,
        });
        
        // Add position
        engine.update_position(Position {
            symbol: 12345,
            side: PositionSide::Long,
            size: 100_00000000,
            entry_price: 50000_00000000,
            current_price: 50000_00000000,
            unrealized_pnl: 0,
            leverage: 5,
            liquidation_price: 40000_00000000,
        });
        
        // Update price
        engine.update_price(12345, 50000_00000000);
        
        let portfolio = engine.calculate_portfolio_margin();
        assert!(portfolio.total_collateral > 0);
    }

    #[test]
    fn test_pnl_calculation() {
        let position = Position {
            symbol: 12345,
            side: PositionSide::Long,
            size: 100_00000000,
            entry_price: 50000_00000000,
            current_price: 51000_00000000,
            unrealized_pnl: 0,
            leverage: 5,
            liquidation_price: 40000_00000000,
        };
        
        let pnl = calculate_pnl(&position, 51000_00000000);
        assert!(pnl > 0);
    }
}
