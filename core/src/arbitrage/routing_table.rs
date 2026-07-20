//! Dynamic Routing Table for Cross-Exchange Execution
//! 
//! Calculates the most profitable execution path across multiple venues,
//! factoring in real-time fees, withdrawal limits, slippage models, and
//! network latency.
//! 
//! Uses graph-based routing algorithms optimized for microsecond decision-making.
//! 
//! Hardware Target: AMD Ryzen AI 5 with cache-optimized data structures
//! Memory Constraint: Pre-allocated graph structures, bounded memory usage

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use dashmap::DashMap;
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering as CmpOrdering;

/// Represents an exchange node in the routing graph
#[derive(Debug, Clone)]
pub struct ExchangeNode {
    pub id: u16,
    pub name: String,
    pub maker_fee_bps: i32,
    pub taker_fee_bps: i32,
    pub withdrawal_fee_bps: i32,
    pub withdrawal_limit_usd: u64,
    pub daily_withdrawal_used: AtomicU64,
    is_active: AtomicBool,
    avg_latency_ns: AtomicU64,
    reliability_score: AtomicU64, // 0-10000 scale
}

/// Represents a trading pair on an exchange
#[derive(Debug, Clone)]
pub struct TradingPair {
    pub symbol: u64,
    pub base_asset: String,
    pub quote_asset: String,
    pub min_qty: u64,
    pub max_qty: u64,
    pub qty_step: u64,
    pub price_step: u64,
    pub current_liquidity_bids: u64, // In base asset units
    pub current_liquidity_asks: u64,
}

/// Edge in the routing graph representing a possible trade leg
#[derive(Debug, Clone)]
pub struct RoutingEdge {
    pub from_exchange: u16,
    pub to_exchange: u16,
    pub symbol: u64,
    pub direction: TradeDirection,
    pub expected_slippage_bps: i32,
    pub total_fees_bps: i32,
    pub estimated_latency_ns: u64,
    pub max_executable_qty: u64,
    pub confidence_score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeDirection {
    Buy,
    Sell,
}

/// Complete execution route from source to destination
#[derive(Debug, Clone)]
pub struct ExecutionRoute {
    pub route_id: u64,
    pub legs: Vec<RoutingEdge>,
    pub total_cost_bps: i32,
    pub expected_profit_bps: i32,
    pub total_latency_ns: u64,
    pub max_executable_qty: u64,
    pub risk_score: f32,
    pub created_at_ns: u64,
}

/// Priority queue item for route finding
#[derive(Debug, Clone)]
struct RouteQueueItem {
    cost: i32,
    exchange: u16,
    symbol: u64,
}

impl PartialEq for RouteQueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for RouteQueueItem {}

impl PartialOrd for RouteQueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for RouteQueueItem {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Reverse ordering for min-heap behavior
        other.cost.cmp(&self.cost)
    }
}

/// Main routing table engine
pub struct RoutingTable {
    /// All registered exchanges
    exchanges: DashMap<u16, ExchangeNode>,
    
    /// Trading pairs per exchange
    pairs: DashMap<(u16, u64), TradingPair>,
    
    /// Pre-computed routes for common arbitrage patterns
    cached_routes: DashMap<u64, Vec<ExecutionRoute>>,
    
    /// Real-time slippage estimates per exchange/pair
    slippage_estimates: DashMap<(u16, u64), SlippageModel>,
    
    /// Network latency matrix
    latency_matrix: DashMap<(u16, u16), u64>,
    
    /// Route counter for unique IDs
    route_counter: AtomicU64,
    
    /// Minimum profitability threshold in bps
    min_profit_bps: i32,
    
    /// Maximum allowed risk score
    max_risk_score: f32,
}

/// Slippage model using linear market impact
#[derive(Debug, Clone)]
pub struct SlippageModel {
    pub base_slippage_bps: i32,
    pub impact_coefficient: f64, // Slippage per unit of quantity
    pub liquidity_factor: f32,
    pub last_updated_ns: u64,
}

impl SlippageModel {
    #[inline]
    pub fn calculate_slippage(&self, qty: u64) -> i32 {
        let impact = (qty as f64 * self.impact_coefficient) as i32;
        (self.base_slippage_bps + impact) as i32
    }
}

impl RoutingTable {
    pub fn new(min_profit_bps: i32, max_risk_score: f32) -> Self {
        Self {
            exchanges: DashMap::with_capacity(64),
            pairs: DashMap::with_capacity(4096),
            cached_routes: DashMap::with_capacity(256),
            slippage_estimates: DashMap::with_capacity(4096),
            latency_matrix: DashMap::with_capacity(256),
            route_counter: AtomicU64::new(0),
            min_profit_bps,
            max_risk_score,
        }
    }

    /// Register a new exchange
    pub fn register_exchange(&self, node: ExchangeNode) {
        self.exchanges.insert(node.id, node);
    }

    /// Register a trading pair on an exchange
    pub fn register_pair(&self, exchange_id: u16, pair: TradingPair) {
        self.pairs.insert((exchange_id, pair.symbol), pair);
        
        // Initialize slippage model
        self.slippage_estimates.insert(
            (exchange_id, pair.symbol),
            SlippageModel {
                base_slippage_bps: 2,
                impact_coefficient: 0.000001,
                liquidity_factor: 1.0,
                last_updated_ns: get_timestamp_ns(),
            },
        );
    }

    /// Update network latency between two exchanges
    pub fn update_latency(&self, from: u16, to: u16, latency_ns: u64) {
        self.latency_matrix.insert((from, to), latency_ns);
    }

    /// Update slippage model based on recent executions
    pub fn update_slippage_model(&self, exchange_id: u16, symbol: u64, actual_slippage_bps: i32, qty: u64) {
        if let Some(mut model) = self.slippage_estimates.get_mut(&(exchange_id, symbol)) {
            // EMA update
            let alpha = 0.1;
            let current_base = model.base_slippage_bps as f64;
            model.base_slippage_bps = ((current_base * (1.0 - alpha)) + (actual_slippage_bps as f64 * alpha)) as i32;
            
            // Update impact coefficient
            if qty > 0 {
                let implied_impact = (actual_slippage_bps - model.base_slippage_bps) as f64 / qty as f64;
                model.impact_coefficient = (model.impact_coefficient * (1.0 - alpha)) + (implied_impact * alpha);
            }
            
            model.last_updated_ns = get_timestamp_ns();
        }
    }

    /// Find optimal execution route for a given trade
    pub fn find_optimal_route(
        &self,
        symbol: u64,
        qty: u64,
        direction: TradeDirection,
    ) -> Option<ExecutionRoute> {
        let mut best_route: Option<ExecutionRoute> = None;
        
        // Get all active exchanges with this trading pair
        let candidate_exchanges: Vec<u16> = self.exchanges
            .iter()
            .filter(|e| e.value().is_active.load(Ordering::Relaxed))
            .filter(|e| self.pairs.contains_key(&(e.key().clone(), symbol)))
            .map(|e| *e.key())
            .collect();
        
        if candidate_exchanges.len() < 2 {
            return None;
        }
        
        // Try all pairs of exchanges for simple arb routes
        for i in 0..candidate_exchanges.len() {
            for j in (i + 1)..candidate_exchanges.len() {
                let ex1 = candidate_exchanges[i];
                let ex2 = candidate_exchanges[j];
                
                if let Some(route) = self.build_two_leg_route(symbol, qty, direction, ex1, ex2) {
                    if route.expected_profit_bps > self.min_profit_bps 
                        && route.risk_score < self.max_risk_score 
                    {
                        match &best_route {
                            None => best_route = Some(route),
                            Some(best) => {
                                if route.expected_profit_bps > best.expected_profit_bps {
                                    best_route = Some(route);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        best_route
    }

    /// Build a two-leg arbitrage route
    fn build_two_leg_route(
        &self,
        symbol: u64,
        qty: u64,
        direction: TradeDirection,
        ex1: u16,
        ex2: u16,
    ) -> Option<ExecutionRoute> {
        let pair1 = self.pairs.get(&(ex1, symbol))?;
        let pair2 = self.pairs.get(&(ex2, symbol))?;
        let node1 = self.exchanges.get(&ex1)?;
        let node2 = self.exchanges.get(&ex2)?;
        
        // Check withdrawal limits
        let daily_used1 = node1.daily_withdrawal_used.load(Ordering::Relaxed);
        let daily_used2 = node2.daily_withdrawal_used.load(Ordering::Relaxed);
        
        if daily_used1 >= node1.withdrawal_limit_usd || daily_used2 >= node2.withdrawal_limit_usd {
            return None;
        }
        
        // Calculate slippage for each leg
        let slippage1 = self.slippage_estimates
            .get(&(ex1, symbol))
            .map(|s| s.calculate_slippage(qty))
            .unwrap_or(5);
        
        let slippage2 = self.slippage_estimates
            .get(&(ex2, symbol))
            .map(|s| s.calculate_slippage(qty))
            .unwrap_or(5);
        
        // Calculate fees
        let fee1 = node1.taker_fee_bps;
        let fee2 = node2.taker_fee_bps;
        
        // Get latency
        let latency = self.latency_matrix
            .get(&(ex1, ex2))
            .copied()
            .unwrap_or(10_000_000); // Default 10ms
        
        // Build legs
        let leg1 = RoutingEdge {
            from_exchange: ex1,
            to_exchange: ex2,
            symbol,
            direction,
            expected_slippage_bps: slippage1,
            total_fees_bps: fee1,
            estimated_latency_ns: latency / 2,
            max_executable_qty: std::cmp::min(qty, pair1.current_liquidity_bids),
            confidence_score: 0.9,
        };
        
        let leg2 = RoutingEdge {
            from_exchange: ex2,
            to_exchange: ex1,
            symbol,
            direction: match direction {
                TradeDirection::Buy => TradeDirection::Sell,
                TradeDirection::Sell => TradeDirection::Buy,
            },
            expected_slippage_bps: slippage2,
            total_fees_bps: fee2,
            estimated_latency_ns: latency / 2,
            max_executable_qty: std::cmp::min(qty, pair2.current_liquidity_asks),
            confidence_score: 0.9,
        };
        
        let total_cost_bps = slippage1 + slippage2 + fee1 + fee2;
        let max_qty = std::cmp::min(leg1.max_executable_qty, leg2.max_executable_qty);
        
        // Estimate profit (simplified - actual calculation requires price data)
        let expected_profit_bps = 10 - total_cost_bps; // Placeholder
        
        let route = ExecutionRoute {
            route_id: self.route_counter.fetch_add(1, Ordering::Relaxed),
            legs: vec![leg1, leg2],
            total_cost_bps,
            expected_profit_bps,
            total_latency_ns: latency,
            max_executable_qty: max_qty,
            risk_score: self.calculate_risk_score(&[leg1.clone(), leg2.clone()]),
            created_at_ns: get_timestamp_ns(),
        };
        
        Some(route)
    }

    /// Calculate risk score for a route
    fn calculate_risk_score(&self, legs: &[RoutingEdge]) -> f32 {
        let mut risk = 0.0;
        
        for leg in legs {
            // Latency risk
            risk += (leg.estimated_latency_ns as f32 / 10_000_000.0).min(1.0) * 0.3;
            
            // Slippage risk
            risk += (leg.expected_slippage_bps as f32 / 20.0).min(1.0) * 0.4;
            
            // Confidence factor
            risk += (1.0 - leg.confidence_score) * 0.3;
        }
        
        risk / legs.len() as f32
    }

    /// Get all viable routes for a symbol
    pub fn get_all_routes(&self, symbol: u64, qty: u64) -> Vec<ExecutionRoute> {
        let mut routes = Vec::with_capacity(32);
        
        let candidate_exchanges: Vec<u16> = self.exchanges
            .iter()
            .filter(|e| e.value().is_active.load(Ordering::Relaxed))
            .filter(|e| self.pairs.contains_key(&(e.key().clone(), symbol)))
            .map(|e| *e.key())
            .collect();
        
        for i in 0..candidate_exchanges.len() {
            for j in (i + 1)..candidate_exchanges.len() {
                if let Some(route) = self.build_two_leg_route(symbol, qty, TradeDirection::Buy, candidate_exchanges[i], candidate_exchanges[j]) {
                    routes.push(route);
                }
            }
        }
        
        // Sort by expected profit
        routes.sort_by(|a, b| b.expected_profit_bps.cmp(&a.expected_profit_bps));
        
        routes
    }

    /// Update exchange availability
    pub fn set_exchange_active(&self, exchange_id: u16, active: bool) {
        if let Some(exchange) = self.exchanges.get(&exchange_id) {
            exchange.is_active.store(active, Ordering::Relaxed);
        }
    }

    /// Record withdrawal usage
    pub fn record_withdrawal(&self, exchange_id: u16, amount_usd: u64) {
        if let Some(exchange) = self.exchanges.get(&exchange_id) {
            exchange.daily_withdrawal_used.fetch_add(amount_usd, Ordering::Relaxed);
        }
    }

    /// Reset daily withdrawal counters (call at UTC midnight)
    pub fn reset_daily_counters(&self) {
        for exchange in self.exchanges.iter() {
            exchange.value().daily_withdrawal_used.store(0, Ordering::Relaxed);
        }
    }

    /// Get routing statistics
    pub fn get_stats(&self) -> RoutingStats {
        RoutingStats {
            total_exchanges: self.exchanges.len(),
            active_exchanges: self.exchanges.iter().filter(|e| e.value().is_active.load(Ordering::Relaxed)).count(),
            total_pairs: self.pairs.len(),
            cached_routes: self.cached_routes.len(),
        }
    }
}

#[derive(Debug)]
pub struct RoutingStats {
    pub total_exchanges: usize,
    pub active_exchanges: usize,
    pub total_pairs: usize,
    pub cached_routes: usize,
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
    use std::sync::atomic::AtomicU64;

    #[test]
    fn test_routing_table_creation() {
        let table = RoutingTable::new(5, 0.8);
        
        let exchange1 = ExchangeNode {
            id: 1,
            name: "Binance".to_string(),
            maker_fee_bps: 1,
            taker_fee_bps: 1,
            withdrawal_fee_bps: 0,
            withdrawal_limit_usd: 1_000_000,
            daily_withdrawal_used: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
            avg_latency_ns: AtomicU64::new(5_000_000),
            reliability_score: AtomicU64::new(9500),
        };
        
        table.register_exchange(exchange1);
        
        let stats = table.get_stats();
        assert_eq!(stats.total_exchanges, 1);
        assert_eq!(stats.active_exchanges, 1);
    }

    #[test]
    fn test_slippage_calculation() {
        let model = SlippageModel {
            base_slippage_bps: 2,
            impact_coefficient: 0.000001,
            liquidity_factor: 1.0,
            last_updated_ns: 0,
        };
        
        let slippage_small = model.calculate_slippage(100_000_000); // 1 BTC
        let slippage_large = model.calculate_slippage(1000_000_000); // 10 BTC
        
        assert!(slippage_large > slippage_small);
    }
}
