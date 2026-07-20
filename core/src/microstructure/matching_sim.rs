//! Microsecond Matching Engine Simulator
//! 
//! A local matching engine simulator used to dry-run complex multi-leg orders
//! before sending them to the exchange. This prevents rejected orders and saves
//! API rate limits.
//! 
//! Simulates FIFO matching, partial fills, queue positioning, and latency.
//! 
//! Hardware Target: AMD Ryzen AI 5 with optimized order matching
//! Memory Constraint: Bounded order book depth and history

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use dashmap::DashMap;
use std::collections::BTreeMap;

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Side {
    Buy,
    Sell,
}

/// Order types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderType {
    Market,
    Limit(u64), // Price
    ImmediateOrCancel,
    FillOrKill,
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderStatus {
    New,
    PartiallyFilled { filled: u64, remaining: u64 },
    Filled,
    Cancelled,
    Rejected,
}

/// Order representation
#[derive(Debug, Clone)]
pub struct Order {
    pub order_id: u64,
    pub symbol: u64,
    pub side: Side,
    pub order_type: OrderType,
    pub price: u64,
    pub quantity: u64,
    pub filled_quantity: u64,
    pub status: OrderStatus,
    pub timestamp_ns: u64,
    pub time_in_force: TimeInForce,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeInForce {
    GTC, // Good Till Cancel
    IOC, // Immediate Or Cancel
    FOK, // Fill Or Kill
    GTD, // Good Till Date
}

/// Trade execution result
#[derive(Debug, Clone)]
pub struct Trade {
    pub trade_id: u64,
    pub symbol: u64,
    pub price: u64,
    pub quantity: u64,
    pub maker_order_id: u64,
    pub taker_order_id: u64,
    pub timestamp_ns: u64,
}

/// Order book level
#[derive(Debug, Clone)]
struct BookLevel {
    price: u64,
    orders: Vec<Order>,
    total_size: u64,
}

/// Simulated order book
struct SimBook {
    bids: BTreeMap<u64, BookLevel>, // Price -> Level (descending)
    asks: BTreeMap<u64, BookLevel>, // Price -> Level (ascending)
}

impl SimBook {
    fn new() -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
        }
    }

    fn get_best_bid(&self) -> Option<u64> {
        self.bids.last_key_value().map(|(k, _)| *k)
    }

    fn get_best_ask(&self) -> Option<u64> {
        self.asks.first_key_value().map(|(k, _)| *k)
    }

    fn add_order(&mut self, order: &Order) {
        let levels = match order.side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };

        let entry = levels.entry(order.price).or_insert_with(|| BookLevel {
            price: order.price,
            orders: Vec::new(),
            total_size: 0,
        });

        entry.orders.push(order.clone());
        entry.total_size += order.quantity - order.filled_quantity;
    }

    fn remove_order(&mut self, order_id: u64, side: Side, price: u64) {
        let levels = match side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };

        if let Some(level) = levels.get_mut(&price) {
            if let Some(pos) = level.orders.iter().position(|o| o.order_id == order_id) {
                let order = &level.orders[pos];
                level.total_size = level.total_size.saturating_sub(order.quantity - order.filled_quantity);
                level.orders.remove(pos);
            }
            if level.orders.is_empty() {
                levels.remove(&price);
            }
        }
    }
}

/// Simulation result for an order
#[derive(Debug, Clone)]
pub struct SimResult {
    pub order_id: u64,
    pub status: OrderStatus,
    pub filled_quantity: u64,
    pub avg_fill_price: u64,
    pub trades: Vec<Trade>,
    pub rejection_reason: Option<String>,
    pub simulation_time_ns: u64,
}

/// Main matching engine simulator
pub struct MatchingSimulator {
    /// Order books per symbol
    books: DashMap<u64, SimBook>,
    
    /// All orders by ID
    orders: DashMap<u64, Order>,
    
    /// Trade history
    trades: DashMap<u64, Vec<Trade>>,
    
    /// Order ID counter
    order_counter: AtomicU64,
    trade_counter: AtomicU64,
    
    /// Simulation statistics
    orders_processed: AtomicU64,
    trades_simulated: AtomicU64,
    rejections: AtomicU64,
    
    /// Running flag
    is_running: AtomicBool,
    
    /// Simulated latency in nanoseconds
    simulated_latency_ns: AtomicU64,
}

impl MatchingSimulator {
    pub fn new(simulated_latency_ns: u64) -> Self {
        Self {
            books: DashMap::with_capacity(256),
            orders: DashMap::with_capacity(1024),
            trades: DashMap::with_capacity(256),
            order_counter: AtomicU64::new(0),
            trade_counter: AtomicU64::new(0),
            orders_processed: AtomicU64::new(0),
            trades_simulated: AtomicU64::new(0),
            rejections: AtomicU64::new(0),
            is_running: AtomicBool::new(false),
            simulated_latency_ns: AtomicU64::new(simulated_latency_ns),
        }
    }

    /// Initialize order book with a snapshot
    pub fn initialize_book(&self, symbol: u64, bids: Vec<(u64, u64)>, asks: Vec<(u64, u64)>) {
        let mut book = SimBook::new();
        
        for (price, size) in bids {
            let order = Order {
                order_id: self.order_counter.fetch_add(1, Ordering::Relaxed),
                symbol,
                side: Side::Buy,
                order_type: OrderType::Limit(price),
                price,
                quantity: size,
                filled_quantity: 0,
                status: OrderStatus::New,
                timestamp_ns: get_timestamp_ns(),
                time_in_force: TimeInForce::GTC,
            };
            book.add_order(&order);
        }
        
        for (price, size) in asks {
            let order = Order {
                order_id: self.order_counter.fetch_add(1, Ordering::Relaxed),
                symbol,
                side: Side::Sell,
                order_type: OrderType::Limit(price),
                price,
                quantity: size,
                filled_quantity: 0,
                status: OrderStatus::New,
                timestamp_ns: get_timestamp_ns(),
                time_in_force: TimeInForce::GTC,
            };
            book.add_order(&order);
        }
        
        self.books.insert(symbol, book);
    }

    /// Simulate an order and return the result without modifying state
    pub fn simulate_order(&self, symbol: u64, side: Side, order_type: OrderType, quantity: u64) -> SimResult {
        let start_time = get_timestamp_ns();
        let order_id = self.order_counter.fetch_add(1, Ordering::Relaxed);
        
        let mut result = SimResult {
            order_id,
            status: OrderStatus::New,
            filled_quantity: 0,
            avg_fill_price: 0,
            trades: Vec::new(),
            rejection_reason: None,
            simulation_time_ns: 0,
        };
        
        // Get or create book
        let book = self.books.entry(symbol).or_insert_with(SimBook::new);
        
        let price_limit = match order_type {
            OrderType::Market => None,
            OrderType::Limit(p) => Some(p),
            OrderType::ImmediateOrCancel => None,
            OrderType::FillOrKill => None,
        };
        
        let opposite_side = match side {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        };
        
        let opposite_book = match side {
            Side::Buy => &book.asks,
            Side::Sell => &book.bids,
        };
        
        let mut remaining = quantity;
        let mut total_value = 0u64;
        let mut trades = Vec::new();
        
        // Try to match against opposite book
        for (_, level) in opposite_book.iter() {
            if remaining == 0 {
                break;
            }
            
            // Check price limit
            if let Some(limit) = price_limit {
                if side == Side::Buy && level.price > limit {
                    break;
                }
                if side == Side::Sell && level.price < limit {
                    break;
                }
            }
            
            let fill_qty = remaining.min(level.total_size);
            if fill_qty > 0 {
                total_value += fill_qty * level.price;
                remaining -= fill_qty;
                
                let trade = Trade {
                    trade_id: self.trade_counter.fetch_add(1, Ordering::Relaxed),
                    symbol,
                    price: level.price,
                    quantity: fill_qty,
                    maker_order_id: level.orders.first().map(|o| o.order_id).unwrap_or(0),
                    taker_order_id: order_id,
                    timestamp_ns: get_timestamp_ns(),
                };
                trades.push(trade);
                
                self.trades_simulated.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        let filled_qty = quantity - remaining;
        
        // Determine final status
        if filled_qty == 0 {
            result.status = match order_type {
                OrderType::FillOrKill => OrderStatus::Rejected,
                OrderType::ImmediateOrCancel => OrderStatus::Cancelled,
                _ => OrderStatus::New,
            };
            if matches!(order_type, OrderType::FillOrKill) {
                result.rejection_reason = Some("FillOrKill: Could not fill any quantity".to_string());
                self.rejections.fetch_add(1, Ordering::Relaxed);
            }
        } else if remaining == 0 {
            result.status = OrderStatus::Filled;
        } else {
            result.status = OrderStatus::PartiallyFilled {
                filled: filled_qty,
                remaining,
            };
            
            // For IOC, cancel remaining
            if matches!(order_type, OrderType::ImmediateOrCancel) {
                result.status = OrderStatus::PartiallyFilled {
                    filled: filled_qty,
                    remaining: 0,
                };
            }
        }
        
        result.filled_quantity = filled_qty;
        result.avg_fill_price = if filled_qty > 0 { total_value / filled_qty } else { 0 };
        result.trades = trades;
        result.simulation_time_ns = get_timestamp_ns() - start_time + self.simulated_latency_ns.load(Ordering::Relaxed);
        
        self.orders_processed.fetch_add(1, Ordering::Relaxed);
        
        result
    }

    /// Simulate a multi-leg order (e.g., pairs trade, arbitrage)
    pub fn simulate_multi_leg(&self, legs: Vec<LegOrder>) -> MultiLegResult {
        let start_time = get_timestamp_ns();
        let mut results = Vec::with_capacity(legs.len());
        let mut all_would_fill = true;
        
        // First pass: check if all legs can fill (for FOK logic)
        for leg in &legs {
            let sim = self.simulate_order(leg.symbol, leg.side, leg.order_type, leg.quantity);
            if matches!(leg.order_type, OrderType::FillOrKill) && sim.filled_quantity < leg.quantity {
                all_would_fill = false;
            }
            results.push(sim);
        }
        
        // If FOK and not all fill, mark all as rejected
        if !all_would_fill {
            for result in &mut results {
                result.status = OrderStatus::Rejected;
                result.rejection_reason = Some("Multi-leg FOK: Not all legs could fill".to_string());
            }
        }
        
        let total_trades: usize = results.iter().map(|r| r.trades.len()).sum();
        
        MultiLegResult {
            legs: results,
            all_filled: all_would_fill,
            total_trades,
            simulation_time_ns: get_timestamp_ns() - start_time,
        }
    }

    /// Update simulated order book from real market data
    pub fn update_book_snapshot(&self, symbol: u64, bids: Vec<(u64, u64)>, asks: Vec<(u64, u64)>) {
        self.initialize_book(symbol, bids, asks);
    }

    /// Set simulated network latency
    pub fn set_simulated_latency(&self, latency_ns: u64) {
        self.simulated_latency_ns.store(latency_ns, Ordering::Relaxed);
    }

    /// Get simulation statistics
    pub fn get_stats(&self) -> SimStats {
        SimStats {
            symbols_tracked: self.books.len(),
            orders_processed: self.orders_processed.load(Ordering::Relaxed),
            trades_simulated: self.trades_simulated.load(Ordering::Relaxed),
            rejections: self.rejections.load(Ordering::Relaxed),
            simulated_latency_ns: self.simulated_latency_ns.load(Ordering::Relaxed),
        }
    }
}

/// Single leg of a multi-leg order
#[derive(Debug, Clone)]
pub struct LegOrder {
    pub symbol: u64,
    pub side: Side,
    pub order_type: OrderType,
    pub quantity: u64,
}

/// Result of multi-leg simulation
#[derive(Debug, Clone)]
pub struct MultiLegResult {
    pub legs: Vec<SimResult>,
    pub all_filled: bool,
    pub total_trades: usize,
    pub simulation_time_ns: u64,
}

#[derive(Debug)]
pub struct SimStats {
    pub symbols_tracked: usize,
    pub orders_processed: u64,
    pub trades_simulated: u64,
    pub rejections: u64,
    pub simulated_latency_ns: u64,
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
    fn test_market_order_simulation() {
        let sim = MatchingSimulator::new(1_000_000); // 1ms simulated latency
        
        // Initialize book with some liquidity
        sim.initialize_book(
            12345,
            vec![(49999_00000000, 100_00000000), (49998_00000000, 200_00000000)],
            vec![(50001_00000000, 100_00000000), (50002_00000000, 200_00000000)],
        );
        
        // Simulate a market buy
        let result = sim.simulate_order(
            12345,
            Side::Buy,
            OrderType::Market,
            50_00000000,
        );
        
        assert!(result.filled_quantity > 0);
        assert_eq!(result.trades.len(), 1);
        assert!(result.simulation_time_ns >= 1_000_000);
    }

    #[test]
    fn test_limit_order_partial_fill() {
        let sim = MatchingSimulator::new(0);
        
        sim.initialize_book(
            12345,
            vec![],
            vec![(50000_00000000, 100_00000000)],
        );
        
        // Simulate a limit buy that won't fill
        let result = sim.simulate_order(
            12345,
            Side::Buy,
            OrderType::Limit(49999_00000000),
            50_00000000,
        );
        
        assert_eq!(result.filled_quantity, 0);
        assert_eq!(result.status, OrderStatus::New);
    }

    #[test]
    fn test_fok_rejection() {
        let sim = MatchingSimulator::new(0);
        
        sim.initialize_book(
            12345,
            vec![],
            vec![(50000_00000000, 50_00000000)],
        );
        
        // FOK order that can't fully fill
        let result = sim.simulate_order(
            12345,
            Side::Buy,
            OrderType::FillOrKill,
            100_00000000,
        );
        
        assert_eq!(result.status, OrderStatus::Rejected);
        assert!(result.rejection_reason.is_some());
    }
}
