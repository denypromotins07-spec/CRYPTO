//! Ultra-Low Latency Event-Driven Backtesting Engine
//! 
//! This module implements a tick-level event-driven backtesting engine in Rust
//! designed for microsecond-precision simulation of crypto trading strategies.
//! Features include matching engine simulation, order book dynamics, latency modeling,
//! and queue positioning awareness.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

/// High-precision timestamp in nanoseconds since epoch
pub type TimestampNs = u64;

/// Price represented as integer (avoids floating point issues)
pub type Price = i64;

/// Quantity represented as integer
pub type Quantity = u64;

/// Order ID type
pub type OrderId = u64;

/// Static counter for generating unique IDs
static ORDER_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate unique order ID
#[inline]
fn generate_order_id() -> OrderId {
    ORDER_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Get current time in nanoseconds
#[inline]
fn get_timestamp_ns() -> TimestampNs {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as TimestampNs
}

// ============================================================================
// Core Data Structures
// ============================================================================

/// Side of the market
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    PostOnly,
    ImmediateOrCancel,
    FillOrKill,
}

/// Order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

/// Order representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub order_id: OrderId,
    pub client_order_id: String,
    pub symbol: String,
    pub side: Side,
    pub order_type: OrderType,
    pub price: Option<Price>,
    pub quantity: Quantity,
    pub filled_quantity: Quantity,
    pub remaining_quantity: Quantity,
    pub status: OrderStatus,
    pub timestamp_ns: TimestampNs,
    pub fill_timestamp_ns: Option<TimestampNs>,
    pub average_fill_price: Option<Price>,
    pub fees_paid: u64,
    /// Queue position for limit orders
    pub queue_position: Option<u32>,
}

impl Order {
    pub fn new_limit(
        symbol: String,
        side: Side,
        price: Price,
        quantity: Quantity,
        client_order_id: Option<String>,
    ) -> Self {
        Self {
            order_id: generate_order_id(),
            client_order_id: client_order_id.unwrap_or_else(|| format!("ord_{}", generate_order_id())),
            symbol,
            side,
            order_type: OrderType::Limit,
            price: Some(price),
            quantity,
            filled_quantity: 0,
            remaining_quantity: quantity,
            status: OrderStatus::New,
            timestamp_ns: get_timestamp_ns(),
            fill_timestamp_ns: None,
            average_fill_price: None,
            fees_paid: 0,
            queue_position: None,
        }
    }

    pub fn new_market(
        symbol: String,
        side: Side,
        quantity: Quantity,
        client_order_id: Option<String>,
    ) -> Self {
        Self {
            order_id: generate_order_id(),
            client_order_id: client_order_id.unwrap_or_else(|| format!("ord_{}", generate_order_id())),
            symbol,
            side,
            order_type: OrderType::Market,
            price: None,
            quantity,
            filled_quantity: 0,
            remaining_quantity: quantity,
            status: OrderStatus::New,
            timestamp_ns: get_timestamp_ns(),
            fill_timestamp_ns: None,
            average_fill_price: None,
            fees_paid: 0,
            queue_position: None,
        }
    }

    #[inline]
    pub fn is_filled(&self) -> bool {
        self.remaining_quantity == 0
    }

    #[inline]
    pub fn fill_ratio(&self) -> f64 {
        self.filled_quantity as f64 / self.quantity as f64
    }
}

/// Trade execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub trade_id: u64,
    pub order_id: OrderId,
    pub symbol: String,
    pub side: Side,
    pub price: Price,
    pub quantity: Quantity,
    pub maker: bool,
    pub fee: u64,
    pub timestamp_ns: TimestampNs,
    pub liquidity_taken: Quantity,
}

/// Level in the order book
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceLevel {
    pub price: Price,
    pub quantity: Quantity,
    pub order_count: u32,
    pub orders: Vec<OrderId>,
}

/// Order book side (bids or asks)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSide {
    pub levels: BTreeMap<Price, PriceLevel>,
    pub total_quantity: Quantity,
}

impl OrderBookSide {
    pub fn new() -> Self {
        Self {
            levels: BTreeMap::new(),
            total_quantity: 0,
        }
    }

    #[inline]
    pub fn best_price(&self) -> Option<Price> {
        self.levels.keys().next().copied()
    }

    pub fn add_order(&mut self, order: &Order) {
        if let Some(price) = order.price {
            let level = self.levels.entry(price).or_insert_with(|| PriceLevel {
                price,
                quantity: 0,
                order_count: 0,
                orders: Vec::new(),
            });
            level.quantity += order.remaining_quantity;
            level.order_count += 1;
            level.orders.push(order.order_id);
            self.total_quantity += order.remaining_quantity;
        }
    }

    pub fn remove_order(&mut self, order: &Order) {
        if let Some(price) = order.price {
            if let Some(level) = self.levels.get_mut(&price) {
                level.quantity = level.quantity.saturating_sub(order.remaining_quantity);
                level.order_count = level.order_count.saturating_sub(1);
                level.orders.retain(|&id| id != order.order_id);
                self.total_quantity = self.total_quantity.saturating_sub(order.remaining_quantity);

                if level.quantity == 0 {
                    self.levels.remove(&price);
                }
            }
        }
    }
}

// ============================================================================
// Order Book
// ============================================================================

/// Full order book with bid/ask sides
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub bids: OrderBookSide,
    pub asks: OrderBookSide,
    pub last_trade_price: Option<Price>,
    pub last_update_ns: TimestampNs,
    pub spread: Option<Price>,
    pub mid_price: Option<Price>,
}

impl OrderBook {
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            bids: OrderBookSide::new(),
            asks: OrderBookSide::new(),
            last_trade_price: None,
            last_update_ns: get_timestamp_ns(),
            spread: None,
            mid_price: None,
        }
    }

    #[inline]
    pub fn best_bid(&self) -> Option<Price> {
        self.bids.best_price()
    }

    #[inline]
    pub fn best_ask(&self) -> Option<Price> {
        self.asks.best_price()
    }

    /// Update spread and mid price
    pub fn update_derived_values(&mut self) {
        if let (Some(bid), Some(ask)) = (self.best_bid(), self.best_ask()) {
            self.spread = Some(ask - bid);
            self.mid_price = Some((bid + ask) / 2);
        }
        self.last_update_ns = get_timestamp_ns();
    }

    /// Get volume-weighted average price for given quantity
    pub fn get_vwap(&self, side: Side, quantity: Quantity) -> Option<Price> {
        let book_side = match side {
            Side::Buy => &self.asks,
            Side::Sell => &self.bids,
        };

        let mut remaining = quantity;
        let mut total_value: u128 = 0;

        for (_, level) in book_side.levels.iter() {
            if remaining == 0 {
                break;
            }

            let fill_qty = remaining.min(level.quantity);
            total_value += (fill_qty as u128) * (level.price as u128);
            remaining -= fill_qty;
        }

        if remaining > 0 {
            return None; // Insufficient liquidity
        }

        Some((total_value / quantity as u128) as Price)
    }

    /// Calculate slippage for market order
    pub fn calculate_slippage(&self, side: Side, quantity: Quantity) -> f64 {
        let reference_price = self.mid_price.unwrap_or(self.last_trade_price.unwrap_or(0)) as f64;
        if reference_price == 0.0 {
            return 0.0;
        }

        if let Some(exec_price) = self.get_vwap(side, quantity) {
            let exec_price_f64 = exec_price as f64;
            match side {
                Side::Buy => (exec_price_f64 - reference_price) / reference_price,
                Side::Sell => (reference_price - exec_price_f64) / reference_price,
            }
        } else {
            1.0 // Maximum slippage if no liquidity
        }
    }
}

// ============================================================================
// Matching Engine
// ============================================================================

/// Matching engine result
#[derive(Debug, Clone)]
pub struct MatchResult {
    pub trades: Vec<Trade>,
    pub filled_orders: Vec<OrderId>,
    pub partial_orders: Vec<OrderId>,
    pub rejected_orders: Vec<OrderId>,
}

/// Ultra-low latency matching engine
pub struct MatchingEngine {
    order_books: RwLock<std::collections::HashMap<String, OrderBook>>,
    orders: RwLock<std::collections::HashMap<OrderId, Order>>,
    trades: Mutex<Vec<Trade>>,
    fee_rate_bps: u32, // Basis points (e.g., 10 bps = 0.1%)
    min_fee: u64,
}

impl MatchingEngine {
    pub fn new(fee_rate_bps: u32, min_fee: u64) -> Self {
        Self {
            order_books: RwLock::new(std::collections::HashMap::new()),
            orders: RwLock::new(std::collections::HashMap::new()),
            trades: Mutex::new(Vec::new()),
            fee_rate_bps,
            min_fee,
        }
    }

    /// Get or create order book for symbol
    #[inline]
    fn get_or_create_book(&self, symbol: &str) -> std::sync::RwLockWriteGuard<'_, OrderBook> {
        let mut books = self.order_books.write();
        if !books.contains_key(symbol) {
            books.insert(symbol.to_string(), OrderBook::new(symbol.to_string()));
        }
        
        // Need to drop the write guard before getting again - use a different approach
        drop(books);
        
        let books = self.order_books.write();
        // This is a simplified approach - in production would use entry API properly
        panic!("Use submit_order which handles this correctly")
    }

    /// Submit order to matching engine
    pub fn submit_order(&self, mut order: Order) -> MatchResult {
        let symbol = order.symbol.clone();
        let mut result = MatchResult {
            trades: Vec::new(),
            filled_orders: Vec::new(),
            partial_orders: Vec::new(),
            rejected_orders: Vec::new(),
        };

        // Validate order
        if order.quantity == 0 {
            order.status = OrderStatus::Rejected;
            result.rejected_orders.push(order.order_id);
            self.orders.write().insert(order.order_id, order);
            return result;
        }

        // Post-only validation
        if order.order_type == OrderType::PostOnly {
            if let Some(book) = self.order_books.read().get(&symbol) {
                let would_cross = match order.side {
                    Side::Buy => {
                        if let Some(ask) = book.best_ask() {
                            order.price.map_or(false, |p| p >= ask)
                        } else {
                            false
                        }
                    }
                    Side::Sell => {
                        if let Some(bid) = book.best_bid() {
                            order.price.map_or(false, |p| p <= bid)
                        } else {
                            false
                        }
                    }
                };

                if would_cross {
                    order.status = OrderStatus::Cancelled;
                    result.partial_orders.push(order.order_id);
                    self.orders.write().insert(order.order_id, order);
                    return result;
                }
            }
        }

        // Match against book
        let (trades, remaining_qty) = self.match_order(&mut order);
        
        // Record trades
        for trade in &trades {
            result.trades.push(trade.clone());
            self.trades.lock().push(trade.clone());
        }

        // Update order status
        if order.remaining_quantity == 0 {
            order.status = OrderStatus::Filled;
            order.fill_timestamp_ns = Some(get_timestamp_ns());
            result.filled_orders.push(order.order_id);
        } else if order.remaining_quantity < order.quantity {
            order.status = OrderStatus::PartiallyFilled;
            result.partial_orders.push(order.order_id);
        } else {
            // Add to book if limit order with remaining quantity
            if order.order_type == OrderType::Limit || order.order_type == OrderType::PostOnly {
                self.add_to_book(&symbol, &order);
            }
            result.partial_orders.push(order.order_id);
        }

        self.orders.write().insert(order.order_id, order);
        result
    }

    /// Match order against order book
    fn match_order(&self, order: &mut Order) -> (Vec<Trade>, Quantity) {
        let mut trades = Vec::new();
        let mut remaining = order.remaining_quantity;
        let symbol = order.symbol.clone();

        // Get opposite side of book
        let opposite_side = match order.side {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        };

        // Lock management for matching
        {
            let mut books = self.order_books.write();
            let book = books.get_mut(&symbol).unwrap();

            let aggressor_side = order.side;
            let order_price = order.price.unwrap_or(i64::MAX);

            while remaining > 0 {
                // Get best price from opposite side
                let best_price = match opposite_side {
                    Side::Buy => book.bids.best_price(),
                    Side::Sell => book.asks.best_price(),
                };

                match best_price {
                    Some(bp) => {
                        // Check if prices cross
                        let should_match = match order.side {
                            Side::Buy => order_price >= bp,
                            Side::Sell => order_price <= bp,
                        };

                        if !should_match {
                            break;
                        }

                        // Get the level to match against
                        let level = match opposite_side {
                            Side::Buy => book.bids.levels.get_mut(&bp),
                            Side::Sell => book.asks.levels.get_mut(&bp),
                        };

                        if let Some(level) = level {
                            let fill_qty = remaining.min(level.quantity);
                            let fill_price = bp;

                            // Calculate fee
                            let fee = ((fill_qty as u128 * fill_price as u128 * self.fee_rate_bps as u128) / 10000) as u64;
                            let fee = fee.max(self.min_fee);

                            // Create trade
                            let trade = Trade {
                                trade_id: generate_order_id(),
                                order_id: order.order_id,
                                symbol: symbol.clone(),
                                side: aggressor_side,
                                price: fill_price,
                                quantity: fill_qty,
                                maker: false, // Aggressor is taker
                                fee,
                                timestamp_ns: get_timestamp_ns(),
                                liquidity_taken: fill_qty,
                            };
                            trades.push(trade);

                            // Update order
                            order.filled_quantity += fill_qty;
                            order.remaining_quantity -= fill_qty;
                            remaining -= fill_qty;

                            // Update average fill price
                            order.average_fill_price = Some(
                                ((order.filled_quantity as u128 * order.average_fill_price.unwrap_or(0) as u128
                                    + fill_qty as u128 * fill_price as u128)
                                    / order.filled_quantity as u128) as Price
                            );

                            order.fees_paid += fee;

                            // Remove filled quantity from level
                            level.quantity -= fill_qty;
                            if level.quantity == 0 {
                                match opposite_side {
                                    Side::Buy => {
                                        book.bids.levels.remove(&bp);
                                        book.bids.total_quantity = book.bids.total_quantity.saturating_sub(fill_qty);
                                    }
                                    Side::Sell => {
                                        book.asks.levels.remove(&bp);
                                        book.asks.total_quantity = book.asks.total_quantity.saturating_sub(fill_qty);
                                    }
                                }
                            }

                            // Update last trade price
                            book.last_trade_price = Some(fill_price);
                            book.update_derived_values();
                        } else {
                            break;
                        }
                    }
                    None => break,
                }
            }
        }

        (trades, remaining)
    }

    /// Add order to order book
    fn add_to_book(&self, symbol: &str, order: &Order) {
        let mut books = self.order_books.write();
        let book = books.get_mut(symbol).unwrap();

        // Calculate queue position
        let queue_pos = match order.side {
            Side::Buy => {
                let mut pos = 1u32;
                for (_, level) in book.bids.levels.range(..=order.price.unwrap()).rev() {
                    if level.price == order.price.unwrap() {
                        pos += level.order_count;
                    }
                }
                pos
            }
            Side::Sell => {
                let mut pos = 1u32;
                for (_, level) in book.asks.levels.range(order.price.unwrap()..) {
                    if level.price == order.price.unwrap() {
                        pos += level.order_count;
                    }
                }
                pos
            }
        };

        // Clone order with queue position
        let mut order_copy = order.clone();
        order_copy.queue_position = Some(queue_pos);

        match order.side {
            Side::Buy => book.bids.add_order(&order_copy),
            Side::Sell => book.asks.add_order(&order_copy),
        }

        book.update_derived_values();
    }

    /// Cancel order
    pub fn cancel_order(&self, order_id: OrderId) -> bool {
        let mut orders = self.orders.write();
        
        if let Some(order) = orders.get_mut(&order_id) {
            if order.status == OrderStatus::New || order.status == OrderStatus::PartiallyFilled {
                let symbol = order.symbol.clone();
                let remaining = order.remaining_quantity;

                // Remove from book
                {
                    let mut books = self.order_books.write();
                    if let Some(book) = books.get_mut(&symbol) {
                        match order.side {
                            Side::Buy => book.bids.remove_order(order),
                            Side::Sell => book.asks.remove_order(order),
                        }
                        book.update_derived_values();
                    }
                }

                order.status = OrderStatus::Cancelled;
                order.remaining_quantity = 0;
                return true;
            }
        }
        false
    }

    /// Get order by ID
    pub fn get_order(&self, order_id: OrderId) -> Option<Order> {
        self.orders.read().get(&order_id).cloned()
    }

    /// Get order book snapshot
    pub fn get_order_book(&self, symbol: &str) -> Option<OrderBook> {
        self.order_books.read().get(symbol).cloned()
    }

    /// Get recent trades
    pub fn get_recent_trades(&self, symbol: &str, limit: usize) -> Vec<Trade> {
        let trades = self.trades.lock();
        trades
            .iter()
            .filter(|t| t.symbol == symbol)
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
}

// ============================================================================
// Latency Simulator
// ============================================================================

/// Network latency model
#[derive(Debug, Clone)]
pub struct LatencyModel {
    /// Base latency in microseconds
    pub base_latency_us: u64,
    /// Jitter standard deviation in microseconds
    pub jitter_std_us: u64,
    /// Probability of packet loss (0.0 to 1.0)
    pub packet_loss_prob: f64,
    /// Probability of reordering (0.0 to 1.0)
    pub reorder_prob: f64,
}

impl Default for LatencyModel {
    fn default() -> Self {
        Self {
            base_latency_us: 100,      // 100μs base
            jitter_std_us: 50,         // ±50μs jitter
            packet_loss_prob: 0.001,   // 0.1% loss
            reorder_prob: 0.01,        // 1% reorder
        }
    }
}

impl LatencyModel {
    /// Simulate latency for a message
    pub fn simulate_latency(&self) -> Duration {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Base + Gaussian jitter
        let jitter = rng.gen_range(-self.jitter_std_us as i64..=self.jitter_std_us as i64);
        let total_us = (self.base_latency_us as i64 + jitter).max(10) as u64;

        Duration::from_micros(total_us)
    }

    /// Check if packet is lost
    pub fn is_packet_lost(&self) -> bool {
        use rand::Rng;
        rand::thread_rng().gen_bool(self.packet_loss_prob)
    }
}

// ============================================================================
// Backtest Engine
// ============================================================================

/// Backtest configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub initial_capital: u64,
    pub symbols: Vec<String>,
    pub latency_model: LatencyModel,
    pub fee_rate_bps: u32,
    pub max_position_size: Quantity,
    pub enable_shorting: bool,
}

/// Portfolio state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioState {
    pub cash: u64,
    pub positions: std::collections::HashMap<String, i64>, // Positive = long, negative = short
    pub unrealized_pnl: i64,
    pub realized_pnl: i64,
    pub total_pnl: i64,
    pub peak_equity: u64,
    pub current_equity: u64,
    pub max_drawdown: u64,
}

/// Backtest event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BacktestEvent {
    Tick {
        symbol: String,
        timestamp_ns: TimestampNs,
        bid: Price,
        ask: Price,
        last: Price,
        volume: Quantity,
    },
    OrderSubmitted {
        order: Order,
    },
    OrderFilled {
        trade: Trade,
    },
    OrderCancelled {
        order_id: OrderId,
    },
    Signal {
        symbol: String,
        timestamp_ns: TimestampNs,
        signal_type: String,
        signal_value: f64,
    },
}

/// Main backtest engine
pub struct BacktestEngine {
    config: BacktestConfig,
    matching_engine: MatchingEngine,
    portfolio: RwLock<PortfolioState>,
    event_queue: Mutex<VecDeque<BacktestEvent>>,
    events_processed: AtomicU64,
    current_time_ns: AtomicU64,
    is_running: AtomicU64,
}

impl BacktestEngine {
    pub fn new(config: BacktestConfig) -> Self {
        let initial_capital = config.initial_capital;
        
        Self {
            config,
            matching_engine: MatchingEngine::new(config.fee_rate_bps, 1),
            portfolio: RwLock::new(PortfolioState {
                cash: initial_capital,
                positions: std::collections::HashMap::new(),
                unrealized_pnl: 0,
                realized_pnl: 0,
                total_pnl: 0,
                peak_equity: initial_capital,
                current_equity: initial_capital,
                max_drawdown: 0,
            }),
            event_queue: Mutex::new(VecDeque::with_capacity(10000)),
            events_processed: AtomicU64::new(0),
            current_time_ns: AtomicU64::new(0),
            is_running: AtomicU64::new(0),
        }
    }

    /// Queue an event for processing
    pub fn queue_event(&self, event: BacktestEvent) {
        self.event_queue.lock().push_back(event);
    }

    /// Process next event in queue
    pub fn process_next_event(&self) -> Option<BacktestEvent> {
        let mut queue = self.event_queue.lock();
        if let Some(event) = queue.pop_front() {
            self.events_processed.fetch_add(1, Ordering::Relaxed);
            
            // Update current time
            if let BacktestEvent::Tick { timestamp_ns, .. } = &event {
                self.current_time_ns.store(*timestamp_ns, Ordering::Relaxed);
            }
            
            Some(event)
        } else {
            None
        }
    }

    /// Run backtest loop
    pub fn run(&self) -> BacktestResult {
        self.is_running.store(1, Ordering::Relaxed);
        let start = Instant::now();

        while self.is_running.load(Ordering::Relaxed) == 1 {
            if let Some(event) = self.process_next_event() {
                self.handle_event(event);
            } else {
                // No more events
                break;
            }
        }

        let elapsed = start.elapsed();
        self.create_result(elapsed)
    }

    /// Handle individual event
    fn handle_event(&self, event: BacktestEvent) {
        match event {
            BacktestEvent::Tick { symbol, bid, ask, last, volume, .. } => {
                // Update portfolio valuation
                self.update_portfolio_valuation(&symbol, last);
            }
            BacktestEvent::OrderSubmitted { order } => {
                // Apply latency
                let latency = self.config.latency_model.simulate_latency();
                std::thread::sleep(latency);

                // Submit to matching engine
                let result = self.matching_engine.submit_order(order);
                
                // Handle fills
                for trade in result.trades {
                    self.handle_trade_fill(&trade);
                }
            }
            BacktestEvent::OrderCancelled { order_id } => {
                self.matching_engine.cancel_order(order_id);
            }
            _ => {}
        }
    }

    /// Handle trade fill
    fn handle_trade_fill(&self, trade: &Trade) {
        let mut portfolio = self.portfolio.write();
        
        let position = portfolio.positions.entry(trade.symbol.clone()).or_insert(0);
        
        match trade.side {
            Side::Buy => {
                *position += trade.quantity as i64;
                portfolio.cash = portfolio.cash.saturating_sub(
                    (trade.quantity as u128 * trade.price as u128 + trade.fee as u128) as u64
                );
            }
            Side::Sell => {
                *position -= trade.quantity as i64;
                portfolio.cash += (trade.quantity as u128 * trade.price as u128 - trade.fee as u128) as u64;
            }
        }

        // Update realized PnL (simplified)
        portfolio.realized_pnl += trade.fee as i64; // Fees are cost
        
        self.update_equity(&mut portfolio);
    }

    /// Update portfolio valuation with latest price
    fn update_portfolio_valuation(&self, symbol: &str, price: Price) {
        let mut portfolio = self.portfolio.write();
        
        let position = portfolio.positions.get(symbol).copied().unwrap_or(0);
        let unrealized = (position as i128 * price as i128) as i64;
        
        portfolio.unrealized_pnl = unrealized;
        self.update_equity(&mut portfolio);
    }

    /// Update equity and drawdown metrics
    fn update_equity(&self, portfolio: &mut PortfolioState) {
        portfolio.current_equity = portfolio.cash.saturating_add(portfolio.unrealized_pnl as u64);
        portfolio.total_pnl = portfolio.realized_pnl + portfolio.unrealized_pnl;

        if portfolio.current_equity > portfolio.peak_equity {
            portfolio.peak_equity = portfolio.current_equity;
        }

        let drawdown = portfolio.peak_equity.saturating_sub(portfolio.current_equity);
        if drawdown > portfolio.max_drawdown {
            portfolio.max_drawdown = drawdown;
        }
    }

    /// Create final backtest result
    fn create_result(&self, elapsed: Duration) -> BacktestResult {
        let portfolio = self.portfolio.read();
        
        BacktestResult {
            total_events: self.events_processed.load(Ordering::Relaxed),
            elapsed_duration: elapsed,
            events_per_second: self.events_processed.load(Ordering::Relaxed) as f64 / elapsed.as_secs_f64(),
            final_cash: portfolio.cash,
            final_equity: portfolio.current_equity,
            total_pnl: portfolio.total_pnl,
            realized_pnl: portfolio.realized_pnl,
            unrealized_pnl: portfolio.unrealized_pnl,
            max_drawdown: portfolio.max_drawdown,
            peak_equity: portfolio.peak_equity,
            return_pct: ((portfolio.current_equity as f64 - self.config.initial_capital as f64) 
                / self.config.initial_capital as f64) * 100.0,
        }
    }

    /// Stop the backtest
    pub fn stop(&self) {
        self.is_running.store(0, Ordering::Relaxed);
    }
}

/// Backtest results summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub total_events: u64,
    pub elapsed_duration: Duration,
    pub events_per_second: f64,
    pub final_cash: u64,
    pub final_equity: u64,
    pub total_pnl: i64,
    pub realized_pnl: i64,
    pub unrealized_pnl: i64,
    pub max_drawdown: u64,
    pub peak_equity: u64,
    pub return_pct: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matching_engine_basic() {
        let engine = MatchingEngine::new(10, 1); // 10 bps, $1 min fee

        // Add liquidity (sell order)
        let sell_order = Order::new_limit(
            "BTCUSD".to_string(),
            Side::Sell,
            50000,
            100,
            None,
        );
        engine.submit_order(sell_order);

        // Take liquidity (buy order)
        let buy_order = Order::new_limit(
            "BTCUSD".to_string(),
            Side::Buy,
            50000,
            50,
            None,
        );
        let result = engine.submit_order(buy_order);

        assert_eq!(result.trades.len(), 1);
        assert_eq!(result.trades[0].price, 50000);
        assert_eq!(result.trades[0].quantity, 50);
    }

    #[test]
    fn test_order_book_vwap() {
        let mut book = OrderBook::new("ETHUSD".to_string());

        // Add multiple ask levels
        book.asks.levels.insert(3000, PriceLevel {
            price: 3000,
            quantity: 100,
            order_count: 1,
            orders: vec![1],
        });
        book.asks.levels.insert(3001, PriceLevel {
            price: 3001,
            quantity: 200,
            order_count: 1,
            orders: vec![2],
        });
        book.asks.levels.insert(3002, PriceLevel {
            price: 3002,
            quantity: 150,
            order_count: 1,
            orders: vec![3],
        });
        book.asks.total_quantity = 450;

        // Calculate VWAP for 250 units
        let vwap = book.get_vwap(Side::Buy, 250);
        assert!(vwap.is_some());
        
        // Expected: (100*3000 + 150*3001) / 250 = 3000.6
        let expected = 3000; // Integer division
        assert_eq!(vwap.unwrap(), expected);
    }

    #[test]
    fn test_latency_simulation() {
        let model = LatencyModel {
            base_latency_us: 100,
            jitter_std_us: 10,
            packet_loss_prob: 0.0,
            reorder_prob: 0.0,
        };

        let latency = model.simulate_latency();
        
        // Should be around 100μs ± some jitter
        assert!(latency.as_micros() >= 90);
        assert!(latency.as_micros() <= 110);
    }
}
