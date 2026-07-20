// core/src/execution/order_manager.rs
// =============================================================================
// STAGE 2 - CHAPTER 4 - FILE 2
// Focus: Lock-free order lifecycle state machine with MPSC/SPSC queues.
// Handles order states without mutex contention for microsecond execution.
// =============================================================================

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::Arc;
use crossbeam::channel::{bounded, Sender, Receiver, TrySendError};
use std::time::{SystemTime, UNIX_EPOCH};

/// Order states in the lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OrderState {
    /// Order created locally, not yet sent
    Pending = 0,
    /// Order sent to exchange, awaiting acknowledgment
    Submitted = 1,
    /// Order acknowledged by exchange
    Acknowledged = 2,
    /// Partially filled
    PartiallyFilled = 3,
    /// Fully filled
    Filled = 4,
    /// Cancelled by user
    Cancelled = 5,
    /// Cancelled by exchange (rejected, expired)
    Rejected = 6,
}

impl OrderState {
    /// Check if order is in a terminal state
    #[inline]
    pub fn is_terminal(&self) -> bool {
        matches!(self, OrderState::Filled | OrderState::Cancelled | OrderState::Rejected)
    }

    /// Check if order can be modified
    #[inline]
    pub fn is_modifiable(&self) -> bool {
        matches!(
            self,
            OrderState::Pending | OrderState::Submitted | OrderState::Acknowledged | OrderState::PartiallyFilled
        )
    }
}

/// Order side (buy/sell)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OrderSide {
    Buy = 0,
    Sell = 1,
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OrderType {
    Market = 0,
    Limit = 1,
    StopLoss = 2,
    StopLossLimit = 3,
    TakeProfit = 4,
    TakeProfitLimit = 5,
}

/// Time in force for limit orders
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TimeInForce {
    GoodTillCancelled = 0,
    ImmediateOrCancel = 1,
    FillOrKill = 2,
}

/// Compact order representation optimized for cache locality
#[repr(C)]
pub struct Order {
    /// Unique order ID (client-assigned)
    pub client_order_id: u64,
    /// Exchange order ID (assigned after submission)
    pub exchange_order_id: AtomicU64,
    /// Symbol hash for fast lookup
    pub symbol_hash: u64,
    /// Order side
    pub side: OrderSide,
    /// Order type
    pub order_type: OrderType,
    /// Time in force
    pub time_in_force: TimeInForce,
    /// Order quantity
    pub quantity: f64,
    /// Order price (0 for market orders)
    pub price: f64,
    /// Filled quantity
    pub filled_quantity: AtomicU64, // Stored as scaled integer
    /// Current state
    pub state: AtomicUsize, // Stores OrderState as usize
    /// Creation timestamp (microseconds)
    pub created_at: u64,
    /// Last update timestamp
    pub updated_at: AtomicU64,
    /// Number of fills
    pub fill_count: AtomicUsize,
}

impl Order {
    /// Create a new order
    pub fn new(
        client_order_id: u64,
        symbol_hash: u64,
        side: OrderSide,
        order_type: OrderType,
        quantity: f64,
        price: f64,
        time_in_force: TimeInForce,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        Order {
            client_order_id,
            exchange_order_id: AtomicU64::new(0),
            symbol_hash,
            side,
            order_type,
            time_in_force,
            quantity,
            price,
            filled_quantity: AtomicU64::new(0),
            state: AtomicUsize::new(OrderState::Pending as usize),
            created_at: now,
            updated_at: AtomicU64::new(now),
            fill_count: AtomicUsize::new(0),
        }
    }

    /// Get current state
    #[inline]
    pub fn get_state(&self) -> OrderState {
        match self.state.load(AtomicOrdering::Acquire) {
            0 => OrderState::Pending,
            1 => OrderState::Submitted,
            2 => OrderState::Acknowledged,
            3 => OrderState::PartiallyFilled,
            4 => OrderState::Filled,
            5 => OrderState::Cancelled,
            6 => OrderState::Rejected,
            _ => unreachable!(),
        }
    }

    /// Update order state atomically
    /// Returns true if transition was valid, false otherwise
    #[inline]
    pub fn transition_state(&self, new_state: OrderState) -> bool {
        let current = self.state.load(AtomicOrdering::Acquire);
        
        // Validate state transition
        if !Self::is_valid_transition(current as u8, new_state as u8) {
            return false;
        }

        self.state.store(new_state as usize, AtomicOrdering::Release);
        self.updated_at.store(
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64,
            AtomicOrdering::Release,
        );
        true
    }

    /// Validate state transition
    fn is_valid_transition(from: u8, to: u8) -> bool {
        match from {
            0 => true, // Pending can go anywhere
            1 => matches!(to, 2..=6), // Submitted can go to Ack/Partial/Fill/Cancel/Reject
            2 => matches!(to, 3..=6), // Acknowledged can go to Partial/Fill/Cancel/Reject
            3 => matches!(to, 4..=5), // PartiallyFilled can go to Fill/Cancel
            _ => false, // Terminal states cannot transition
        }
    }

    /// Update filled quantity
    #[inline]
    pub fn add_fill(&self, fill_qty: f64, exchange_order_id: u64) {
        // Convert to scaled integer (assuming 8 decimal places)
        let fill_scaled = (fill_qty * 100_000_000.0) as u64;
        self.filled_quantity.fetch_add(fill_scaled, AtomicOrdering::Relaxed);
        self.fill_count.fetch_add(1, AtomicOrdering::Relaxed);
        
        if exchange_order_id > 0 {
            self.exchange_order_id.store(exchange_order_id, AtomicOrdering::Release);
        }

        // Update state based on fill
        let current_filled = self.filled_quantity.load(AtomicOrdering::Acquire) as f64 / 100_000_000.0;
        if current_filled >= self.quantity {
            self.transition_state(OrderState::Filled);
        } else if current_filled > 0.0 {
            self.transition_state(OrderState::PartiallyFilled);
        }
    }

    /// Get remaining quantity
    #[inline]
    pub fn remaining_quantity(&self) -> f64 {
        let filled = self.filled_quantity.load(AtomicOrdering::Acquire) as f64 / 100_000_000.0;
        self.quantity - filled
    }

    /// Check if order is fully filled
    #[inline]
    pub fn is_filled(&self) -> bool {
        self.get_state() == OrderState::Filled
    }

    /// Check if order is active
    #[inline]
    pub fn is_active(&self) -> bool {
        !self.get_state().is_terminal()
    }
}

/// Order event types for the event queue
#[derive(Debug, Clone)]
pub enum OrderEvent {
    /// Order submitted to exchange
    Submitted { order_id: u64 },
    /// Order acknowledged by exchange
    Acknowledged { order_id: u64, exchange_id: u64 },
    /// Order partially filled
    PartialFill {
        order_id: u64,
        fill_qty: f64,
        fill_price: f64,
        commission: f64,
    },
    /// Order fully filled
    Filled {
        order_id: u64,
        total_qty: f64,
        avg_price: f64,
    },
    /// Order cancelled
    Cancelled { order_id: u64, reason: String },
    /// Order rejected
    Rejected { order_id: u64, reason: String },
}

/// Lock-free order manager using SPSC/MPSC channels
pub struct OrderManager {
    /// Active orders stored in Arc for lock-free access
    orders: dashmap::DashMap<u64, Arc<Order>>,
    /// Event sender for order updates
    event_tx: Sender<OrderEvent>,
    /// Event receiver for order updates
    event_rx: Receiver<OrderEvent>,
    /// Order counter for generating unique IDs
    order_counter: AtomicU64,
    /// Statistics
    total_orders: AtomicU64,
    active_orders: AtomicUsize,
    filled_orders: AtomicU64,
    cancelled_orders: AtomicU64,
}

impl OrderManager {
    /// Create a new order manager with specified channel capacity
    pub fn new(channel_capacity: usize) -> Self {
        let (event_tx, event_rx) = bounded(channel_capacity);

        OrderManager {
            orders: dashmap::DashMap::new(),
            event_tx,
            event_rx,
            order_counter: AtomicU64::new(0),
            total_orders: AtomicU64::new(0),
            active_orders: AtomicUsize::new(0),
            filled_orders: AtomicU64::new(0),
            cancelled_orders: AtomicU64::new(0),
        }
    }

    /// Generate a unique client order ID
    #[inline]
    pub fn generate_order_id(&self) -> u64 {
        self.order_counter.fetch_add(1, AtomicOrdering::Relaxed)
    }

    /// Create and register a new order
    pub fn create_order(
        &self,
        symbol_hash: u64,
        side: OrderSide,
        order_type: OrderType,
        quantity: f64,
        price: f64,
        time_in_force: TimeInForce,
    ) -> Arc<Order> {
        let order_id = self.generate_order_id();
        
        let order = Arc::new(Order::new(
            order_id,
            symbol_hash,
            side,
            order_type,
            quantity,
            price,
            time_in_force,
        ));

        self.orders.insert(order_id, Arc::clone(&order));
        self.total_orders.fetch_add(1, AtomicOrdering::Relaxed);
        self.active_orders.fetch_add(1, AtomicOrdering::Relaxed);

        order
    }

    /// Get an order by ID
    pub fn get_order(&self, order_id: u64) -> Option<Arc<Order>> {
        self.orders.get(&order_id).map(|r| Arc::clone(r.value()))
    }

    /// Submit order (transition to Submitted state)
    pub fn submit_order(&self, order_id: u64) -> Result<(), &'static str> {
        let order = self.get_order(order_id).ok_or("Order not found")?;
        
        if order.transition_state(OrderState::Submitted) {
            self.event_tx
                .try_send(OrderEvent::Submitted { order_id })
                .map_err(|_| "Event queue full")?;
            Ok(())
        } else {
            Err("Invalid state transition")
        }
    }

    /// Process order acknowledgment from exchange
    pub fn acknowledge_order(&self, order_id: u64, exchange_id: u64) -> Result<(), &'static str> {
        let order = self.get_order(order_id).ok_or("Order not found")?;
        
        if order.transition_state(OrderState::Acknowledged) {
            order.exchange_order_id.store(exchange_id, AtomicOrdering::Release);
            self.event_tx
                .try_send(OrderEvent::Acknowledged { order_id, exchange_id })
                .map_err(|_| "Event queue full")?;
            Ok(())
        } else {
            Err("Invalid state transition")
        }
    }

    /// Process a fill update
    pub fn process_fill(
        &self,
        order_id: u64,
        fill_qty: f64,
        fill_price: f64,
        commission: f64,
    ) -> Result<(), &'static str> {
        let order = self.get_order(order_id).ok_or("Order not found")?;
        
        let prev_state = order.get_state();
        order.add_fill(fill_qty, 0);
        let new_state = order.get_state();

        // Send appropriate event
        let event = match new_state {
            OrderState::Filled => OrderEvent::Filled {
                order_id,
                total_qty: order.quantity,
                avg_price: fill_price, // Simplified - would track average in production
            },
            _ => OrderEvent::PartialFill {
                order_id,
                fill_qty,
                fill_price,
                commission,
            },
        };

        self.event_tx.try_send(event).map_err(|_| "Event queue full")?;

        // Update statistics
        if new_state == OrderState::Filled && prev_state != OrderState::Filled {
            self.active_orders.fetch_sub(1, AtomicOrdering::Relaxed);
            self.filled_orders.fetch_add(1, AtomicOrdering::Relaxed);
        }

        Ok(())
    }

    /// Cancel an order
    pub fn cancel_order(&self, order_id: u64, reason: &str) -> Result<(), &'static str> {
        let order = self.get_order(order_id).ok_or("Order not found")?;
        
        if order.transition_state(OrderState::Cancelled) {
            self.event_tx
                .try_send(OrderEvent::Cancelled {
                    order_id,
                    reason: reason.to_string(),
                })
                .map_err(|_| "Event queue full")?;
            
            self.active_orders.fetch_sub(1, AtomicOrdering::Relaxed);
            self.cancelled_orders.fetch_add(1, AtomicOrdering::Relaxed);
            Ok(())
        } else {
            Err("Cannot cancel - order in terminal state")
        }
    }

    /// Get the event receiver for consuming order events
    pub fn get_event_receiver(&self) -> &Receiver<OrderEvent> {
        &self.event_rx
    }

    /// Get statistics
    pub fn get_stats(&self) -> OrderManagerStats {
        OrderManagerStats {
            total_orders: self.total_orders.load(AtomicOrdering::Acquire),
            active_orders: self.active_orders.load(AtomicOrdering::Acquire),
            filled_orders: self.filled_orders.load(AtomicOrdering::Acquire),
            cancelled_orders: self.cancelled_orders.load(AtomicOrdering::Acquire),
        }
    }
}

/// Order manager statistics
#[derive(Debug, Clone)]
pub struct OrderManagerStats {
    pub total_orders: u64,
    pub active_orders: usize,
    pub filled_orders: u64,
    pub cancelled_orders: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_lifecycle() {
        let manager = OrderManager::new(1000);
        
        // Create order
        let order = manager.create_order(
            12345,
            OrderSide::Buy,
            OrderType::Limit,
            0.1,
            50000.0,
            TimeInForce::GoodTillCancelled,
        );
        
        assert_eq!(order.get_state(), OrderState::Pending);
        
        // Submit
        manager.submit_order(order.client_order_id).unwrap();
        assert_eq!(order.get_state(), OrderState::Submitted);
        
        // Acknowledge
        manager.acknowledge_order(order.client_order_id, 99999).unwrap();
        assert_eq!(order.get_state(), OrderState::Acknowledged);
        
        // Partial fill
        manager.process_fill(order.client_order_id, 0.05, 50000.0, 0.1).unwrap();
        assert_eq!(order.get_state(), OrderState::PartiallyFilled);
        
        // Full fill
        manager.process_fill(order.client_order_id, 0.05, 50000.0, 0.1).unwrap();
        assert_eq!(order.get_state(), OrderState::Filled);
    }

    #[test]
    fn test_invalid_transitions() {
        let manager = OrderManager::new(1000);
        
        let order = manager.create_order(
            12345,
            OrderSide::Sell,
            OrderType::Market,
            0.1,
            0.0,
            TimeInForce::ImmediateOrCancel,
        );
        
        // Fill without submitting should fail state logic
        manager.submit_order(order.client_order_id).unwrap();
        manager.acknowledge_order(order.client_order_id, 1).unwrap();
        
        // Try to go back to pending (invalid)
        assert!(!order.transition_state(OrderState::Pending));
    }
}
