// core/src/execution/smart_router.rs
// =============================================================================
// STAGE 2 - CHAPTER 4 - FILE 3
// Focus: Smart Order Routing (SOR) foundation with TWAP, VWAP, Iceberg algorithms.
// Minimizes market impact and slippage for large orders.
// =============================================================================

use super::order_manager::{Order, OrderManager, OrderSide, OrderType, TimeInForce};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Execution algorithm types
#[derive(Debug, Clone, Copy)]
pub enum ExecutionAlgo {
    /// Execute immediately at market
    Immediate,
    /// Time-Weighted Average Price
    TWAP {
        duration_seconds: u64,
        num_slices: u64,
    },
    /// Volume-Weighted Average Price
    VWAP {
        duration_seconds: u64,
        volume_profile: Vec<f64>, // Expected volume distribution
    },
    /// Iceberg order (hide true size)
    Iceberg {
        display_qty: f64,
        min_slice_size: f64,
    },
}

/// Order child for algorithmic execution
struct ChildOrder {
    parent_order_id: u64,
    child_sequence: u64,
    quantity: f64,
    price_limit: Option<f64>,
    status: ChildOrderStatus,
    created_at: Instant,
    executed_at: Option<Instant>,
    fill_qty: f64,
    fill_price: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ChildOrderStatus {
    Pending,
    Submitted,
    Filled,
    Cancelled,
    Failed,
}

/// Smart Order Router for executing large orders with minimal impact
pub struct SmartOrderRouter {
    order_manager: Arc<OrderManager>,
    /// Maximum slippage tolerance (basis points)
    max_slippage_bps: u64,
    /// Default participation rate (percentage of volume)
    default_participation_rate: f64,
    /// Active algorithm executions
    active_executions: dashmap::DashMap<u64, ExecutionState>,
    /// Execution counter
    execution_counter: AtomicU64,
    /// Emergency stop flag
    emergency_stop: AtomicBool,
}

/// State of an algorithmic execution
struct ExecutionState {
    algo_type: ExecutionAlgo,
    total_quantity: f64,
    remaining_quantity: f64,
    executed_quantity: f64,
    average_price: f64,
    child_orders: Vec<ChildOrder>,
    start_time: Instant,
    target_end_time: Option<Instant>,
    is_complete: bool,
}

impl SmartOrderRouter {
    /// Create a new smart order router
    pub fn new(order_manager: Arc<OrderManager>) -> Self {
        SmartOrderRouter {
            order_manager,
            max_slippage_bps: 10, // 0.1% default slippage tolerance
            default_participation_rate: 0.10, // 10% of volume
            active_executions: dashmap::DashMap::new(),
            execution_counter: AtomicU64::new(0),
            emergency_stop: AtomicBool::new(false),
        }
    }

    /// Set maximum slippage tolerance in basis points
    pub fn set_max_slippage(&mut self, slippage_bps: u64) {
        self.max_slippage_bps = slippage_bps;
    }

    /// Set default participation rate
    pub fn set_participation_rate(&mut self, rate: f64) {
        self.default_participation_rate = rate.clamp(0.01, 1.0);
    }

    /// Trigger emergency stop for all executions
    pub fn emergency_stop(&self) {
        self.emergency_stop.store(true, AtomicOrdering::Release);
        
        // Cancel all pending child orders
        for (_, state) in self.active_executions.iter() {
            for child in &state.child_orders {
                if child.status == ChildOrderStatus::Pending || child.status == ChildOrderStatus::Submitted {
                    // In production, would send cancel request
                }
            }
        }
    }

    /// Execute a TWAP order
    /// 
    /// Splits the order into equal slices over a specified duration
    pub fn execute_twap(
        &self,
        symbol_hash: u64,
        side: OrderSide,
        total_quantity: f64,
        duration_seconds: u64,
        num_slices: u64,
    ) -> Result<u64, &'static str> {
        if self.emergency_stop.load(AtomicOrdering::Acquire) {
            return Err("Emergency stop active");
        }

        let execution_id = self.execution_counter.fetch_add(1, AtomicOrdering::Relaxed);
        
        let algo = ExecutionAlgo::TWAP {
            duration_seconds,
            num_slices,
        };

        let slice_qty = total_quantity / num_slices as f64;
        let interval = Duration::from_secs(duration_seconds / num_slices);

        let state = ExecutionState {
            algo_type: algo,
            total_quantity,
            remaining_quantity: total_quantity,
            executed_quantity: 0.0,
            average_price: 0.0,
            child_orders: Vec::new(),
            start_time: Instant::now(),
            target_end_time: Some(Instant::now() + Duration::from_secs(duration_seconds)),
            is_complete: false,
        };

        self.active_executions.insert(execution_id, state);

        // Schedule first slice
        self.submit_twap_slice(execution_id, symbol_hash, side, slice_qty, 0)?;

        Ok(execution_id)
    }

    /// Submit a single TWAP slice
    fn submit_twap_slice(
        &self,
        execution_id: u64,
        symbol_hash: u64,
        side: OrderSide,
        quantity: f64,
        slice_num: u64,
    ) -> Result<(), &'static str> {
        let mut state = self.active_executions.get_mut(&execution_id).ok_or("Execution not found")?;
        
        if state.remaining_quantity <= 0.0 {
            state.is_complete = true;
            return Ok(());
        }

        // Adjust last slice to fill remaining
        let actual_qty = if slice_num == state.total_quantity as u64 - 1 {
            state.remaining_quantity
        } else {
            quantity.min(state.remaining_quantity)
        };

        // Create child order
        let child = ChildOrder {
            parent_order_id: execution_id,
            child_sequence: slice_num,
            quantity: actual_qty,
            price_limit: None, // Market order for TWAP
            status: ChildOrderStatus::Pending,
            created_at: Instant::now(),
            executed_at: None,
            fill_qty: 0.0,
            fill_price: 0.0,
        };

        // Create and submit order through order manager
        let order = self.order_manager.create_order(
            symbol_hash,
            side,
            OrderType::Market,
            actual_qty,
            0.0,
            TimeInForce::ImmediateOrCancel,
        );

        child.status = ChildOrderStatus::Submitted;
        state.child_orders.push(child);
        state.remaining_quantity -= actual_qty;

        // Submit the order
        self.order_manager.submit_order(order.client_order_id)?;

        Ok(())
    }

    /// Execute a VWAP order
    /// 
    /// Distributes orders according to expected volume profile
    pub fn execute_vwap(
        &self,
        symbol_hash: u64,
        side: OrderSide,
        total_quantity: f64,
        duration_seconds: u64,
        volume_profile: Vec<f64>,
    ) -> Result<u64, &'static str> {
        if self.emergency_stop.load(AtomicOrdering::Acquire) {
            return Err("Emergency stop active");
        }

        let execution_id = self.execution_counter.fetch_add(1, AtomicOrdering::Relaxed);

        let algo = ExecutionAlgo::VWAP {
            duration_seconds,
            volume_profile: volume_profile.clone(),
        };

        let state = ExecutionState {
            algo_type: algo,
            total_quantity,
            remaining_quantity: total_quantity,
            executed_quantity: 0.0,
            average_price: 0.0,
            child_orders: Vec::new(),
            start_time: Instant::now(),
            target_end_time: Some(Instant::now() + Duration::from_secs(duration_seconds)),
            is_complete: false,
        };

        self.active_executions.insert(execution_id, state);

        // Schedule first slice based on volume profile
        if !volume_profile.is_empty() {
            let first_slice_qty = total_quantity * volume_profile[0];
            self.submit_vwap_slice(execution_id, symbol_hash, side, first_slice_qty, 0, &volume_profile)?;
        }

        Ok(execution_id)
    }

    /// Submit a VWAP slice based on volume profile
    fn submit_vwap_slice(
        &self,
        execution_id: u64,
        symbol_hash: u64,
        side: OrderSide,
        quantity: f64,
        slice_num: usize,
        volume_profile: &[f64],
    ) -> Result<(), &'static str> {
        let mut state = self.active_executions.get_mut(&execution_id).ok_or("Execution not found")?;

        if state.remaining_quantity <= 0.0 {
            state.is_complete = true;
            return Ok(());
        }

        let actual_qty = quantity.min(state.remaining_quantity);

        let child = ChildOrder {
            parent_order_id: execution_id,
            child_sequence: slice_num as u64,
            quantity: actual_qty,
            price_limit: None,
            status: ChildOrderStatus::Pending,
            created_at: Instant::now(),
            executed_at: None,
            fill_qty: 0.0,
            fill_price: 0.0,
        };

        let order = self.order_manager.create_order(
            symbol_hash,
            side,
            OrderType::Limit, // Use limit orders for VWAP
            actual_qty,
            0.0, // Price would be set based on current market
            TimeInForce::GoodTillCancelled,
        );

        child.status = ChildOrderStatus::Submitted;
        state.child_orders.push(child);
        state.remaining_quantity -= actual_qty;

        self.order_manager.submit_order(order.client_order_id)?;

        Ok(())
    }

    /// Execute an Iceberg order
    /// 
    /// Shows only a portion of the total order size
    pub fn execute_iceberg(
        &self,
        symbol_hash: u64,
        side: OrderSide,
        total_quantity: f64,
        display_qty: f64,
        min_slice_size: f64,
        price: f64,
    ) -> Result<u64, &'static str> {
        if self.emergency_stop.load(AtomicOrdering::Acquire) {
            return Err("Emergency stop active");
        }

        if display_qty > total_quantity {
            return Err("Display quantity cannot exceed total quantity");
        }

        let execution_id = self.execution_counter.fetch_add(1, AtomicOrdering::Relaxed);

        let algo = ExecutionAlgo::Iceberg {
            display_qty,
            min_slice_size,
        };

        let state = ExecutionState {
            algo_type: algo,
            total_quantity,
            remaining_quantity: total_quantity,
            executed_quantity: 0.0,
            average_price: 0.0,
            child_orders: Vec::new(),
            start_time: Instant::now(),
            target_end_time: None,
            is_complete: false,
        };

        self.active_executions.insert(execution_id, state);

        // Submit initial visible slice
        self.submit_iceberg_slice(execution_id, symbol_hash, side, display_qty, price, min_slice_size)?;

        Ok(execution_id)
    }

    /// Submit an iceberg slice
    fn submit_iceberg_slice(
        &self,
        execution_id: u64,
        symbol_hash: u64,
        side: OrderSide,
        display_qty: f64,
        price: f64,
        min_slice_size: f64,
    ) -> Result<(), &'static str> {
        let mut state = self.active_executions.get_mut(&execution_id).ok_or("Execution not found")?;

        if state.remaining_quantity <= 0.0 {
            state.is_complete = true;
            return Ok(());
        }

        // Calculate slice size
        let slice_qty = display_qty.min(state.remaining_quantity).max(min_slice_size);

        let child = ChildOrder {
            parent_order_id: execution_id,
            child_sequence: state.child_orders.len() as u64,
            quantity: slice_qty,
            price_limit: Some(price),
            status: ChildOrderStatus::Pending,
            created_at: Instant::now(),
            executed_at: None,
            fill_qty: 0.0,
            fill_price: 0.0,
        };

        let order = self.order_manager.create_order(
            symbol_hash,
            side,
            OrderType::Limit,
            slice_qty,
            price,
            TimeInForce::GoodTillCancelled,
        );

        child.status = ChildOrderStatus::Submitted;
        state.child_orders.push(child);
        state.remaining_quantity -= slice_qty;

        self.order_manager.submit_order(order.client_order_id)?;

        Ok(())
    }

    /// Process a fill for an algorithmic execution
    pub fn process_execution_fill(
        &self,
        execution_id: u64,
        child_sequence: u64,
        fill_qty: f64,
        fill_price: f64,
    ) -> Result<(), &'static str> {
        let mut state = self.active_executions.get_mut(&execution_id).ok_or("Execution not found")?;

        // Update child order
        if let Some(child) = state.child_orders.get_mut(child_sequence as usize) {
            child.fill_qty = fill_qty;
            child.fill_price = fill_price;
            child.status = ChildOrderStatus::Filled;
            child.executed_at = Some(Instant::now());
        }

        // Update execution statistics
        state.executed_quantity += fill_qty;

        // Update average price
        let total_value = (state.average_price * (state.executed_quantity - fill_qty)) + (fill_price * fill_qty);
        state.average_price = total_value / state.executed_quantity;

        // Check if more slices needed
        match state.algo_type {
            ExecutionAlgo::TWAP { num_slices, .. } => {
                if state.child_orders.len() < num_slices as usize && state.remaining_quantity > 0.0 {
                    // Next slice will be scheduled by the event loop
                }
            }
            ExecutionAlgo::Iceberg { display_qty, min_slice_size } => {
                if state.remaining_quantity > 0.0 {
                    // Submit next iceberg slice
                    // In production, this would be triggered by order fill events
                }
            }
            _ => {}
        }

        // Check completion
        if state.remaining_quantity <= 0.0 || state.executed_quantity >= state.total_quantity * 0.999 {
            state.is_complete = true;
        }

        Ok(())
    }

    /// Get execution status
    pub fn get_execution_status(&self, execution_id: u64) -> Option<ExecutionStatus> {
        let state = self.active_executions.get(&execution_id)?;

        Some(ExecutionStatus {
            execution_id,
            algo_type: format!("{:?}", state.algo_type),
            total_quantity: state.total_quantity,
            executed_quantity: state.executed_quantity,
            remaining_quantity: state.remaining_quantity,
            average_price: state.average_price,
            progress_pct: (state.executed_quantity / state.total_quantity) * 100.0,
            is_complete: state.is_complete,
            elapsed_secs: state.start_time.elapsed().as_secs(),
        })
    }

    /// Clean up completed executions
    pub fn cleanup_completed(&self) -> usize {
        let mut cleaned = 0;
        
        self.active_executions.retain(|_, state| {
            if state.is_complete {
                cleaned += 1;
                false // Remove from map
            } else {
                true // Keep in map
            }
        });

        cleaned
    }
}

/// Execution status for external queries
#[derive(Debug, Clone)]
pub struct ExecutionStatus {
    pub execution_id: u64,
    pub algo_type: String,
    pub total_quantity: f64,
    pub executed_quantity: f64,
    pub remaining_quantity: f64,
    pub average_price: f64,
    pub progress_pct: f64,
    pub is_complete: bool,
    pub elapsed_secs: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_twap_execution() {
        let order_manager = Arc::new(OrderManager::new(1000));
        let router = SmartOrderRouter::new(Arc::clone(&order_manager));

        // Execute TWAP: 1.0 BTC over 60 seconds in 6 slices
        let execution_id = router.execute_twap(
            12345,
            OrderSide::Buy,
            1.0,
            60,
            6,
        ).unwrap();

        // Check status
        let status = router.get_execution_status(execution_id).unwrap();
        assert_eq!(status.total_quantity, 1.0);
        assert!(!status.is_complete);
        assert!(status.progress_pct >= 0.0);
    }

    #[test]
    fn test_iceberg_execution() {
        let order_manager = Arc::new(OrderManager::new(1000));
        let router = SmartOrderRouter::new(Arc::clone(&order_manager));

        // Execute Iceberg: 10.0 BTC total, show 0.5 at a time
        let execution_id = router.execute_iceberg(
            12345,
            OrderSide::Sell,
            10.0,
            0.5,
            0.1,
            50000.0,
        ).unwrap();

        let status = router.get_execution_status(execution_id).unwrap();
        assert_eq!(status.total_quantity, 10.0);
    }

    #[test]
    fn test_emergency_stop() {
        let order_manager = Arc::new(OrderManager::new(1000));
        let router = SmartOrderRouter::new(Arc::clone(&order_manager));

        router.emergency_stop();

        // Should fail after emergency stop
        let result = router.execute_twap(12345, OrderSide::Buy, 1.0, 60, 6);
        assert!(result.is_err());
    }
}
