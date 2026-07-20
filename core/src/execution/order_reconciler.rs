//! order_reconciler.rs
//! ===================
//! Robust handling of partial fills, order rejections, and exchange disconnects.
//! Continuously polls the Binance REST API to reconcile local Rust order state
//! with actual exchange state, resolving any discrepancies instantly.
//!
//! Critical for maintaining accurate position tracking during high-frequency trading.

use std::time::{Duration, Instant};
use std::sync::Arc;
use parking_lot::RwLock;
use log::{info, debug, warn, error};
use tokio::time::sleep;

/// Order status as reported by exchange
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExchangeOrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Canceled,
    Rejected,
    Expired,
}

/// Local order state
#[derive(Debug, Clone)]
pub struct LocalOrder {
    pub order_id: String,
    pub client_order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: f64,
    pub price: Option<f64>,
    pub filled_qty: f64,
    pub remaining_qty: f64,
    pub avg_fill_price: f64,
    pub status: LocalOrderStatus,
    pub created_at: Instant,
    pub last_update: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Limit,
    Market,
    StopLimit,
    StopMarket,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalOrderStatus {
    Pending,
    Submitted,
    PartiallyFilled,
    Filled,
    Canceling,
    Canceled,
    Rejected,
    Unknown,
}

/// Order from exchange API response
#[derive(Debug, Clone)]
pub struct ExchangeOrder {
    pub order_id: i64,
    pub client_order_id: String,
    pub symbol: String,
    pub side: String,
    pub order_type: String,
    pub quantity: f64,
    pub price: f64,
    pub executed_qty: f64,
    pub cummulative_quote_qty: f64,
    pub status: String,
    pub time_in_force: String,
    pub update_time: u64,
}

/// Reconciliation result
#[derive(Debug)]
pub struct ReconciliationResult {
    pub order_id: String,
    pub is_matched: bool,
    pub discrepancy_type: Option<DiscrepancyType>,
    pub local_filled: f64,
    pub exchange_filled: f64,
    pub fill_difference: f64,
    pub action_taken: ReconciliationAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscrepancyType {
    FillMismatch,
    StatusMismatch,
    QuantityMismatch,
    PriceMismatch,
    MissingLocally,
    MissingOnExchange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconciliationAction {
    NoAction,
    UpdateLocalState,
    CancelOrder,
    AlertUser,
    EmergencyHalt,
}

/// Configuration for reconciler
#[derive(Debug, Clone)]
pub struct ReconcilerConfig {
    /// Polling interval in milliseconds
    pub poll_interval_ms: u64,
    /// Maximum acceptable fill difference before alert
    pub max_fill_tolerance: f64,
    /// Number of consecutive failures before emergency halt
    pub max_consecutive_failures: usize,
    /// Enable automatic state correction
    pub auto_correct: bool,
}

impl Default for ReconcilerConfig {
    fn default() -> Self {
        Self {
            poll_interval_ms: 100, // 100ms polling
            max_fill_tolerance: 0.0001, // 0.01% tolerance
            max_consecutive_failures: 5,
            auto_correct: true,
        }
    }
}

/// Statistics for reconciliation monitoring
#[derive(Debug, Default)]
pub struct ReconciliationStats {
    pub total_reconciliations: u64,
    pub discrepancies_found: u64,
    pub corrections_made: u64,
    pub api_failures: u64,
    pub consecutive_failures: usize,
    pub last_successful_reconciliation: Option<Instant>,
}

/// Order Reconciler - maintains consistency between local and exchange state
pub struct OrderReconciler {
    config: ReconcilerConfig,
    stats: RwLock<ReconciliationStats>,
    /// Local order cache
    local_orders: RwLock<std::collections::HashMap<String, LocalOrder>>,
    /// Orders pending reconciliation
    pending_reconciliation: RwLock<std::collections::HashSet<String>>,
    /// Callback for critical discrepancies
    on_critical_discrepancy: Option<Arc<dyn Fn(ReconciliationResult) + Send + Sync>>,
}

impl OrderReconciler {
    /// Create a new order reconciler
    pub fn new(config: ReconcilerConfig) -> Self {
        info!("Initializing OrderReconciler with config: {:?}", config);
        Self {
            config,
            stats: RwLock::new(ReconciliationStats::default()),
            local_orders: RwLock::new(std::collections::HashMap::new()),
            pending_reconciliation: RwLock::new(std::collections::HashSet::new()),
            on_critical_discrepancy: None,
        }
    }

    /// Set callback for critical discrepancies
    pub fn set_critical_callback<F>(&mut self, callback: F)
    where
        F: Fn(ReconciliationResult) + Send + Sync + 'static,
    {
        self.on_critical_discrepancy = Some(Arc::new(callback));
    }

    /// Add a local order to tracking
    pub fn track_order(&self, order: LocalOrder) {
        let mut orders = self.local_orders.write();
        orders.insert(order.order_id.clone(), order);
        
        let mut pending = self.pending_reconciliation.write();
        pending.insert(order.order_id.clone());
        
        debug!("Started tracking order: {}", order.order_id);
    }

    /// Remove an order from tracking (when fully filled or canceled)
    pub fn untrack_order(&self, order_id: &str) {
        let mut orders = self.local_orders.write();
        orders.remove(order_id);
        
        let mut pending = self.pending_reconciliation.write();
        pending.remove(order_id);
        
        debug!("Stopped tracking order: {}", order_id);
    }

    /// Reconcile a single order with exchange state
    pub async fn reconcile_order(
        &self,
        order_id: &str,
        exchange_order: Option<ExchangeOrder>,
    ) -> ReconciliationResult {
        let start_time = Instant::now();
        
        let mut stats = self.stats.write();
        stats.total_reconciliations += 1;

        // Get local order
        let local_order = {
            let orders = self.local_orders.read();
            orders.get(order_id).cloned()
        };

        let result = match (local_order, exchange_order) {
            (Some(local), Some(exchange)) => {
                self.reconcile_matched_orders(local, exchange)
            }
            (Some(local), None) => {
                // Order missing on exchange
                ReconciliationResult {
                    order_id: order_id.to_string(),
                    is_matched: false,
                    discrepancy_type: Some(DiscrepancyType::MissingOnExchange),
                    local_filled: local.filled_qty,
                    exchange_filled: 0.0,
                    fill_difference: local.filled_qty,
                    action_taken: if local.status == LocalOrderStatus::Submitted {
                        ReconciliationAction::AlertUser
                    } else {
                        ReconciliationAction::UpdateLocalState
                    },
                }
            }
            (None, Some(exchange)) => {
                // Order exists on exchange but not locally (orphaned)
                ReconciliationResult {
                    order_id: order_id.to_string(),
                    is_matched: false,
                    discrepancy_type: Some(DiscrepancyType::MissingLocally),
                    local_filled: 0.0,
                    exchange_filled: exchange.executed_qty,
                    fill_difference: exchange.executed_qty,
                    action_taken: ReconciliationAction::AlertUser,
                }
            }
            (None, None) => {
                // Neither exists - nothing to do
                ReconciliationResult {
                    order_id: order_id.to_string(),
                    is_matched: true,
                    discrepancy_type: None,
                    local_filled: 0.0,
                    exchange_filled: 0.0,
                    fill_difference: 0.0,
                    action_taken: ReconciliationAction::NoAction,
                }
            }
        };

        // Update statistics
        if result.discrepancy_type.is_some() {
            stats.discrepancies_found += 1;
        }

        if result.action_taken != ReconciliationAction::NoAction 
            && result.action_taken != ReconciliationAction::AlertUser 
        {
            stats.corrections_made += 1;
        }

        stats.last_successful_reconciliation = Some(Instant::now());
        stats.consecutive_failures = 0;

        debug!(
            "Reconciled order {} in {:.3}ms | Action: {:?}",
            order_id,
            start_time.elapsed().as_secs_f64() * 1000.0,
            result.action_taken
        );

        // Execute action if auto-correct enabled
        if self.config.auto_correct && result.action_taken != ReconciliationAction::NoAction {
            self.execute_correction(&result);
        }

        // Notify callback for critical issues
        if matches!(
            result.action_taken,
            ReconciliationAction::AlertUser | ReconciliationAction::EmergencyHalt
        ) {
            if let Some(callback) = &self.on_critical_discrepancy {
                callback(result.clone());
            }
        }

        // Remove from pending if reconciled successfully
        if result.is_matched && result.action_taken == ReconciliationAction::NoAction {
            let mut pending = self.pending_reconciliation.write();
            pending.remove(order_id);
        }

        result
    }

    /// Reconcile matched local and exchange orders
    fn reconcile_matched_orders(
        &self,
        mut local: LocalOrder,
        exchange: ExchangeOrder,
    ) -> ReconciliationResult {
        let exchange_filled = exchange.executed_qty;
        let local_filled = local.filled_qty;
        let fill_diff = (exchange_filled - local_filled).abs();

        // Parse exchange status
        let exchange_status = parse_exchange_status(&exchange.status);

        // Check for discrepancies
        let discrepancy_type = if fill_diff > self.config.max_fill_tolerance {
            Some(DiscrepancyType::FillMismatch)
        } else if exchange_status != map_status_to_local(exchange.status.as_str()) {
            Some(DiscrepancyType::StatusMismatch)
        } else if (exchange.quantity - local.quantity).abs() > self.config.max_fill_tolerance {
            Some(DiscrepancyType::QuantityMismatch)
        } else {
            None
        };

        // Determine action
        let action_taken = match discrepancy_type {
            Some(DiscrepancyType::FillMismatch) => {
                if fill_diff > local.quantity * 0.01 {
                    // More than 1% difference - alert
                    ReconciliationAction::AlertUser
                } else {
                    ReconciliationAction::UpdateLocalState
                }
            }
            Some(DiscrepancyType::StatusMismatch) => {
                if exchange_status == LocalOrderStatus::Canceled
                    || exchange_status == LocalOrderStatus::Rejected
                {
                    ReconciliationAction::UpdateLocalState
                } else {
                    ReconciliationAction::UpdateLocalState
                }
            }
            Some(_) => ReconciliationAction::AlertUser,
            None => ReconciliationAction::NoAction,
        };

        // Update local state if needed
        if matches!(action_taken, ReconciliationAction::UpdateLocalState) {
            local.filled_qty = exchange_filled;
            local.remaining_qty = exchange.quantity - exchange_filled;
            local.status = exchange_status;
            
            if exchange.cummulative_quote_qty > 0.0 && exchange_filled > 0.0 {
                local.avg_fill_price = exchange.cummulative_quote_qty / exchange_filled;
            }
            
            local.last_update = Instant::now();
            
            // Write back updated state
            let mut orders = self.local_orders.write();
            orders.insert(local.order_id.clone(), local);
        }

        ReconciliationResult {
            order_id: local.order_id,
            is_matched: discrepancy_type.is_none(),
            discrepancy_type,
            local_filled,
            exchange_filled,
            fill_difference: fill_diff,
            action_taken,
        }
    }

    /// Execute correction action
    fn execute_correction(&self, result: &ReconciliationResult) {
        match result.action_taken {
            ReconciliationAction::UpdateLocalState => {
                debug!("Corrected local state for order {}", result.order_id);
            }
            ReconciliationAction::CancelOrder => {
                warn!("Initiating cancel for order {} due to discrepancy", result.order_id);
                // In production, send cancel request to exchange
            }
            ReconciliationAction::AlertUser => {
                error!(
                    "Critical discrepancy for order {}: {:?} | Fill diff: {}",
                    result.order_id,
                    result.discrepancy_type,
                    result.fill_difference
                );
            }
            ReconciliationAction::EmergencyHalt => {
                error!("EMERGENCY HALT triggered by order {}", result.order_id);
                // In production, trigger system-wide halt
            }
            ReconciliationAction::NoAction => {}
        }
    }

    /// Run continuous reconciliation loop
    pub async fn run_reconciliation_loop(
        self: Arc<Self>,
        mut exchange_feed: impl FnMut(&str) -> Option<ExchangeOrder> + Send + 'static,
    ) {
        info!("Starting reconciliation loop");
        
        loop {
            sleep(Duration::from_millis(self.config.poll_interval_ms)).await;
            
            // Get orders needing reconciliation
            let order_ids: Vec<String> = {
                let pending = self.pending_reconciliation.read();
                pending.iter().cloned().collect()
            };
            
            for order_id in order_ids {
                let exchange_order = exchange_feed(&order_id);
                
                let result = self.reconcile_order(&order_id, exchange_order).await;
                
                // Handle API failure
                if exchange_order.is_none() {
                    let mut stats = self.stats.write();
                    stats.api_failures += 1;
                    stats.consecutive_failures += 1;
                    
                    if stats.consecutive_failures >= self.config.max_consecutive_failures {
                        error!(
                            "Consecutive API failures exceeded threshold ({})",
                            stats.consecutive_failures
                        );
                        // Could trigger emergency halt here
                    }
                }
                
                // Untrack completed orders
                if result.is_matched 
                    && (result.exchange_filled >= result.local_filled * 0.99)
                {
                    let orders = self.local_orders.read();
                    if let Some(order) = orders.get(&order_id) {
                        if order.status == LocalOrderStatus::Filled
                            || order.status == LocalOrderStatus::Canceled
                        {
                            drop(orders);
                            self.untrack_order(&order_id);
                        }
                    }
                }
            }
        }
    }

    /// Get current statistics
    pub fn get_statistics(&self) -> ReconciliationStats {
        self.stats.read().clone()
    }

    /// Get pending reconciliation count
    pub fn pending_count(&self) -> usize {
        self.pending_reconciliation.read().len()
    }
}

/// Parse exchange status string to enum
fn parse_exchange_status(status: &str) -> LocalOrderStatus {
    match status {
        "NEW" => LocalOrderStatus::Submitted,
        "PARTIALLY_FILLED" => LocalOrderStatus::PartiallyFilled,
        "FILLED" => LocalOrderStatus::Filled,
        "CANCELED" => LocalOrderStatus::Canceled,
        "REJECTED" => LocalOrderStatus::Rejected,
        "EXPIRED" => LocalOrderStatus::Canceled,
        _ => LocalOrderStatus::Unknown,
    }
}

/// Map exchange status to local status
fn map_status_to_local(status: &str) -> LocalOrderStatus {
    parse_exchange_status(status)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_exchange_status() {
        assert_eq!(parse_exchange_status("NEW"), LocalOrderStatus::Submitted);
        assert_eq!(parse_exchange_status("FILLED"), LocalOrderStatus::Filled);
        assert_eq!(parse_exchange_status("CANCELED"), LocalOrderStatus::Canceled);
    }

    #[test]
    fn test_reconciliation_result_creation() {
        let result = ReconciliationResult {
            order_id: "test".to_string(),
            is_matched: true,
            discrepancy_type: None,
            local_filled: 1.0,
            exchange_filled: 1.0,
            fill_difference: 0.0,
            action_taken: ReconciliationAction::NoAction,
        };

        assert!(result.is_matched);
        assert!(result.discrepancy_type.is_none());
        assert!((result.fill_difference - 0.0).abs() < 0.001);
    }
}
