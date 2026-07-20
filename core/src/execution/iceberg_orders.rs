//! Advanced Smart Order Router (SOR) - Iceberg Orders Implementation
//! 
//! This module implements sophisticated iceberg order logic that dynamically calculates
//! visible clip sizes based on order book depth and bid-ask spread to hide true parent
//! order size from the market and prevent front-running.
//!
//! Target Hardware: AMD Ryzen AI 5 + AMD Radeon GPU + 16GB RAM (8GB cap)

use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::{RwLock, Mutex};
use serde::{Serialize, Deserialize};
use log::{info, warn, debug, error};

use crate::execution::order_book::OrderBook;
use crate::types::{OrderId, Symbol, Side, Price, Quantity};

/// Configuration for iceberg order behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergConfig {
    /// Minimum clip size as percentage of average daily volume
    pub min_clip_pct_adv: f64,
    /// Maximum clip size as percentage of top-of-book quantity
    pub max_clip_pct_tob: f64,
    /// Dynamic adjustment sensitivity (0.0 - 1.0)
    pub sensitivity: f64,
    /// Minimum hidden quantity ratio (hidden/visible)
    pub min_hidden_ratio: f64,
    /// Maximum hidden quantity ratio
    pub max_hidden_ratio: f64,
    /// Refresh interval for clip recalculation
    pub refresh_interval_ms: u64,
    /// Aggressiveness level (0=passive, 1=aggressive)
    pub aggressiveness: f64,
}

impl Default for IcebergConfig {
    fn default() -> Self {
        Self {
            min_clip_pct_adv: 0.0001,      // 0.01% of ADV
            max_clip_pct_tob: 0.25,         // 25% of TOB quantity
            sensitivity: 0.5,               // Medium sensitivity
            min_hidden_ratio: 1.0,          // At least equal hidden/visible
            max_hidden_ratio: 10.0,         // At most 10x hidden/visible
            refresh_interval_ms: 100,       // Recalculate every 100ms
            aggressiveness: 0.3,            // Slightly passive
        }
    }
}

/// State of an active iceberg order
#[derive(Debug, Clone)]
pub struct IcebergState {
    /// Parent order ID
    pub parent_order_id: OrderId,
    /// Original total quantity
    pub total_quantity: Quantity,
    /// Remaining quantity to execute
    pub remaining_quantity: Quantity,
    /// Currently visible clip quantity
    pub current_clip_quantity: Quantity,
    /// Hidden quantity (remaining - clip)
    pub hidden_quantity: Quantity,
    /// Number of clips executed so far
    pub clips_executed: u32,
    /// Average fill price of executed clips
    pub avg_fill_price: Option<Price>,
    /// Last clip refresh time
    pub last_refresh: Instant,
    /// Order side
    pub side: Side,
    /// Limit price
    pub limit_price: Price,
}

impl IcebergState {
    pub fn new(
        parent_order_id: OrderId,
        total_quantity: Quantity,
        side: Side,
        limit_price: Price,
    ) -> Self {
        Self {
            parent_order_id,
            total_quantity,
            remaining_quantity: total_quantity,
            current_clip_quantity: 0.0,
            hidden_quantity: total_quantity,
            clips_executed: 0,
            avg_fill_price: None,
            last_refresh: Instant::now(),
            side,
            limit_price,
        }
    }

    /// Update state after a fill
    pub fn on_fill(&mut self, fill_qty: Quantity, fill_price: Price) {
        self.remaining_quantity = (self.remaining_quantity - fill_qty).max(0.0);
        self.hidden_quantity = self.remaining_quantity - self.current_clip_quantity;
        
        // Update average fill price
        let total_filled = self.total_quantity - self.remaining_quantity;
        if total_filled > 0.0 {
            match self.avg_fill_price {
                Some(current_avg) => {
                    let prev_filled = total_filled - fill_qty;
                    self.avg_fill_price = Some(
                        (current_avg * prev_filled + fill_price * fill_qty) / total_filled
                    );
                }
                None => {
                    self.avg_fill_price = Some(fill_price);
                }
            }
        }
    }

    /// Check if iceberg is complete
    pub fn is_complete(&self) -> bool {
        self.remaining_quantity <= 0.0
    }

    /// Get execution progress (0.0 - 1.0)
    pub fn progress(&self) -> f64 {
        1.0 - (self.remaining_quantity / self.total_quantity)
    }
}

/// Smart Order Router with Iceberg order support
pub struct IcebergRouter {
    config: IcebergConfig,
    /// Active iceberg orders
    icebergs: RwLock<Vec<IcebergState>>,
    /// Reference to order book for depth analysis
    order_book: Arc<RwLock<OrderBook>>,
    /// Average daily volume cache (updated periodically)
    adv_cache: RwLock<f64>,
    /// Last ADV update time
    adv_last_update: Mutex<Instant>,
}

impl IcebergRouter {
    pub fn new(config: IcebergConfig, order_book: Arc<RwLock<OrderBook>>) -> Self {
        Self {
            config,
            icebergs: RwLock::new(Vec::new()),
            order_book,
            adv_cache: RwLock::new(1_000_000.0), // Default 1M units
            adv_last_update: Mutex::new(Instant::now()),
        }
    }

    /// Create a new iceberg order
    pub fn create_iceberg(
        &self,
        parent_order_id: OrderId,
        total_quantity: Quantity,
        side: Side,
        limit_price: Price,
    ) -> IcebergState {
        let mut state = IcebergState::new(parent_order_id, total_quantity, side, limit_price);
        
        // Calculate initial clip size
        state.current_clip_quantity = self.calculate_clip_size(&state);
        state.hidden_quantity = state.remaining_quantity - state.current_clip_quantity;
        
        info!(
            "Created iceberg order {}: total={}, initial_clip={}, hidden={}",
            parent_order_id, total_quantity, state.current_clip_quantity, state.hidden_quantity
        );
        
        self.icebergs.write().push(state.clone());
        state
    }

    /// Calculate optimal clip size based on market conditions
    pub fn calculate_clip_size(&self, iceberg: &IcebergState) -> Quantity {
        let ob = self.order_book.read();
        let adv = *self.adv_cache.read();

        // Get relevant quantities based on side
        let (tob_qty, spread_bps) = match iceberg.side {
            Side::Buy => {
                let best_bid_qty = ob.best_bid_qty().unwrap_or(0.0);
                let best_ask = ob.best_ask().unwrap_or(iceberg.limit_price);
                let spread = (best_ask - iceberg.limit_price).max(0.0);
                let spread_bps = if iceberg.limit_price > 0.0 {
                    (spread / iceberg.limit_price) * 10000.0
                } else {
                    0.0
                };
                (best_bid_qty, spread_bps)
            }
            Side::Sell => {
                let best_ask_qty = ob.best_ask_qty().unwrap_or(0.0);
                let best_bid = ob.best_bid().unwrap_or(iceberg.limit_price);
                let spread = (iceberg.limit_price - best_bid).max(0.0);
                let spread_bps = if iceberg.limit_price > 0.0 {
                    (spread / iceberg.limit_price) * 10000.0
                } else {
                    0.0
                };
                (best_ask_qty, spread_bps)
            }
        };

        // Base clip size from ADV
        let base_clip = adv * self.config.min_clip_pct_adv;

        // Adjust based on TOB quantity
        let tob_adjusted = tob_qty * self.config.max_clip_pct_tob;

        // Spread penalty: reduce clip size when spread is wide
        let spread_factor = 1.0 - (spread_bps / 100.0).min(0.5).max(0.0);

        // Combine factors with sensitivity weighting
        let raw_clip = base_clip * 0.5 + tob_adjusted * 0.5;
        let adjusted_clip = raw_clip * spread_factor * (0.5 + self.config.sensitivity);

        // Apply aggressiveness modifier
        let aggressiveness_factor = 0.5 + self.config.aggressiveness * 0.5;
        let final_clip = adjusted_clip * aggressiveness_factor;

        // Ensure clip doesn't exceed remaining quantity
        let clipped = final_clip.min(iceberg.remaining_quantity);

        // Apply minimum threshold
        clipped.max(iceberg.remaining_quantity * 0.01) // At least 1% of remaining
    }

    /// Refresh clip size for an active iceberg
    pub fn refresh_clip(&self, parent_order_id: OrderId) -> Option<Quantity> {
        let mut icebergs = self.icebergs.write();
        
        if let Some(iceberg) = icebergs.iter_mut().find(|i| i.parent_order_id == parent_order_id) {
            if iceberg.is_complete() {
                return None;
            }

            let new_clip = self.calculate_clip_size(iceberg);
            
            // Only update if significant change (>10%)
            let change_pct = (new_clip - iceberg.current_clip_quantity).abs() 
                / iceberg.current_clip_quantity.max(0.001);
            
            if change_pct > 0.1 {
                let old_clip = iceberg.current_clip_quantity;
                iceberg.current_clip_quantity = new_clip;
                iceberg.hidden_quantity = iceberg.remaining_quantity - new_clip;
                iceberg.last_refresh = Instant::now();
                
                debug!(
                    "Refreshed iceberg {} clip: {} -> {}",
                    parent_order_id, old_clip, new_clip
                );
            }
            
            Some(new_clip)
        } else {
            None
        }
    }

    /// Notify router of a child order fill
    pub fn on_child_fill(
        &self,
        parent_order_id: OrderId,
        fill_qty: Quantity,
        fill_price: Price,
    ) -> Option<IcebergState> {
        let mut icebergs = self.icebergs.write();
        
        if let Some(iceberg) = icebergs.iter_mut().find(|i| i.parent_order_id == parent_order_id) {
            iceberg.on_fill(fill_qty, fill_price);
            iceberg.clips_executed += 1;
            
            let state = iceberg.clone();
            
            if iceberg.is_complete() {
                info!(
                    "Iceberg order {} complete: executed {} clips, avg_price={:?}",
                    parent_order_id, iceberg.clips_executed, iceberg.avg_fill_price
                );
            }
            
            Some(state)
        } else {
            None
        }
    }

    /// Get next child order parameters for an iceberg
    pub fn get_next_child_order(&self, parent_order_id: OrderId) -> Option<(Quantity, Price)> {
        let icebergs = self.icebergs.read();
        
        if let Some(iceberg) = icebergs.iter().find(|i| i.parent_order_id == parent_order_id) {
            if iceberg.is_complete() {
                return None;
            }
            
            // Clip size is minimum of calculated clip and remaining quantity
            let child_qty = iceberg.current_clip_quantity.min(iceberg.remaining_quantity);
            
            if child_qty > 0.0 {
                Some((child_qty, iceberg.limit_price))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Remove completed iceberg from tracking
    pub fn cleanup_completed(&self) -> usize {
        let mut icebergs = self.icebergs.write();
        let before = icebergs.len();
        icebergs.retain(|i| !i.is_complete());
        let removed = before - icebergs.len();
        
        if removed > 0 {
            debug!("Cleaned up {} completed iceberg orders", removed);
        }
        
        removed
    }

    /// Update ADV cache (call periodically)
    pub fn update_adv(&self, new_adv: f64) {
        *self.adv_cache.write() = new_adv;
        *self.adv_last_update.lock() = Instant::now();
        debug!("Updated ADV cache: {}", new_adv);
    }

    /// Get statistics for monitoring
    pub fn get_stats(&self) -> IcebergStats {
        let icebergs = self.icebergs.read();
        let active_count = icebergs.iter().filter(|i| !i.is_complete()).count();
        let total_remaining: Quantity = icebergs.iter().map(|i| i.remaining_quantity).sum();
        let total_executed: Quantity = icebergs.iter()
            .map(|i| i.total_quantity - i.remaining_quantity)
            .sum();
        
        IcebergStats {
            active_count,
            total_icebergs: icebergs.len(),
            total_remaining,
            total_executed,
            avg_progress: if active_count > 0 {
                icebergs.iter()
                    .filter(|i| !i.is_complete())
                    .map(|i| i.progress())
                    .sum::<f64>() / active_count as f64
            } else {
                0.0
            },
        }
    }
}

/// Statistics for iceberg orders
#[derive(Debug, Clone)]
pub struct IcebergStats {
    pub active_count: usize,
    pub total_icebergs: usize,
    pub total_remaining: Quantity,
    pub total_executed: Quantity,
    pub avg_progress: f64,
}

/// Child order generator for iceberg execution
pub struct IcebergChildGenerator {
    router: Arc<IcebergRouter>,
    /// Pending child orders to submit
    pending_children: RwLock<Vec<PendingChild>>,
}

#[derive(Debug, Clone)]
struct PendingChild {
    parent_order_id: OrderId,
    quantity: Quantity,
    price: Price,
    created_at: Instant,
}

impl IcebergChildGenerator {
    pub fn new(router: Arc<IcebergRouter>) -> Self {
        Self {
            router,
            pending_children: RwLock::new(Vec::new()),
        }
    }

    /// Generate child orders for all active icebergs
    pub fn generate_children(&self) -> Vec<(OrderId, Quantity, Price)> {
        let mut children = Vec::new();
        let mut pending = self.pending_children.write();
        
        // First, add any pending children from previous cycles
        let pending_copy: Vec<_> = pending.drain(..).collect();
        for pc in pending_copy {
            children.push((pc.parent_order_id, pc.quantity, pc.price));
        }
        
        // Then generate new children for active icebergs
        // (In real implementation, this would iterate through active icebergs)
        
        children
    }

    /// Schedule a child order for later submission
    pub fn schedule_child(
        &self,
        parent_order_id: OrderId,
        quantity: Quantity,
        price: Price,
        delay_ms: u64,
    ) {
        let pending = PendingChild {
            parent_order_id,
            quantity,
            price,
            created_at: Instant::now() + Duration::from_millis(delay_ms),
        };
        
        self.pending_children.write().push(pending);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_iceberg_creation() {
        let config = IcebergConfig::default();
        let order_book = Arc::new(RwLock::new(OrderBook::new(Symbol::BTCUSDT)));
        let router = IcebergRouter::new(config, order_book);
        
        let state = router.create_iceberg(
            OrderId::new(1001),
            100.0,
            Side::Buy,
            50000.0,
        );
        
        assert_eq!(state.total_quantity, 100.0);
        assert_eq!(state.remaining_quantity, 100.0);
        assert!(!state.is_complete());
        assert!(state.current_clip_quantity > 0.0);
    }
    
    #[test]
    fn test_iceberg_fill_progress() {
        let config = IcebergConfig::default();
        let order_book = Arc::new(RwLock::new(OrderBook::new(Symbol::BTCUSDT)));
        let router = IcebergRouter::new(config, order_book);
        
        let mut state = router.create_iceberg(
            OrderId::new(1002),
            100.0,
            Side::Sell,
            50100.0,
        );
        
        // Simulate fills
        state.on_fill(25.0, 50100.0);
        assert_eq!(state.remaining_quantity, 75.0);
        assert_eq!(state.clips_executed, 1);
        assert_eq!(state.avg_fill_price, Some(50100.0));
        
        state.on_fill(25.0, 50100.0);
        assert_eq!(state.remaining_quantity, 50.0);
        assert_eq!(state.clips_executed, 2);
        
        state.on_fill(50.0, 50100.0);
        assert!(state.is_complete());
        assert_eq!(state.progress(), 1.0);
    }
    
    #[test]
    fn test_clip_calculation() {
        let config = IcebergConfig::default();
        let order_book = Arc::new(RwLock::new(OrderBook::new(Symbol::BTCUSDT)));
        
        // Add some liquidity to order book
        {
            let mut ob = order_book.write();
            ob.add_bid(50000.0, 10.0);
            ob.add_ask(50010.0, 15.0);
        }
        
        let router = IcebergRouter::new(config, order_book);
        
        let iceberg = IcebergState::new(
            OrderId::new(1003),
            100.0,
            Side::Buy,
            50000.0,
        );
        
        let clip = router.calculate_clip_size(&iceberg);
        assert!(clip > 0.0);
        assert!(clip <= iceberg.total_quantity);
    }
}
