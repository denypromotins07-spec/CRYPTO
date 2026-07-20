//! Inventory Risk Manager for Market Making
//! ==========================================
//! Chapter 4, File 2: Rust Advanced Market Making & Inventory Risk
//!
//! Advanced inventory risk management. Skews quotes aggressively to flatten 
//! inventory when it breaches predefined thresholds, preventing catastrophic 
//! loss during sudden trend moves.
//!
//! Features: Dynamic skew, position limits, inventory-aware quoting

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;
use chrono::{DateTime, Utc};

/// Inventory risk levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InventoryLevel {
    /// Within normal bounds
    Normal,
    /// Approaching limits
    Warning,
    /// Near maximum allowed
    Critical,
    /// Exceeded limits
    Breach,
}

/// Inventory risk configuration
#[derive(Debug, Clone)]
pub struct InventoryConfig {
    /// Maximum absolute inventory (units)
    pub max_inventory: f64,
    
    /// Warning threshold as fraction of max
    pub warning_threshold: f64,
    
    /// Critical threshold as fraction of max
    pub critical_threshold: f64,
    
    /// Maximum skew adjustment (bps)
    pub max_skew_bps: f64,
    
    /// Skew aggressiveness (higher = more aggressive flattening)
    pub skew_aggressiveness: f64,
    
    /// Enable dynamic skew based on volatility
    pub dynamic_skew: bool,
    
    /// Volatility scaling factor for skew
    pub vol_scale_factor: f64,
    
    /// Time decay for inventory (target flat position over time)
    pub time_decay_enabled: bool,
    
    /// Target inventory (usually 0 for market neutral)
    pub target_inventory: f64,
}

impl Default for InventoryConfig {
    fn default() -> Self {
        Self {
            max_inventory: 100.0,
            warning_threshold: 0.5,   // 50% of max
            critical_threshold: 0.8,  // 80% of max
            max_skew_bps: 50.0,       // Max 50 bps skew
            skew_aggressiveness: 2.0, // Moderate aggressiveness
            dynamic_skew: true,
            vol_scale_factor: 1.0,
            time_decay_enabled: true,
            target_inventory: 0.0,
        }
    }
}

/// Inventory state with metrics
#[derive(Debug, Clone)]
pub struct InventoryState {
    /// Current inventory
    pub current: f64,
    /// Peak inventory (absolute)
    pub peak: f64,
    /// Average inventory over time
    pub average: f64,
    /// Inventory observation count
    pub n_observations: u64,
    /// Last update time
    pub last_update: DateTime<Utc>,
    /// Entry prices for P&L calculation
    pub cost_basis: f64,
    /// Total quantity traded
    pub total_traded: f64,
}

impl InventoryState {
    pub fn new() -> Self {
        Self {
            current: 0.0,
            peak: 0.0,
            average: 0.0,
            n_observations: 0,
            last_update: Utc::now(),
            cost_basis: 0.0,
            total_traded: 0.0,
        }
    }
    
    /// Update inventory and track metrics
    pub fn update(&mut self, new_inventory: f64, price: f64, quantity: f64) {
        let now = Utc::now();
        
        // Update running average
        if self.n_observations > 0 {
            let weight = 1.0 / (self.n_observations + 1) as f64;
            self.average = self.average * (1.0 - weight) + new_inventory.abs() * weight;
        } else {
            self.average = new_inventory.abs();
        }
        
        self.n_observations += 1;
        self.current = new_inventory;
        self.peak = self.peak.max(new_inventory.abs());
        self.last_update = now;
        
        // Update cost basis
        if quantity > 0.0 {
            if (self.current - new_inventory).abs() < quantity {
                // Adding to position
                self.cost_basis = (self.cost_basis * self.current.abs() + price * quantity) 
                    / (self.current.abs() + quantity);
            }
            self.total_traded += quantity;
        }
    }
    
    /// Get current level
    pub fn get_level(&self, max_inventory: f64) -> InventoryLevel {
        let ratio = self.current.abs() / max_inventory;
        
        if ratio >= 1.0 {
            InventoryLevel::Breach
        } else if ratio >= 0.8 {
            InventoryLevel::Critical
        } else if ratio >= 0.5 {
            InventoryLevel::Warning
        } else {
            InventoryLevel::Normal
        }
    }
    
    /// Get inventory as fraction of max
    pub fn utilization(&self, max_inventory: f64) -> f64 {
        self.current.abs() / max_inventory
    }
}

/// Quote skew adjustment
#[derive(Debug, Clone)]
pub struct QuoteSkew {
    /// Bid price adjustment (bps)
    pub bid_adjustment_bps: f64,
    /// Ask price adjustment (bps)
    pub ask_adjustment_bps: f64,
    /// Reason for skew
    pub reason: String,
    /// Urgency level (0-1)
    pub urgency: f64,
}

impl QuoteSkew {
    pub fn neutral() -> Self {
        Self {
            bid_adjustment_bps: 0.0,
            ask_adjustment_bps: 0.0,
            reason: "neutral".to_string(),
            urgency: 0.0,
        }
    }
    
    /// Create skew to reduce long inventory
    pub fn reduce_long(intensity: f64, max_skew_bps: f64) -> Self {
        let skew = (intensity * max_skew_bps).min(max_skew_bps);
        Self {
            bid_adjustment_bps: -skew,  // Lower bid to discourage buys
            ask_adjustment_bps: skew,   // Raise ask to encourage sells
            reason: format!("reduce_long_intensity_{:.2}", intensity),
            urgency: intensity,
        }
    }
    
    /// Create skew to reduce short inventory
    pub fn reduce_short(intensity: f64, max_skew_bps: f64) -> Self {
        let skew = (intensity * max_skrow_bps).min(max_skew_bps);
        Self {
            bid_adjustment_bps: skew,   // Raise bid to encourage buys
            ask_adjustment_bps: -skew,  // Lower ask to discourage sells
            reason: format!("reduce_short_intensity_{:.2}", intensity),
            urgency: intensity,
        }
    }
}

/// Inventory Risk Manager
pub struct InventoryManager {
    /// Configuration
    config: RwLock<InventoryConfig>,
    /// Current state
    state: RwLock<InventoryState>,
    /// Circuit breaker for extreme conditions
    circuit_breaker: AtomicBool,
    /// Breach counter
    breach_count: AtomicU64,
    /// Last skew applied
    last_skew: RwLock<Option<QuoteSkew>>,
}

impl InventoryManager {
    /// Create new inventory manager
    pub fn new(config: InventoryConfig) -> Self {
        Self {
            config: RwLock::new(config),
            state: RwLock::new(InventoryState::new()),
            circuit_breaker: AtomicBool::new(false),
            breach_count: AtomicU64::new(0),
            last_skew: RwLock::new(None),
        }
    }
    
    /// Create with default config
    pub fn with_max_inventory(max_inventory: f64) -> Self {
        let mut config = InventoryConfig::default();
        config.max_inventory = max_inventory;
        Self::new(config)
    }
    
    /// Update configuration
    pub fn update_config(&self, config: InventoryConfig) {
        *self.config.write() = config;
    }
    
    /// Record inventory change
    pub fn record_change(&self, new_inventory: f64, price: f64, quantity: f64) {
        let mut state = self.state.write();
        state.update(new_inventory, price, quantity);
        
        // Check for breach
        let config = self.config.read();
        if state.get_level(config.max_inventory) == InventoryLevel::Breach {
            self.breach_count.fetch_add(1, Ordering::Relaxed);
            
            // Trigger circuit breaker if too many breaches
            if self.breach_count.load(Ordering::Relaxed) >= 3 {
                self.circuit_breaker.store(true, Ordering::Relaxed);
            }
        }
    }
    
    /// Calculate quote skew based on current inventory
    pub fn calculate_skew(&self, current_volatility: Option<f64>) -> QuoteSkew {
        let config = self.config.read();
        let state = self.state.read();
        
        // Check circuit breaker
        if self.circuit_breaker.load(Ordering::Relaxed) {
            return QuoteSkew::neutral();
        }
        
        let utilization = state.utilization(config.max_inventory);
        let level = state.get_level(config.max_inventory);
        
        // Base intensity from utilization
        let base_intensity = match level {
            InventoryLevel::Normal => 0.0,
            InventoryLevel::Warning => (utilization - config.warning_threshold) 
                / (config.critical_threshold - config.warning_threshold),
            InventoryLevel::Critical => 0.5 + (utilization - config.critical_threshold) 
                / (1.0 - config.critical_threshold) * 0.5,
            InventoryLevel::Breach => 1.0,
        };
        
        // Adjust for volatility if enabled
        let intensity = if config.dynamic_skew && current_volatility.is_some() {
            let vol = current_volatility.unwrap();
            let vol_multiplier = vol * config.vol_scale_factor;
            base_intensity * vol_multiplier.min(2.0)
        } else {
            base_intensity
        };
        
        // Apply aggressiveness factor
        let adjusted_intensity = (intensity * config.skew_aggressiveness).min(1.0);
        
        // Calculate skew direction based on inventory sign
        let skew = if state.current > config.target_inventory {
            // Long inventory - skew to reduce
            QuoteSkew::reduce_long(adjusted_intensity, config.max_skew_bps)
        } else if state.current < config.target_inventory {
            // Short inventory - skew to reduce
            QuoteSkew::reduce_short(adjusted_intensity, config.max_skew_bps)
        } else {
            QuoteSkew::neutral()
        };
        
        // Store last skew
        *self.last_skew.write() = Some(skew.clone());
        
        skew
    }
    
    /// Apply skew to bid/ask prices
    pub fn apply_skew(&self, bid_price: f64, ask_price: f64) -> (f64, f64) {
        let skew = self.calculate_skew(None);
        
        let adjusted_bid = bid_price * (1.0 + skew.bid_adjustment_bps / 10000.0);
        let adjusted_ask = ask_price * (1.0 + skew.ask_adjustment_bps / 10000.0);
        
        (adjusted_bid, adjusted_ask)
    }
    
    /// Get current inventory level
    pub fn get_level(&self) -> InventoryLevel {
        let config = self.config.read();
        let state = self.state.read();
        state.get_level(config.max_inventory)
    }
    
    /// Get current inventory
    pub fn get_inventory(&self) -> f64 {
        self.state.read().current
    }
    
    /// Get inventory state
    pub fn get_state(&self) -> InventoryState {
        self.state.read().clone()
    }
    
    /// Check if trading should be halted
    pub fn should_halt(&self) -> bool {
        self.circuit_breaker.load(Ordering::Relaxed) 
            || self.get_level() == InventoryLevel::Breach
    }
    
    /// Reset circuit breaker
    pub fn reset_circuit_breaker(&self) {
        self.circuit_breaker.store(false, Ordering::Relaxed);
        self.breach_count.store(0, Ordering::Relaxed);
    }
    
    /// Manually set inventory (for initialization or correction)
    pub fn set_inventory(&self, inventory: f64, price: f64) {
        let mut state = self.state.write();
        state.current = inventory;
        state.cost_basis = price;
        state.last_update = Utc::now();
    }
    
    /// Get statistics
    pub fn get_statistics(&self) -> InventoryStatistics {
        let config = self.config.read();
        let state = self.state.read();
        let last_skew = self.last_skew.read();
        
        InventoryStatistics {
            current_inventory: state.current,
            peak_inventory: state.peak,
            average_inventory: state.average,
            utilization: state.utilization(config.max_inventory),
            level: state.get_level(config.max_inventory),
            cost_basis: state.cost_basis,
            total_traded: state.total_traded,
            circuit_breaker_active: self.circuit_breaker.load(Ordering::Relaxed),
            breach_count: self.breach_count.load(Ordering::Relaxed),
            last_skew_bps: last_skew.as_ref().map(|s| s.bid_adjustment_bps).unwrap_or(0.0),
            max_skew_bps: config.max_skew_bps,
        }
    }
    
    /// Get time until mean reversion target
    pub fn estimated_flatten_time(&self, avg_fill_rate: f64) -> f64 {
        let state = self.state.read();
        
        if avg_fill_rate <= 0.0 || state.current.abs() < 1e-6 {
            return 0.0;
        }
        
        state.current.abs() / avg_fill_rate
    }
}

/// Inventory statistics snapshot
#[derive(Debug, Clone)]
pub struct InventoryStatistics {
    pub current_inventory: f64,
    pub peak_inventory: f64,
    pub average_inventory: f64,
    pub utilization: f64,
    pub level: InventoryLevel,
    pub cost_basis: f64,
    pub total_traded: f64,
    pub circuit_breaker_active: bool,
    pub breach_count: u64,
    pub last_skew_bps: f64,
    pub max_skew_bps: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_inventory_levels() {
        let manager = InventoryManager::with_max_inventory(100.0);
        
        // Test normal level
        manager.record_change(30.0, 100.0, 30.0);
        assert_eq!(manager.get_level(), InventoryLevel::Normal);
        
        // Test warning level
        manager.record_change(60.0, 100.0, 30.0);
        assert_eq!(manager.get_level(), InventoryLevel::Warning);
        
        // Test critical level
        manager.record_change(85.0, 100.0, 25.0);
        assert_eq!(manager.get_level(), InventoryLevel::Critical);
        
        // Test breach level
        manager.record_change(110.0, 100.0, 25.0);
        assert_eq!(manager.get_level(), InventoryLevel::Breach);
    }
    
    #[test]
    fn test_quote_skew() {
        let manager = InventoryManager::with_max_inventory(100.0);
        
        // No skew at zero inventory
        manager.record_change(0.0, 100.0, 0.0);
        let skew = manager.calculate_skew(None);
        assert!(skew.bid_adjustment_bps.abs() < 0.1);
        assert!(skew.ask_adjustment_bps.abs() < 0.1);
        
        // Skew to reduce long
        manager.record_change(80.0, 100.0, 80.0);
        let skew = manager.calculate_skew(None);
        assert!(skew.bid_adjustment_bps < 0.0); // Lower bid
        assert!(skew.ask_adjustment_bps > 0.0);  // Higher ask
        
        // Skew to reduce short
        manager.record_change(-80.0, 100.0, -160.0);
        let skew = manager.calculate_skew(None);
        assert!(skew.bid_adjustment_bps > 0.0);  // Higher bid
        assert!(skew.ask_adjustment_bps < 0.0); // Lower ask
    }
    
    #[test]
    fn test_circuit_breaker() {
        let manager = InventoryManager::with_max_inventory(100.0);
        
        // Trigger multiple breaches
        for _ in 0..3 {
            manager.record_change(110.0, 100.0, 10.0);
            manager.reset_circuit_breaker();
        }
        
        // Should trigger after 3 breaches
        manager.record_change(110.0, 100.0, 10.0);
        assert!(manager.should_halt());
        
        // Reset and verify
        manager.reset_circuit_breaker();
        assert!(!manager.should_halt());
    }
}
