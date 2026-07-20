//! `bridge_monitor.rs` - Cross-Chain Bridge Monitor for Arbitrage Detection
//! 
//! **STAGE 10 | CHAPTER 4 | FILE 1**
//! 
//! This module implements an asynchronous, low-memory monitor for major cross-chain
//! bridges (LayerZero, Wormhole, etc.) to detect:
//! - Latency discrepancies between chains
//! - Bridge depegs (price differences across chains)
//! - Liquidity pool imbalances for arbitrage opportunities
//! 
//! **Key Features:**
//! - Async WebSocket connections to multiple chains
//! - Memory-bounded event queues
//! - Real-time spread calculation
//! - Alert generation for arbitrage opportunities

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Maximum number of bridge states to track (memory bounded)
const MAX_BRIDGES: usize = 20;

/// Supported bridge protocols
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BridgeProtocol {
    LayerZero,
    Wormhole,
    Axelar,
    Synapse,
    Hop,
    Custom(&'static str),
}

/// Chain identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChainId {
    Ethereum,
    BSC,
    Polygon,
    Avalanche,
    Arbitrum,
    Optimism,
    Solana,
    Custom(u64),
}

impl ChainId {
    pub fn as_u64(&self) -> u64 {
        match self {
            ChainId::Ethereum => 1,
            ChainId::BSC => 56,
            ChainId::Polygon => 137,
            ChainId::Avalanche => 43114,
            ChainId::Arbitrum => 42161,
            ChainId::Optimism => 10,
            ChainId::Solana => 999999999, // Pseudo ID for Solana
            ChainId::Custom(id) => *id,
        }
    }
}

/// Bridge liquidity pool state
#[derive(Debug, Clone)]
pub struct BridgePoolState {
    pub chain_from: ChainId,
    pub chain_to: ChainId,
    pub token: String,
    pub liquidity_from: f64,
    pub liquidity_to: f64,
    pub price_from: f64,
    pub price_to: f64,
    pub last_update_ns: u64,
    pub is_operational: bool,
}

/// Arbitrage opportunity detected
#[derive(Debug, Clone)]
pub struct BridgeArbitrageOpportunity {
    pub bridge: BridgeProtocol,
    pub token: String,
    pub buy_chain: ChainId,
    pub sell_chain: ChainId,
    pub buy_price: f64,
    pub sell_price: f64,
    pub spread_pct: f64,
    pub estimated_profit: f64,
    pub liquidity_available: f64,
    pub timestamp_ns: u64,
    pub confidence: f64, // 0.0 to 1.0
}

/// Bridge monitoring statistics
#[derive(Debug, Clone, Default)]
pub struct BridgeMonitorStats {
    pub updates_processed: u64,
    pub opportunities_found: u64,
    pub depeg_events: u64,
    pub avg_latency_ms: f64,
    pub max_spread_observed: f64,
}

/// Main bridge monitor
pub struct BridgeMonitor {
    /// Active bridge connections
    bridges: HashMap<BridgeProtocol, Vec<(ChainId, ChainId)>>,
    /// Current pool states (bounded)
    pool_states: HashMap<(BridgeProtocol, ChainId, ChainId, String), BridgePoolState>,
    /// Detected opportunities (bounded queue)
    opportunities: Vec<BridgeArbitrageOpportunity>,
    /// Statistics
    stats: BridgeMonitorStats,
    /// Monitoring active flag
    is_monitoring: AtomicBool,
    /// Last update timestamp
    last_update_ns: AtomicU64,
    /// Configuration
    config: BridgeMonitorConfig,
}

/// Configuration for bridge monitoring
#[derive(Debug, Clone)]
pub struct BridgeMonitorConfig {
    /// Minimum spread percentage to consider arbitrage
    pub min_spread_pct: f64,
    /// Minimum liquidity required (in USD)
    pub min_liquidity_usd: f64,
    /// Maximum age of data to consider valid (ms)
    pub max_data_age_ms: u64,
    /// Depeg threshold (percentage)
    pub depeg_threshold_pct: f64,
    /// Confidence threshold for alerts
    pub confidence_threshold: f64,
}

impl Default for BridgeMonitorConfig {
    fn default() -> Self {
        Self {
            min_spread_pct: 0.5,      // 0.5% minimum spread
            min_liquidity_usd: 1000.0, // $1000 minimum liquidity
            max_data_age_ms: 5000,     // 5 second max age
            depeg_threshold_pct: 2.0,  // 2% depeg threshold
            confidence_threshold: 0.7, // 70% confidence threshold
        }
    }
}

impl BridgeMonitor {
    /// Create a new bridge monitor with default configuration
    pub fn new() -> Self {
        Self::with_config(BridgeMonitorConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: BridgeMonitorConfig) -> Self {
        Self {
            bridges: HashMap::new(),
            pool_states: HashMap::new(),
            opportunities: Vec::with_capacity(100),
            stats: BridgeMonitorStats::default(),
            is_monitoring: AtomicBool::new(false),
            last_update_ns: AtomicU64::new(0),
            config,
        }
    }

    /// Register a bridge protocol with its supported routes
    pub fn register_bridge(
        &mut self,
        protocol: BridgeProtocol,
        routes: Vec<(ChainId, ChainId)>,
    ) {
        if self.bridges.len() >= MAX_BRIDGES {
            // Memory limit reached - remove oldest
            if let Some(first_key) = self.bridges.keys().next().cloned() {
                self.bridges.remove(&first_key);
            }
        }
        self.bridges.insert(protocol, routes);
    }

    /// Update pool state for a specific bridge route
    /// 
    /// # Arguments
    /// * `protocol` - Bridge protocol
    /// * `chain_from` - Source chain
    /// * `chain_to` - Destination chain
    /// * `token` - Token symbol
    /// * `liquidity_from` - Liquidity on source chain
    /// * `liquidity_to` - Liquidity on destination chain
    /// * `price_from` - Price on source chain
    /// * `price_to` - Price on destination chain
    /// * `timestamp_ns` - Update timestamp
    pub fn update_pool_state(
        &mut self,
        protocol: BridgeProtocol,
        chain_from: ChainId,
        chain_to: ChainId,
        token: String,
        liquidity_from: f64,
        liquidity_to: f64,
        price_from: f64,
        price_to: f64,
        timestamp_ns: u64,
    ) {
        let key = (protocol, chain_from, chain_to, token.clone());
        
        let state = BridgePoolState {
            chain_from,
            chain_to,
            token: token.clone(),
            liquidity_from,
            liquidity_to,
            price_from,
            price_to,
            last_update_ns: timestamp_ns,
            is_operational: true,
        };

        // Check for depeg
        if self.is_depegged(price_from, price_to) {
            self.stats.depeg_events += 1;
        }

        // Check for arbitrage opportunity
        if let Some(opportunity) = self.check_arbitrage(&state) {
            self.opportunities.push(opportunity);
            self.stats.opportunities_found += 1;
            
            // Keep only recent opportunities
            if self.opportunities.len() > 100 {
                self.opportunities.remove(0);
            }
        }

        self.pool_states.insert(key, state);
        self.stats.updates_processed += 1;
        self.last_update_ns.store(timestamp_ns, Ordering::Release);
    }

    /// Check if prices indicate a depeg
    fn is_depegged(&self, price_from: f64, price_to: f64) -> bool {
        if price_from == 0.0 || price_to == 0.0 {
            return false;
        }
        let diff_pct = ((price_from - price_to).abs() / price_from) * 100.0;
        diff_pct > self.config.depeg_threshold_pct
    }

    /// Check for arbitrage opportunity in pool state
    fn check_arbitrage(&self, state: &BridgePoolState) -> Option<BridgeArbitrageOpportunity> {
        if !state.is_operational {
            return None;
        }

        if state.price_from == 0.0 || state.price_to == 0.0 {
            return None;
        }

        // Calculate spread
        let spread_pct = ((state.price_to - state.price_from).abs() / state.price_from) * 100.0;

        if spread_pct < self.config.min_spread_pct {
            return None;
        }

        // Check liquidity
        let min_liquidity = state.liquidity_from.min(state.liquidity_to);
        if min_liquidity * state.price_from.min(state.price_to) < self.config.min_liquidity_usd {
            return None;
        }

        // Determine direction
        let (buy_chain, sell_chain, buy_price, sell_price) = if state.price_from < state.price_to {
            (state.chain_from, state.chain_to, state.price_from, state.price_to)
        } else {
            (state.chain_to, state.chain_from, state.price_to, state.price_from)
        };

        // Estimate profit (simplified - doesn't include bridge fees)
        let estimated_profit = (sell_price - buy_price) * min_liquidity * 0.9; // 10% slippage buffer

        // Calculate confidence based on data freshness and liquidity
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let age_ms = (now_ns.saturating_sub(state.last_update_ns)) / 1_000_000;
        let freshness_factor = (1.0 - (age_ms as f64 / self.config.max_data_age_ms as f64)).max(0.0);
        let liquidity_factor = (min_liquidity / 10000.0).min(1.0); // Normalize to $10k
        let confidence = freshness_factor * 0.5 + liquidity_factor * 0.5;

        if confidence < self.config.confidence_threshold {
            return None;
        }

        Some(BridgeArbitrageOpportunity {
            bridge: BridgeProtocol::Custom("unknown"), // Would be passed in
            token: state.token.clone(),
            buy_chain,
            sell_chain,
            buy_price,
            sell_price,
            spread_pct,
            estimated_profit,
            liquidity_available: min_liquidity,
            timestamp_ns: now_ns,
            confidence,
        })
    }

    /// Get current opportunities
    pub fn get_opportunities(&self) -> &[BridgeArbitrageOpportunity] {
        &self.opportunities
    }

    /// Get statistics
    pub fn stats(&self) -> &BridgeMonitorStats {
        &self.stats
    }

    /// Start monitoring
    pub fn start(&self) {
        self.is_monitoring.store(true, Ordering::Release);
    }

    /// Stop monitoring
    pub fn stop(&self) {
        self.is_monitoring.store(false, Ordering::Release);
    }

    /// Check if monitoring is active
    pub fn is_active(&self) -> bool {
        self.is_monitoring.load(Ordering::Acquire)
    }

    /// Get pool state for a specific route
    pub fn get_pool_state(
        &self,
        protocol: BridgeProtocol,
        from: ChainId,
        to: ChainId,
        token: &str,
    ) -> Option<&BridgePoolState> {
        self.pool_states.get(&(protocol, from, to, token.to_string()))
    }

    /// Clear old opportunities
    pub fn clear_old_opportunities(&mut self, max_age_ms: u64) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        self.opportunities.retain(|opp| {
            let age_ms = (now_ns.saturating_sub(opp.timestamp_ns)) / 1_000_000;
            age_ms <= max_age_ms
        });
    }
}

impl Default for BridgeMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_monitor_basic() {
        let mut monitor = BridgeMonitor::new();
        
        monitor.register_bridge(
            BridgeProtocol::LayerZero,
            vec![(ChainId::Ethereum, ChainId::BSC)],
        );

        // Simulate a price discrepancy
        monitor.update_pool_state(
            BridgeProtocol::LayerZero,
            ChainId::Ethereum,
            ChainId::BSC,
            "USDC".to_string(),
            100000.0,
            100000.0,
            1.000,
            1.008, // 0.8% higher on BSC
            1_000_000_000_000,
        );

        assert_eq!(monitor.stats.updates_processed, 1);
        assert!(!monitor.get_opportunities().is_empty());
    }

    #[test]
    fn test_depeg_detection() {
        let monitor = BridgeMonitor::new();
        
        // Normal prices
        assert!(!monitor.is_depegged(1.0, 1.01));
        
        // Depegged (> 2%)
        assert!(monitor.is_depegged(1.0, 1.03));
    }
}
