//! Funding Rate Arbitrage Engine
//! 
//! Real-time calculation of funding rates across Binance perpetual futures
//! and spot markets. Executes cash-and-carry arbitrage when the funding rate
//! exceeds the transaction cost threshold.
//! 
//! Hardware Target: AMD Ryzen AI 5 with SIMD optimizations
//! Memory Constraint: Bounded data structures, pre-allocated buffers

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use dashmap::DashMap;
use std::time::{Instant, Duration};

/// Funding rate data for a perpetual contract
#[derive(Debug, Clone)]
pub struct FundingRate {
    pub symbol: u64,
    pub funding_rate: f64, // Annualized rate as decimal
    pub predicted_rate: f64,
    pub last_funding_time: u64,
    pub next_funding_time: u64,
    pub index_price: u64, // Fixed-point (price * 1e8)
    pub mark_price: u64,
    pub timestamp_ns: u64,
}

/// Arbitrage opportunity from funding rate differential
#[derive(Debug, Clone)]
pub struct FundingArbOpportunity {
    pub symbol: u64,
    pub annualized_rate: f64,
    pub expected_profit_bps: i32,
    pub spot_price: u64,
    pub futures_price: u64,
    pub price_basis_bps: i32,
    pub recommended_position: Position,
    pub risk_score: f32,
    pub confidence: f32,
    pub detected_at_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Position {
    LongSpotShortFuture,
    ShortSpotLongFuture,
    None,
}

/// Transaction costs for arbitrage calculation
#[derive(Debug, Clone, Copy)]
pub struct TransactionCosts {
    pub spot_maker_fee_bps: i32,
    pub spot_taker_fee_bps: i32,
    pub future_maker_fee_bps: i32,
    pub future_taker_fee_bps: i32,
    pub borrowing_cost_bps: i32, // For shorting spot
    pub slippage_bps: i32,
}

impl TransactionCosts {
    /// Calculate total round-trip cost in basis points
    #[inline]
    pub fn total_round_trip_bps(&self) -> i32 {
        self.spot_taker_fee_bps * 2 + 
        self.future_taker_fee_bps * 2 + 
        self.borrowing_cost_bps + 
        self.slippage_bps * 2
    }
}

/// Main funding arbitrage engine
pub struct FundingArbEngine {
    /// Current funding rates per symbol
    funding_rates: DashMap<u64, FundingRate>,
    
    /// Spot prices per symbol
    spot_prices: DashMap<u64, u64>,
    
    /// Historical funding rates for trend analysis
    funding_history: DashMap<u64, VecDequeWrapper<f64>>,
    
    /// Detected opportunities
    opportunities: DashMap<u64, FundingArbOpportunity>,
    
    /// Transaction costs configuration
    costs: TransactionCosts,
    
    /// Minimum profitable rate threshold (annualized, as decimal)
    min_profitable_rate: f64,
    
    /// Statistics
    opportunities_found: AtomicU64,
    arbitrages_executed: AtomicU64,
    total_profit_bps: AtomicU64,
    
    /// Running flag
    is_running: AtomicBool,
}

/// Simple wrapper for bounded VecDeque-like behavior without alloc
struct VecDequeWrapper {
    data: [f64; 100],
    head: usize,
    size: usize,
}

impl VecDequeWrapper {
    fn new() -> Self {
        Self {
            data: [0.0; 100],
            head: 0,
            size: 0,
        }
    }
    
    fn push(&mut self, value: f64) {
        if self.size < 100 {
            self.data[self.head] = value;
            self.head = (self.head + 1) % 100;
            self.size += 1;
        } else {
            // Overwrite oldest
            let idx = self.head;
            self.data[idx] = value;
            self.head = (self.head + 1) % 100;
        }
    }
    
    fn iter(&self) -> impl Iterator<Item = &f64> {
        self.data.iter().take(self.size)
    }
    
    fn average(&self) -> f64 {
        if self.size == 0 {
            return 0.0;
        }
        let sum: f64 = self.iter().sum();
        sum / self.size as f64
    }
}

impl Default for VecDequeWrapper {
    fn default() -> Self {
        Self::new()
    }
}

impl FundingArbEngine {
    pub fn new(min_profitable_rate: f64, costs: TransactionCosts) -> Self {
        Self {
            funding_rates: DashMap::with_capacity(256),
            spot_prices: DashMap::with_capacity(256),
            funding_history: DashMap::with_capacity(256),
            opportunities: DashMap::with_capacity(256),
            costs,
            min_profitable_rate,
            opportunities_found: AtomicU64::new(0),
            arbitrages_executed: AtomicU64::new(0),
            total_profit_bps: AtomicU64::new(0),
            is_running: AtomicBool::new(false),
        }
    }

    /// Update funding rate for a symbol
    #[inline(always)]
    pub fn update_funding_rate(&self, rate: FundingRate) {
        let symbol = rate.symbol;
        
        // Update history for trend analysis
        if let Some(mut history) = self.funding_history.get_mut(&symbol) {
            history.push(rate.funding_rate);
        } else {
            let mut new_history = VecDequeWrapper::new();
            new_history.push(rate.funding_rate);
            self.funding_history.insert(symbol, new_history);
        }
        
        self.funding_rates.insert(symbol, rate);
    }

    /// Update spot price for a symbol
    #[inline]
    pub fn update_spot_price(&self, symbol: u64, price: u64) {
        self.spot_prices.insert(symbol, price);
    }

    /// Detect arbitrage opportunities
    pub fn detect_opportunities(&self) -> Vec<FundingArbOpportunity> {
        let mut opportunities = Vec::with_capacity(64);
        let now_ns = get_timestamp_ns();
        
        for entry in self.funding_rates.iter() {
            let symbol = *entry.key();
            let funding = entry.value();
            
            // Get spot price
            let spot_price = match self.spot_prices.get(&symbol) {
                Some(p) => *p,
                None => continue,
            };
            
            // Calculate annualized funding rate
            // Funding is typically paid every 8 hours (3 times per day)
            // Annualized = rate_per_period * periods_per_year
            let periods_per_year = 365.0 * 3.0; // 3 times daily
            let annualized_rate = funding.funding_rate * periods_per_year;
            
            // Skip if below threshold
            if annualized_rate.abs() < self.min_profitable_rate {
                continue;
            }
            
            // Calculate basis (futures - spot)
            let basis_bps = if spot_price > 0 {
                ((funding.mark_price as i64 - spot_price as i64) * 10000 / spot_price as i64) as i32
            } else {
                0
            };
            
            // Calculate expected profit
            let total_costs = self.costs.total_round_trip_bps();
            let expected_profit_bps = (annualized_rate * 10000.0) as i32 - total_costs;
            
            // Only consider if profitable
            if expected_profit_bps <= 0 {
                continue;
            }
            
            // Determine optimal position
            let position = if funding.funding_rate > 0.0 {
                // Positive funding: longs pay shorts
                // Strategy: Long spot, short future to collect funding
                Position::LongSpotShortFuture
            } else {
                // Negative funding: shorts pay longs
                // Strategy: Short spot, long future to collect funding
                Position::ShortSpotLongFuture
            };
            
            // Calculate risk score
            let risk_score = self.calculate_risk_score(symbol, funding, &spot_price);
            
            // Calculate confidence based on rate stability
            let confidence = self.calculate_confidence(symbol);
            
            let opportunity = FundingArbOpportunity {
                symbol,
                annualized_rate,
                expected_profit_bps,
                spot_price,
                futures_price: funding.mark_price,
                price_basis_bps: basis_bps,
                recommended_position: position,
                risk_score,
                confidence,
                detected_at_ns: now_ns,
            };
            
            self.opportunities_found.fetch_add(1, Ordering::Relaxed);
            self.opportunities.insert(symbol, opportunity.clone());
            opportunities.push(opportunity);
        }
        
        opportunities
    }

    /// Calculate risk score for an arbitrage opportunity
    fn calculate_risk_score(&self, symbol: u64, funding: &FundingRate, spot_price: &u64) -> f32 {
        let mut risk = 0.0;
        
        // Basis risk (divergence between spot and futures)
        let basis_pct = if *spot_price > 0 {
            ((funding.mark_price as f64 - *spot_price as f64) / *spot_price as f64).abs()
        } else {
            0.0
        };
        risk += (basis_pct * 100.0).min(1.0) * 0.3;
        
        // Funding rate volatility risk
        if let Some(history) = self.funding_history.get(&symbol) {
            let avg = history.average();
            let variance: f64 = history.iter()
                .map(|r| (r - avg).powi(2))
                .sum::<f64>() / history.size.max(1) as f64;
            let std_dev = variance.sqrt();
            risk += (std_dev * 100.0).min(1.0) * 0.4;
        }
        
        // Time to next funding risk
        let now = get_timestamp_ns();
        let time_to_funding_ms = (funding.next_funding_time.saturating_sub(now / 1_000_000)) as f64 / 1_000.0;
        let time_factor = (time_to_funding_ms / 3_600_000.0).min(1.0); // Normalize to 1 hour
        risk += time_factor * 0.3;
        
        risk.min(1.0)
    }

    /// Calculate confidence based on funding rate stability
    fn calculate_confidence(&self, symbol: u64) -> f32 {
        if let Some(history) = self.funding_history.get(&symbol) {
            if history.size < 10 {
                return 0.5;
            }
            
            let avg = history.average();
            let variance: f64 = history.iter()
                .map(|r| (r - avg).powi(2))
                .sum::<f64>() / history.size as f64;
            
            // Lower variance = higher confidence
            let confidence = 1.0 / (1.0 + variance * 1000.0);
            confidence as f32
        } else {
            0.5
        }
    }

    /// Get best opportunity by expected profit
    pub fn get_best_opportunity(&self) -> Option<FundingArbOpportunity> {
        self.opportunities
            .iter()
            .max_by(|a, b| a.value().expected_profit_bps.cmp(&b.value().expected_profit_bps))
            .map(|e| e.value().clone())
    }

    /// Record executed arbitrage
    pub fn record_execution(&self, symbol: u64, profit_bps: i32) {
        self.arbitrages_executed.fetch_add(1, Ordering::Relaxed);
        self.total_profit_bps.fetch_add(profit_bps.unsigned_abs(), Ordering::Relaxed);
        self.opportunities.remove(&symbol);
    }

    /// Get engine statistics
    pub fn get_stats(&self) -> FundingArbStats {
        FundingArbStats {
            tracked_symbols: self.funding_rates.len(),
            opportunities_found: self.opportunities_found.load(Ordering::Relaxed),
            arbitrages_executed: self.arbitrages_executed.load(Ordering::Relaxed),
            total_profit_bps: self.total_profit_bps.load(Ordering::Relaxed),
            current_opportunities: self.opportunities.len(),
        }
    }
}

#[derive(Debug)]
pub struct FundingArbStats {
    pub tracked_symbols: usize,
    pub opportunities_found: u64,
    pub arbitrages_executed: u64,
    pub total_profit_bps: u64,
    pub current_opportunities: usize,
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
    fn test_funding_arb_detection() {
        let costs = TransactionCosts {
            spot_maker_fee_bps: 1,
            spot_taker_fee_bps: 1,
            future_maker_fee_bps: 2,
            future_taker_fee_bps: 2,
            borrowing_cost_bps: 5,
            slippage_bps: 2,
        };
        
        let engine = FundingArbEngine::new(0.05, costs); // 5% minimum annualized
        
        // Add funding rate (positive, high)
        let funding = FundingRate {
            symbol: 12345,
            funding_rate: 0.001, // 0.1% per 8 hours = ~109% annualized
            predicted_rate: 0.001,
            last_funding_time: get_timestamp_ns() - 8 * 3_600_000_000_000,
            next_funding_time: get_timestamp_ns() + 4 * 3_600_000_000_000,
            index_price: 50000_00000000,
            mark_price: 50050_00000000,
            timestamp_ns: get_timestamp_ns(),
        };
        engine.update_funding_rate(funding);
        
        // Add spot price
        engine.update_spot_price(12345, 50000_00000000);
        
        let opportunities = engine.detect_opportunities();
        assert!(!opportunities.is_empty());
        assert_eq!(opportunities[0].recommended_position, Position::LongSpotShortFuture);
    }

    #[test]
    fn test_transaction_costs() {
        let costs = TransactionCosts {
            spot_maker_fee_bps: 1,
            spot_taker_fee_bps: 1,
            future_maker_fee_bps: 2,
            future_taker_fee_bps: 2,
            borrowing_cost_bps: 5,
            slippage_bps: 2,
        };
        
        assert_eq!(costs.total_round_trip_bps(), 17);
    }
}
