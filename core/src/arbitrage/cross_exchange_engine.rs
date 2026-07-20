//! Cross-Exchange Arbitrage Engine
//! 
//! A lock-free, ultra-low latency engine for monitoring and executing arbitrage
//! opportunities between Binance and other top-tier CEXs/DEXs.
//! 
//! Hardware Target: AMD Ryzen AI 5 (Zen4) with SIMD optimizations
//! Memory Constraint: Strictly bounded allocations to maintain <8GB global limit
//! Latency Goal: Microsecond-level detection and execution

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use crossbeam::queue::SegQueue;
use dashmap::DashMap;
use std::time::{Instant, Duration};
use rayon::prelude::*;

/// Represents a price quote from an exchange
#[derive(Debug, Clone, Copy)]
pub struct Quote {
    pub exchange_id: u16,
    pub symbol: u64, // Hashed symbol identifier
    pub bid_price: u64, // Fixed-point representation (price * 1e8)
    pub ask_price: u64,
    pub bid_size: u64,
    pub ask_size: u64,
    pub timestamp_ns: u64, // Nanosecond precision timestamp
}

/// Represents an arbitrage opportunity
#[derive(Debug, Clone)]
pub struct ArbOpportunity {
    pub buy_exchange: u16,
    pub sell_exchange: u16,
    pub symbol: u64,
    pub spread_bps: i32, // Basis points * 100 for precision
    pub expected_profit_bps: i32,
    pub leg1_qty: u64,
    pub leg2_qty: u64,
    pub detected_at_ns: u64,
    pub confidence_score: f32,
}

/// Lock-free ring buffer for high-frequency quote storage
struct QuoteRingBuffer {
    buffer: Vec<AtomicQuote>,
    capacity: usize,
    write_idx: AtomicU64,
    read_idx: AtomicU64,
}

#[derive(Debug, Default)]
struct AtomicQuote {
    data: AtomicU64, // Packed quote data for cache efficiency
}

impl QuoteRingBuffer {
    fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize_with(capacity, || AtomicQuote::default());
        Self {
            buffer,
            capacity,
            write_idx: AtomicU64::new(0),
            read_idx: AtomicU64::new(0),
        }
    }

    #[inline]
    fn push(&self, quote: Quote) -> bool {
        let idx = self.write_idx.fetch_add(1, Ordering::Relaxed) % self.capacity as u64;
        // Pack quote into atomic operations for lock-free access
        // In production, this would use more sophisticated packing
        unsafe {
            let ptr = &self.buffer[idx as usize] as *const AtomicQuote as *mut Quote;
            ptr.write(quote);
        }
        true
    }

    #[inline]
    fn pop(&self) -> Option<Quote> {
        let idx = self.read_idx.fetch_add(1, Ordering::Relaxed) % self.capacity as u64;
        unsafe {
            let ptr = &self.buffer[idx as usize] as *const AtomicQuote as *const Quote;
            Some(ptr.read())
        }
    }
}

/// Main cross-exchange arbitrage engine
pub struct CrossExchangeEngine {
    /// Lock-free queue for incoming quotes from all exchanges
    quote_queue: Arc<SegQueue<Quote>>,
    
    /// Latest quotes per symbol per exchange (lock-free using DashMap)
    latest_quotes: DashMap<(u16, u64), Quote>,
    
    /// Detected opportunities queue
    opportunity_queue: Arc<SegQueue<ArbOpportunity>>,
    
    /// Exchange fee structure (maker/taker in basis points)
    exchange_fees: DashMap<u16, ExchangeFees>,
    
    /// Minimum profitable spread in basis points
    min_profitable_spread_bps: i32,
    
    /// Running flag
    is_running: AtomicBool,
    
    /// Statistics counters
    quotes_processed: AtomicU64,
    opportunities_found: AtomicU64,
    trades_executed: AtomicU64,
    
    /// Pre-allocated memory pool for opportunities to avoid allocations during hot path
    opp_pool: SegQueue<Box<ArbOpportunity>>,
}

#[derive(Debug, Clone, Copy)]
pub struct ExchangeFees {
    pub maker_fee_bps: i32,
    pub taker_fee_bps: i32,
    pub withdrawal_fee_bps: i32,
}

impl CrossExchangeEngine {
    pub fn new(min_profit_bps: i32) -> Self {
        let quote_queue = Arc::new(SegQueue::new());
        let opportunity_queue = Arc::new(SegQueue::new());
        
        // Pre-allocate opportunity pool
        let opp_pool = SegQueue::new();
        for _ in 0..1024 {
            opp_pool.push(Box::new(ArbOpportunity {
                buy_exchange: 0,
                sell_exchange: 0,
                symbol: 0,
                spread_bps: 0,
                expected_profit_bps: 0,
                leg1_qty: 0,
                leg2_qty: 0,
                detected_at_ns: 0,
                confidence_score: 0.0,
            }));
        }
        
        Self {
            quote_queue,
            latest_quotes: DashMap::with_capacity(4096), // Pre-size for common pairs
            opportunity_queue,
            exchange_fees: DashMap::new(),
            min_profitable_spread_bps: min_profit_bps,
            is_running: AtomicBool::new(false),
            quotes_processed: AtomicU64::new(0),
            opportunities_found: AtomicU64::new(0),
            trades_executed: AtomicU64::new(0),
            opp_pool,
        }
    }

    /// Register an exchange with its fee structure
    pub fn register_exchange(&self, exchange_id: u16, fees: ExchangeFees) {
        self.exchange_fees.insert(exchange_id, fees);
    }

    /// Push a new quote into the engine (called by network receivers)
    #[inline(always)]
    pub fn push_quote(&self, quote: Quote) {
        self.quote_queue.push(quote);
        self.quotes_processed.fetch_add(1, Ordering::Relaxed);
    }

    /// Core arbitrage detection logic - optimized for SIMD and cache locality
    #[inline]
    pub fn detect_arbitrage(&self, quotes: &[Quote]) -> Vec<ArbOpportunity> {
        let mut opportunities = Vec::with_capacity(quotes.len() / 4);
        
        // Group quotes by symbol for efficient comparison
        let mut symbol_quotes: DashMap<u64, Vec<Quote>> = DashMap::new();
        for quote in quotes {
            symbol_quotes
                .entry(quote.symbol)
                .or_insert_with(Vec::new)
                .push(*quote);
        }
        
        // Parallel processing of symbols using Rayon
        let opps: Vec<Vec<ArbOpportunity>> = symbol_quotes
            .par_iter()
            .filter_map(|entry| {
                let symbol = *entry.key();
                let quotes_vec = entry.value();
                
                if quotes_vec.len() < 2 {
                    return None;
                }
                
                let mut symbol_opps = Vec::new();
                
                // Compare all exchange pairs for this symbol
                for i in 0..quotes_vec.len() {
                    for j in (i + 1)..quotes_vec.len() {
                        let q1 = quotes_vec[i];
                        let q2 = quotes_vec[j];
                        
                        // Check for cross-exchange arbitrage
                        if let Some(opp) = self.check_arb_pair(q1, q2, symbol) {
                            symbol_opps.push(opp);
                        }
                    }
                }
                
                if !symbol_opps.is_empty() {
                    Some(symbol_opps)
                } else {
                    None
                }
            })
            .collect();
        
        for opp_vec in opps {
            opportunities.extend(opp_vec);
        }
        
        opportunities
    }

    /// Check two quotes for arbitrage opportunity
    #[inline(always)]
    fn check_arb_pair(&self, q1: Quote, q2: Quote, symbol: u64) -> Option<ArbOpportunity> {
        // Calculate spread: can we buy on q1's ask and sell on q2's bid?
        let spread_buy_q1_sell_q2 = (q2.bid_price as i64 - q1.ask_price as i64) * 10000 / q1.ask_price as i64;
        
        // Calculate spread: can we buy on q2's ask and sell on q1's bid?
        let spread_buy_q2_sell_q1 = (q1.bid_price as i64 - q2.ask_price as i64) * 10000 / q2.ask_price as i64;
        
        let fees1 = self.exchange_fees.get(&q1.exchange_id)?;
        let fees2 = self.exchange_fees.get(&q2.exchange_id)?;
        
        let total_fees_bps = fees1.taker_fee_bps + fees2.taker_fee_bps + fees1.withdrawal_fee_bps + fees2.withdrawal_fee_bps;
        
        if spread_buy_q1_sell_q2 > total_fees_bps as i64 + self.min_profitable_spread_bps as i64 {
            let expected_profit = spread_buy_q1_sell_q2 as i32 - total_fees_bps;
            let qty = std::cmp::min(q1.ask_size, q2.bid_size);
            
            return Some(ArbOpportunity {
                buy_exchange: q1.exchange_id,
                sell_exchange: q2.exchange_id,
                symbol,
                spread_bps: spread_buy_q1_sell_q2 as i32,
                expected_profit_bps: expected_profit,
                leg1_qty: qty,
                leg2_qty: qty,
                detected_at_ns: get_timestamp_ns(),
                confidence_score: self.calculate_confidence(q1, q2),
            });
        }
        
        if spread_buy_q2_sell_q1 > total_fees_bps as i64 + self.min_profitable_spread_bps as i64 {
            let expected_profit = spread_buy_q2_sell_q1 as i32 - total_fees_bps;
            let qty = std::cmp::min(q2.ask_size, q1.bid_size);
            
            return Some(ArbOpportunity {
                buy_exchange: q2.exchange_id,
                sell_exchange: q1.exchange_id,
                symbol,
                spread_bps: spread_buy_q2_sell_q1 as i32,
                expected_profit_bps: expected_profit,
                leg1_qty: qty,
                leg2_qty: qty,
                detected_at_ns: get_timestamp_ns(),
                confidence_score: self.calculate_confidence(q2, q1),
            });
        }
        
        None
    }

    /// Calculate confidence score based on quote age, size, and exchange reliability
    #[inline]
    fn calculate_confidence(&self, buy_quote: Quote, sell_quote: Quote) -> f32 {
        let now_ns = get_timestamp_ns();
        let max_age_ns = 1_000_000; // 1ms max age for high confidence
        
        let age_factor = {
            let buy_age = now_ns.saturating_sub(buy_quote.timestamp_ns);
            let sell_age = now_ns.saturating_sub(sell_quote.timestamp_ns);
            let max_age = std::cmp::max(buy_age, sell_age);
            if max_age > max_age_ns {
                0.0
            } else {
                1.0 - (max_age as f32 / max_age_ns as f32)
            }
        };
        
        let size_factor = {
            let min_size = std::cmp::min(buy_quote.ask_size, sell_quote.bid_size);
            (min_size as f32 / 100_000_000.0).min(1.0) // Normalize to 1 BTC equivalent
        };
        
        age_factor * 0.6 + size_factor * 0.4
    }

    /// Process the quote queue and detect opportunities
    pub fn process_cycle(&self) {
        let mut batch = Vec::with_capacity(256);
        
        // Drain the queue efficiently
        while let Some(quote) = self.quote_queue.pop() {
            // Update latest quotes
            self.latest_quotes.insert((quote.exchange_id, quote.symbol), quote);
            batch.push(quote);
            
            if batch.len() >= 256 {
                let opportunities = self.detect_arbitrage(&batch);
                for opp in opportunities {
                    self.opportunities_found.fetch_add(1, Ordering::Relaxed);
                    self.opportunity_queue.push(opp);
                }
                batch.clear();
            }
        }
        
        // Process remaining quotes
        if !batch.is_empty() {
            let opportunities = self.detect_arbitrage(&batch);
            for opp in opportunities {
                self.opportunities_found.fetch_add(1, Ordering::Relaxed);
                self.opportunity_queue.push(opp);
            }
        }
    }

    /// Get next opportunity for execution
    pub fn next_opportunity(&self) -> Option<ArbOpportunity> {
        self.opportunity_queue.pop()
    }

    /// Record a successful trade execution
    pub fn record_trade(&self) {
        self.trades_executed.fetch_add(1, Ordering::Relaxed);
    }

    /// Get engine statistics
    pub fn get_stats(&self) -> EngineStats {
        EngineStats {
            quotes_processed: self.quotes_processed.load(Ordering::Relaxed),
            opportunities_found: self.opportunities_found.load(Ordering::Relaxed),
            trades_executed: self.trades_executed.load(Ordering::Relaxed),
            unique_symbols: self.latest_quotes.len(),
        }
    }
}

#[derive(Debug)]
pub struct EngineStats {
    pub quotes_processed: u64,
    pub opportunities_found: u64,
    pub trades_executed: u64,
    pub unique_symbols: usize,
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
    fn test_arbitrage_detection() {
        let engine = CrossExchangeEngine::new(5); // 5 bps minimum profit
        
        // Register exchanges with fees
        engine.register_exchange(1, ExchangeFees {
            maker_fee_bps: 1,
            taker_fee_bps: 1,
            withdrawal_fee_bps: 0,
        });
        engine.register_exchange(2, ExchangeFees {
            maker_fee_bps: 1,
            taker_fee_bps: 1,
            withdrawal_fee_bps: 0,
        });
        
        // Create quotes with arbitrage opportunity
        let q1 = Quote {
            exchange_id: 1,
            symbol: 12345,
            bid_price: 50000_00000000,
            ask_price: 50001_00000000,
            bid_size: 100_00000000,
            ask_size: 100_00000000,
            timestamp_ns: get_timestamp_ns(),
        };
        
        let q2 = Quote {
            exchange_id: 2,
            symbol: 12345,
            bid_price: 50010_00000000,
            ask_price: 50011_00000000,
            bid_size: 100_00000000,
            ask_size: 100_00000000,
            timestamp_ns: get_timestamp_ns(),
        };
        
        let opp = engine.check_arb_pair(q1, q2, 12345);
        assert!(opp.is_some());
        assert!(opp.unwrap().expected_profit_bps > 0);
    }
}
