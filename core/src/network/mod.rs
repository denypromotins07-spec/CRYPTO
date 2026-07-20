//! # Network Module for Binance Connectivity
//! 
//! This module provides the network layer for connecting to Binance WebSocket
//! streams, parsing messages, and managing connection state.
//! 
//! Components:
//! - WebSocket client (websocket.rs)
//! - Message parser (binance_parser.rs)
//! - Connection state management
//! - Error handling traits

pub mod websocket;
pub mod binance_parser;

// Re-export key types for convenience
pub use websocket::{
    WebSocketClient,
    TokioWebSocket,
    WsConfig,
    WsEvent,
    WsStats,
    ConnectionState,
    binance_urls,
    create_binance_config,
};

pub use binance_parser::{
    BinanceParser,
    ParseResult,
    ParseError,
    Trade,
    OrderBookUpdate,
    OrderBookLevel,
    Kline,
};

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Network error types
#[derive(Debug, Clone)]
pub enum NetworkError {
    /// Connection failed
    ConnectionFailed(String),
    /// Connection lost
    ConnectionLost,
    /// Message parse error
    ParseError(ParseError),
    /// Timeout
    Timeout(Duration),
    /// Rate limit exceeded
    RateLimitExceeded,
    /// Invalid message format
    InvalidMessage(String),
    /// Buffer overflow
    BufferOverflow,
}

impl From<ParseError> for NetworkError {
    fn from(err: ParseError) -> Self {
        NetworkError::ParseError(err)
    }
}

impl std::fmt::Display for NetworkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NetworkError::ConnectionFailed(msg) => write!(f, "Connection failed: {}", msg),
            NetworkError::ConnectionLost => write!(f, "Connection lost"),
            NetworkError::ParseError(e) => write!(f, "Parse error: {:?}", e),
            NetworkError::Timeout(d) => write!(f, "Timeout after {:?}", d),
            NetworkError::RateLimitExceeded => write!(f, "Rate limit exceeded"),
            NetworkError::InvalidMessage(msg) => write!(f, "Invalid message: {}", msg),
            NetworkError::BufferOverflow => write!(f, "Buffer overflow"),
        }
    }
}

/// Result type alias for network operations
pub type NetworkResult<T> = Result<T, NetworkError>;

/// Connection state tracker with statistics
pub struct ConnectionManager {
    /// Current connection state
    state: AtomicU32,
    /// Connection attempt count
    attempts: AtomicU32,
    /// Last successful connection timestamp
    last_connected_ns: AtomicU64,
    /// Total bytes received
    bytes_received: AtomicU64,
    /// Total messages received
    messages_received: AtomicU64,
    /// Error count
    errors: AtomicU64,
    /// Is shutting down
    shutting_down: AtomicBool,
}

impl ConnectionManager {
    /// Create a new connection manager
    pub fn new() -> Self {
        ConnectionManager {
            state: AtomicU32::new(ConnectionState::Disconnected as u32),
            attempts: AtomicU32::new(0),
            last_connected_ns: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
            messages_received: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            shutting_down: AtomicBool::new(false),
        }
    }
    
    /// Get current state
    #[inline]
    pub fn state(&self) -> ConnectionState {
        let state_val = self.state.load(Ordering::Acquire);
        match state_val {
            0 => ConnectionState::Disconnected,
            1 => ConnectionState::Connecting,
            2 => ConnectionState::Connected,
            3 => ConnectionState::Reconnecting,
            4 => ConnectionState::Closing,
            _ => ConnectionState::Closed,
        }
    }
    
    /// Set connection state
    #[inline]
    pub fn set_state(&self, state: ConnectionState) {
        self.state.store(state as u32, Ordering::Release);
    }
    
    /// Record a connection attempt
    #[inline]
    pub fn record_attempt(&self) -> u32 {
        self.attempts.fetch_add(1, Ordering::Relaxed) + 1
    }
    
    /// Record successful connection
    #[inline]
    pub fn record_connected(&self) {
        self.last_connected_ns.store(crate::get_timestamp_ns(), Ordering::Relaxed);
        self.attempts.store(0, Ordering::Relaxed);
    }
    
    /// Record received data
    #[inline]
    pub fn record_receive(&self, bytes: usize) {
        self.bytes_received.fetch_add(bytes as u64, Ordering::Relaxed);
        self.messages_received.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Record an error
    #[inline]
    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get time since last connection (nanoseconds)
    #[inline]
    pub fn time_since_last_connection_ns(&self) -> u64 {
        let last = self.last_connected_ns.load(Ordering::Relaxed);
        if last == 0 {
            return 0;
        }
        crate::get_timestamp_ns().saturating_sub(last)
    }
    
    /// Check if should attempt reconnect
    #[inline]
    pub fn should_reconnect(&self, max_attempts: u32) -> bool {
        !self.shutting_down.load(Ordering::Relaxed) 
            && self.attempts.load(Ordering::Relaxed) < max_attempts
    }
    
    /// Signal shutdown
    #[inline]
    pub fn shutdown(&self) {
        self.shutting_down.store(true, Ordering::Release);
        self.set_state(ConnectionState::Closing);
    }
    
    /// Check if shutting down
    #[inline]
    pub fn is_shutting_down(&self) -> bool {
        self.shutting_down.load(Ordering::Relaxed)
    }
    
    /// Get statistics snapshot
    pub fn get_stats(&self) -> ConnectionStats {
        ConnectionStats {
            state: self.state(),
            attempts: self.attempts.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            messages_received: self.messages_received.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
            uptime_ns: self.time_since_last_connection_ns(),
        }
    }
}

impl Default for ConnectionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Connection statistics snapshot
#[derive(Debug, Clone)]
pub struct ConnectionStats {
    pub state: ConnectionState,
    pub attempts: u32,
    pub bytes_received: u64,
    pub messages_received: u64,
    pub errors: u64,
    pub uptime_ns: u64,
}

impl std::fmt::Display for ConnectionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Connection Statistics:")?;
        writeln!(f, "  State: {:?}", self.state)?;
        writeln!(f, "  Attempts: {}", self.attempts)?;
        writeln!(f, "  Messages: {}", self.messages_received)?;
        writeln!(f, "  Bytes: {}", self.bytes_received)?;
        writeln!(f, "  Errors: {}", self.errors)?;
        writeln!(f, "  Uptime: {} ms", self.uptime_ns / 1_000_000)?;
        Ok(())
    }
}

/// Retry strategy for reconnection
pub struct RetryStrategy {
    /// Base delay in milliseconds
    base_delay_ms: u64,
    /// Maximum delay in milliseconds
    max_delay_ms: u64,
    /// Multiplier for exponential backoff
    multiplier: f64,
    /// Current attempt number
    current_attempt: u32,
    /// Jitter factor (0.0 to 1.0)
    jitter: f64,
}

impl RetryStrategy {
    /// Create a new retry strategy with default values
    pub fn new() -> Self {
        RetryStrategy {
            base_delay_ms: 100,
            max_delay_ms: 30000,
            multiplier: 2.0,
            current_attempt: 0,
            jitter: 0.1,
        }
    }
    
    /// Create with custom parameters
    pub fn with_params(base_delay_ms: u64, max_delay_ms: u64, multiplier: f64) -> Self {
        RetryStrategy {
            base_delay_ms,
            max_delay_ms,
            multiplier,
            current_attempt: 0,
            jitter: 0.1,
        }
    }
    
    /// Get the next delay in milliseconds
    pub fn next_delay_ms(&mut self) -> u64 {
        let delay = (self.base_delay_ms as f64 
            * self.multiplier.powi(self.current_attempt as i32)) as u64;
        
        self.current_attempt += 1;
        
        // Apply jitter
        let jitter_range = (delay as f64 * self.jitter) as u64;
        let jitter = if jitter_range > 0 {
            (rand_simple() % (jitter_range * 2)) - jitter_range
        } else {
            0
        };
        
        delay.saturating_add(jitter).min(self.max_delay_ms)
    }
    
    /// Reset the strategy
    pub fn reset(&mut self) {
        self.current_attempt = 0;
    }
    
    /// Set the current attempt number
    pub fn set_attempt(&mut self, attempt: u32) {
        self.current_attempt = attempt;
    }
}

impl Default for RetryStrategy {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple random number generator for jitter (no external dependencies)
fn rand_simple() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    now.as_nanos() as u64
}

/// Stream subscription configuration
#[derive(Clone, Debug)]
pub struct StreamSubscription {
    /// Symbol (e.g., "BTCUSDT")
    pub symbol: String,
    /// Stream types (e.g., "trade", "depth", "kline_1m")
    pub streams: Vec<String>,
    /// Is futures stream
    pub is_futures: bool,
}

impl StreamSubscription {
    /// Create a new subscription
    pub fn new(symbol: &str, streams: &[&str], is_futures: bool) -> Self {
        StreamSubscription {
            symbol: symbol.to_string(),
            streams: streams.iter().map(|s| s.to_string()).collect(),
            is_futures,
        }
    }
    
    /// Get the WebSocket URL path for this subscription
    pub fn url_path(&self) -> String {
        if self.streams.is_empty() {
            return self.symbol.to_lowercase();
        }
        
        let parts: Vec<String> = self.streams
            .iter()
            .map(|s| format!("{}@{}", self.symbol.to_lowercase(), s))
            .collect();
        
        parts.join("/")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_connection_manager() {
        let manager = ConnectionManager::new();
        
        assert_eq!(manager.state(), ConnectionState::Disconnected);
        
        manager.set_state(ConnectionState::Connecting);
        assert_eq!(manager.state(), ConnectionState::Connecting);
        
        manager.record_attempt();
        manager.record_connected();
        assert_eq!(manager.attempts.load(Ordering::Relaxed), 0);
    }
    
    #[test]
    fn test_retry_strategy() {
        let mut strategy = RetryStrategy::new();
        
        let delay1 = strategy.next_delay_ms();
        let delay2 = strategy.next_delay_ms();
        
        assert!(delay2 >= delay1); // Exponential backoff
        
        strategy.reset();
        let delay3 = strategy.next_delay_ms();
        assert_eq!(delay3, strategy.base_delay_ms);
    }
    
    #[test]
    fn test_stream_subscription() {
        let sub = StreamSubscription::new("BTCUSDT", &["trade", "depth"], false);
        
        assert_eq!(sub.symbol, "BTCUSDT");
        assert_eq!(sub.streams.len(), 2);
        assert!(!sub.is_futures);
        
        let path = sub.url_path();
        assert!(path.contains("btcusdt"));
        assert!(path.contains("trade"));
        assert!(path.contains("depth"));
    }
}
