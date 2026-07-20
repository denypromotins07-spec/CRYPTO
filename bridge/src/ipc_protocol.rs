//! # IPC Protocol Definition using FlatBuffers-style Binary Format
//! 
//! This module defines the binary protocol for ultra-fast, zero-copy
//! inter-process communication between Rust and Python components.
//! 
//! Features:
//! - Fixed-size headers for predictable parsing
//! - Zero-copy serialization where possible
//! - Support for OrderBook, Trade, and Signal events
//! - Checksum validation for data integrity

use std::mem;

/// Magic number for protocol validation
const PROTOCOL_MAGIC: u32 = 0x5155414E; // "QUAN" in ASCII

/// Protocol version
const PROTOCOL_VERSION: u16 = 1;

/// Maximum payload size (64KB)
const MAX_PAYLOAD_SIZE: usize = 64 * 1024;

/// Message types
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageType {
    /// Order book snapshot/update
    OrderBook = 0,
    /// Trade execution
    Trade = 1,
    /// Trading signal from ML model
    Signal = 2,
    /// Order acknowledgment
    OrderAck = 3,
    /// Order rejection
    OrderReject = 4,
    /// Heartbeat/keepalive
    Heartbeat = 5,
    /// Custom data
    Custom = 255,
}

impl MessageType {
    #[inline]
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => MessageType::OrderBook,
            1 => MessageType::Trade,
            2 => MessageType::Signal,
            3 => MessageType::OrderAck,
            4 => MessageType::OrderReject,
            5 => MessageType::Heartbeat,
            _ => MessageType::Custom,
        }
    }
}

/// Message header (fixed 32 bytes for alignment)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MessageHeader {
    /// Magic number (4 bytes)
    pub magic: u32,
    /// Protocol version (2 bytes)
    pub version: u16,
    /// Message type (1 byte)
    pub message_type: u8,
    /// Flags (1 byte)
    pub flags: u8,
    /// Payload length (4 bytes)
    pub payload_len: u32,
    /// Sequence number (8 bytes)
    pub sequence: u64,
    /// Timestamp in nanoseconds (8 bytes)
    pub timestamp_ns: u64,
    /// Checksum (4 bytes)
    pub checksum: u32,
}

impl MessageHeader {
    /// Create a new message header
    #[inline]
    pub fn new(message_type: MessageType, payload_len: u32, sequence: u64) -> Self {
        let header = MessageHeader {
            magic: PROTOCOL_MAGIC,
            version: PROTOCOL_VERSION,
            message_type: message_type as u8,
            flags: 0,
            payload_len,
            sequence,
            timestamp_ns: crate::get_timestamp_ns(),
            checksum: 0,
        };
        
        // Calculate checksum
        let checksum = header.calculate_checksum();
        MessageHeader { checksum, ..header }
    }
    
    /// Validate the message header
    #[inline]
    pub fn validate(&self) -> bool {
        // Check magic number
        if self.magic != PROTOCOL_MAGIC {
            return false;
        }
        
        // Check version
        if self.version != PROTOCOL_VERSION {
            return false;
        }
        
        // Check payload size
        if self.payload_len > MAX_PAYLOAD_SIZE as u32 {
            return false;
        }
        
        // Verify checksum
        if self.checksum != self.calculate_checksum() {
            return false;
        }
        
        true
    }
    
    /// Calculate checksum (simple CRC32-like)
    #[inline]
    fn calculate_checksum(&self) -> u32 {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                mem::size_of::<Self>() - 4, // Exclude checksum field
            )
        };
        
        crc32_simple(bytes)
    }
    
    /// Get header size
    #[inline]
    pub const fn size() -> usize {
        mem::size_of::<MessageHeader>()
    }
}

/// Simple CRC32 implementation (no external dependencies)
#[inline]
fn crc32_simple(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    
    crc ^ 0xFFFF_FFFF
}

/// Order book level (price/quantity pair)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PriceLevel {
    /// Price (scaled by 10^8 to avoid floating point)
    pub price: i64,
    /// Quantity (scaled by 10^8)
    pub quantity: i64,
}

impl PriceLevel {
    #[inline]
    pub fn new(price: f64, quantity: f64) -> Self {
        PriceLevel {
            price: (price * 100_000_000.0) as i64,
            quantity: (quantity * 100_000_000.0) as i64,
        }
    }
    
    #[inline]
    pub fn price_f64(&self) -> f64 {
        self.price as f64 / 100_000_000.0
    }
    
    #[inline]
    pub fn quantity_f64(&self) -> f64 {
        self.quantity as f64 / 100_000_000.0
    }
}

/// Order book message payload
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OrderBookPayload {
    /// Symbol (12 bytes, null-terminated)
    pub symbol: [u8; 12],
    /// Number of bid levels
    pub bid_count: u8,
    /// Number of ask levels
    pub ask_count: u8,
    /// Reserved padding
    pub _padding: [u8; 2],
    /// First update ID
    pub first_update_id: i64,
    /// Last update ID
    pub last_update_id: i64,
    /// Bid levels (up to 20)
    pub bids: [PriceLevel; 20],
    /// Ask levels (up to 20)
    pub asks: [PriceLevel; 20],
}

impl OrderBookPayload {
    /// Size of the payload
    pub const SIZE: usize = mem::size_of::<OrderBookPayload>();
    
    /// Create a new order book payload
    pub fn new(symbol: &str) -> Self {
        let mut payload = OrderBookPayload {
            symbol: [0u8; 12],
            bid_count: 0,
            ask_count: 0,
            _padding: [0u8; 2],
            first_update_id: 0,
            last_update_id: 0,
            bids: [PriceLevel { price: 0, quantity: 0 }; 20],
            asks: [PriceLevel { price: 0, quantity: 0 }; 20],
        };
        
        // Set symbol
        let bytes = symbol.as_bytes();
        let len = bytes.len().min(11);
        payload.symbol[..len].copy_from_slice(&bytes[..len]);
        
        payload
    }
    
    /// Set bid levels
    pub fn set_bids(&mut self, levels: &[(f64, f64)]) {
        self.bid_count = levels.len().min(20) as u8;
        for (i, &(price, qty)) in levels.iter().take(20).enumerate() {
            self.bids[i] = PriceLevel::new(price, qty);
        }
    }
    
    /// Set ask levels
    pub fn set_asks(&mut self, levels: &[(f64, f64)]) {
        self.ask_count = levels.len().min(20) as u8;
        for (i, &(price, qty)) in levels.iter().take(20).enumerate() {
            self.asks[i] = PriceLevel::new(price, qty);
        }
    }
}

/// Trade message payload
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TradePayload {
    /// Symbol (12 bytes)
    pub symbol: [u8; 12],
    /// Trade ID
    pub trade_id: i64,
    /// Price (scaled)
    pub price: i64,
    /// Quantity (scaled)
    pub quantity: i64,
    /// Buyer order ID
    pub buyer_order_id: i64,
    /// Seller order ID
    pub seller_order_id: i64,
    /// Buyer is maker flag
    pub buyer_is_maker: bool,
    /// Reserved padding
    pub _padding: [u8; 7],
}

impl TradePayload {
    pub const SIZE: usize = mem::size_of::<TradePayload>();
    
    pub fn new(symbol: &str) -> Self {
        let mut payload = TradePayload {
            symbol: [0u8; 12],
            trade_id: 0,
            price: 0,
            quantity: 0,
            buyer_order_id: 0,
            seller_order_id: 0,
            buyer_is_maker: false,
            _padding: [0u8; 7],
        };
        
        let bytes = symbol.as_bytes();
        let len = bytes.len().min(11);
        payload.symbol[..len].copy_from_slice(&bytes[..len]);
        
        payload
    }
    
    pub fn set_trade(&mut self, trade_id: i64, price: f64, quantity: f64, 
                     buyer_order_id: i64, seller_order_id: i64, buyer_is_maker: bool) {
        self.trade_id = trade_id;
        self.price = (price * 100_000_000.0) as i64;
        self.quantity = (quantity * 100_000_000.0) as i64;
        self.buyer_order_id = buyer_order_id;
        self.seller_order_id = seller_order_id;
        self.buyer_is_maker = buyer_is_maker;
    }
    
    pub fn price_f64(&self) -> f64 {
        self.price as f64 / 100_000_000.0
    }
    
    pub fn quantity_f64(&self) -> f64 {
        self.quantity as f64 / 100_000_000.0
    }
}

/// Signal message payload (from ML model)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SignalPayload {
    /// Symbol (12 bytes)
    pub symbol: [u8; 12],
    /// Signal type: 0=None, 1=Buy, 2=Sell, 3=Strong Buy, 4=Strong Sell
    pub signal_type: u8,
    /// Confidence score (0-100, scaled by 100)
    pub confidence: u8,
    /// Target price (scaled)
    pub target_price: i64,
    /// Stop loss price (scaled)
    pub stop_loss: i64,
    /// Position size recommendation (scaled by 10^6)
    pub position_size: i32,
    /// Time horizon in seconds
    pub time_horizon_secs: u32,
    /// Model ID that generated the signal
    pub model_id: u32,
    /// Reserved
    pub _reserved: [u8; 4],
}

impl SignalPayload {
    pub const SIZE: usize = mem::size_of::<SignalPayload>();
    
    pub fn new(symbol: &str) -> Self {
        let mut payload = SignalPayload {
            symbol: [0u8; 12],
            signal_type: 0,
            confidence: 0,
            target_price: 0,
            stop_loss: 0,
            position_size: 0,
            time_horizon_secs: 0,
            model_id: 0,
            _reserved: [0u8; 4],
        };
        
        let bytes = symbol.as_bytes();
        let len = bytes.len().min(11);
        payload.symbol[..len].copy_from_slice(&bytes[..len]);
        
        payload
    }
    
    pub fn set_signal(&mut self, signal_type: u8, confidence: f32,
                      target_price: f64, stop_loss: f64,
                      position_size: f32, time_horizon_secs: u32,
                      model_id: u32) {
        self.signal_type = signal_type.min(4);
        self.confidence = (confidence * 100.0).min(100.0) as u8;
        self.target_price = (target_price * 100_000_000.0) as i64;
        self.stop_loss = (stop_loss * 100_000_000.0) as i64;
        self.position_size = (position_size * 1_000_000.0) as i32;
        self.time_horizon_secs = time_horizon_secs;
        self.model_id = model_id;
    }
}

/// Complete IPC message with header and payload
#[repr(C)]
pub struct IpcMessage {
    pub header: MessageHeader,
    pub payload: [u8; MAX_PAYLOAD_SIZE],
}

impl IpcMessage {
    /// Total maximum size
    pub const MAX_SIZE: usize = mem::size_of::<MessageHeader>() + MAX_PAYLOAD_SIZE;
    
    /// Create a new IPC message
    pub fn new(message_type: MessageType, payload_data: &[u8], sequence: u64) -> Option<Self> {
        if payload_data.len() > MAX_PAYLOAD_SIZE {
            return None;
        }
        
        let header = MessageHeader::new(message_type, payload_data.len() as u32, sequence);
        
        let mut message = IpcMessage {
            header,
            payload: [0u8; MAX_PAYLOAD_SIZE],
        };
        
        message.payload[..payload_data.len()].copy_from_slice(payload_data);
        
        Some(message)
    }
    
    /// Serialize to bytes (zero-copy view)
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        let total_len = mem::size_of::<MessageHeader>() + self.header.payload_len as usize;
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                total_len,
            )
        }
    }
    
    /// Deserialize from bytes
    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Option<&Self> {
        if bytes.len() < mem::size_of::<MessageHeader>() {
            return None;
        }
        
        let header_ptr = bytes.as_ptr() as *const MessageHeader;
        let header = unsafe { &*header_ptr };
        
        if !header.validate() {
            return None;
        }
        
        let total_len = mem::size_of::<MessageHeader>() + header.payload_len as usize;
        if bytes.len() < total_len {
            return None;
        }
        
        unsafe {
            Some(&*(bytes.as_ptr() as *const IpcMessage))
        }
    }
}

/// Message builder for constructing IPC messages
pub struct MessageBuilder {
    sequence: u64,
    buffer: Vec<u8>,
}

impl MessageBuilder {
    pub fn new() -> Self {
        MessageBuilder {
            sequence: 0,
            buffer: Vec::with_capacity(IpcMessage::MAX_SIZE),
        }
    }
    
    /// Build an order book message
    pub fn order_book(&mut self, symbol: &str, bids: &[(f64, f64)], 
                      asks: &[(f64, f64)], first_id: i64, last_id: i64) -> Vec<u8> {
        let mut payload = OrderBookPayload::new(symbol);
        payload.first_update_id = first_id;
        payload.last_update_id = last_id;
        payload.set_bids(bids);
        payload.set_asks(asks);
        
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &payload as *const OrderBookPayload as *const u8,
                OrderBookPayload::SIZE,
            )
        };
        
        self.sequence += 1;
        let message = IpcMessage::new(MessageType::OrderBook, bytes, self.sequence)
            .expect("OrderBook payload too large");
        
        message.as_bytes().to_vec()
    }
    
    /// Build a trade message
    pub fn trade(&mut self, symbol: &str, trade_id: i64, price: f64, 
                 quantity: f64, buyer_order_id: i64, seller_order_id: i64,
                 buyer_is_maker: bool) -> Vec<u8> {
        let mut payload = TradePayload::new(symbol);
        payload.set_trade(trade_id, price, quantity, buyer_order_id, seller_order_id, buyer_is_maker);
        
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &payload as *const TradePayload as *const u8,
                TradePayload::SIZE,
            )
        };
        
        self.sequence += 1;
        let message = IpcMessage::new(MessageType::Trade, bytes, self.sequence)
            .expect("Trade payload too large");
        
        message.as_bytes().to_vec()
    }
    
    /// Build a signal message
    pub fn signal(&mut self, symbol: &str, signal_type: u8, confidence: f32,
                  target_price: f64, stop_loss: f64, position_size: f32,
                  time_horizon_secs: u32, model_id: u32) -> Vec<u8> {
        let mut payload = SignalPayload::new(symbol);
        payload.set_signal(signal_type, confidence, target_price, stop_loss,
                          position_size, time_horizon_secs, model_id);
        
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &payload as *const SignalPayload as *const u8,
                SignalPayload::SIZE,
            )
        };
        
        self.sequence += 1;
        let message = IpcMessage::new(MessageType::Signal, bytes, self.sequence)
            .expect("Signal payload too large");
        
        message.as_bytes().to_vec()
    }
}

impl Default for MessageBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_message_header() {
        let header = MessageHeader::new(MessageType::Trade, 100, 1);
        assert!(header.validate());
        assert_eq!(header.magic, PROTOCOL_MAGIC);
        assert_eq!(header.version, PROTOCOL_VERSION);
    }
    
    #[test]
    fn test_order_book_payload() {
        let mut payload = OrderBookPayload::new("BTCUSDT");
        payload.set_bids(&[(50000.0, 1.5), (49999.0, 2.0)]);
        payload.set_asks(&[(50001.0, 1.0), (50002.0, 2.5)]);
        
        assert_eq!(payload.bid_count, 2);
        assert_eq!(payload.ask_count, 2);
        assert!((payload.bids[0].price_f64() - 50000.0).abs() < 0.0001);
    }
    
    #[test]
    fn test_trade_payload() {
        let mut payload = TradePayload::new("ETHUSDT");
        payload.set_trade(12345, 3000.50, 0.5, 100, 101, true);
        
        assert_eq!(payload.trade_id, 12345);
        assert!((payload.price_f64() - 3000.50).abs() < 0.0001);
        assert!(payload.buyer_is_maker);
    }
    
    #[test]
    fn test_ipc_message_serialization() {
        let mut builder = MessageBuilder::new();
        let bytes = builder.trade("BTCUSDT", 1, 50000.0, 0.001, 100, 101, false);
        
        let message = IpcMessage::from_bytes(&bytes).unwrap();
        assert_eq!(message.header.message_type, MessageType::Trade as u8);
        assert!(message.header.validate());
    }
}
