//! FIX Protocol 4.4 Engine Foundation
//! 
//! Implementation of Financial Information eXchange (FIX) protocol version 4.4.
//! While Binance primarily uses REST/WS, implementing a FIX engine prepares the 
//! Rust core for institutional-grade latency when connecting to future prime brokers 
//! or faster venues.
//!
//! Target Hardware: AMD Ryzen AI 5 + AMD Radeon GPU + 16GB RAM (8GB cap)

use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use parking_lot::RwLock;
use log::{info, warn, error, debug};
use serde::{Serialize, Deserialize};

/// FIX message types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixMsgType {
    Heartbeat = b'H' as isize,
    TestRequest = b'1' as isize,
    ResendRequest = b'2' as isize,
    Reject = b'3' as isize,
    SequenceReset = b'4' as isize,
    Logout = b'5' as isize,
    ExecutionReport = b'8' as isize,
    OrderCancelReject = b'9' as isize,
    NewOrderSingle = b'D' as isize,
    NewOrderList = b'E' as isize,
    OrderCancelRequest = b'F' as isize,
    OrderCancelReplaceRequest = b'G' as isize,
    MarketDataSnapshotFullRefresh = b'W' as isize,
    MarketDataIncrementalRefresh = b'X' as isize,
    MarketDataRequest = b'V' as isize,
    MarketDataRequestReject = b'Y' as isize,
}

impl FixMsgType {
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            'H' => Some(Self::Heartbeat),
            '1' => Some(Self::TestRequest),
            '2' => Some(Self::ResendRequest),
            '3' => Some(Self::Reject),
            '4' => Some(Self::SequenceReset),
            '5' => Some(Self::Logout),
            '8' => Some(Self::ExecutionReport),
            '9' => Some(Self::OrderCancelReject),
            'D' => Some(Self::NewOrderSingle),
            'E' => Some(Self::NewOrderList),
            'F' => Some(Self::OrderCancelRequest),
            'G' => Some(Self::OrderCancelReplaceRequest),
            'W' => Some(Self::MarketDataSnapshotFullRefresh),
            'X' => Some(Self::MarketDataIncrementalRefresh),
            'V' => Some(Self::MarketDataRequest),
            'Y' => Some(Self::MarketDataRequestReject),
            _ => None,
        }
    }
    
    pub fn to_char(self) -> char {
        (self as isize as u8) as char
    }
}

/// Standard FIX tags
pub mod fix_tags {
    pub const BEGIN_STRING: usize = 8;
    pub const BODY_LENGTH: usize = 9;
    pub const MSG_TYPE: usize = 35;
    pub const SENDER_COMP_ID: usize = 49;
    pub const TARGET_COMP_ID: usize = 56;
    pub const MSG_SEQ_NUM: usize = 34;
    pub const SENDING_TIME: usize = 52;
    pub const CHECK_SUM: usize = 10;
    pub const POSS_DUP_FLAG: usize = 43;
    pub const TEST_MSG_ID: usize = 112;
    
    // Order related
    pub const ORDER_ID: usize = 11;
    pub const CLIENT_ORDER_ID: usize = 11;
    pub const EXEC_ID: usize = 17;
    pub const EXEC_TYPE: usize = 150;
    pub const ORD_STATUS: usize = 39;
    pub const SIDE: usize = 54;
    pub const ORDER_QTY: usize = 38;
    pub const PRICE: usize = 44;
    pub const SYMBOL: usize = 55;
    pub const SECURITY_TYPE: usize = 167;
    pub const TIME_IN_FORCE: usize = 59;
    pub const ORD_TYPE: usize = 40;
    
    // Market data
    pub const MD_REQ_ID: usize = 262;
    pub const MD_BOOK_TYPE: usize = 264;
    pub const MD_UPDATE_TYPE: usize = 265;
    pub const NO_MD_ENTRY_TYPES: usize = 267;
    pub const NO_MD_ENTRIES: usize = 268;
    pub const MD_ENTRY_TYPE: usize = 269;
    pub const MD_ENTRY_PRICE: usize = 270;
    pub const MD_ENTRY_SIZE: usize = 271;
    pub const MD_ENTRY_DATE: usize = 272;
}

/// FIX field value types
#[derive(Debug, Clone)]
pub enum FixValue {
    String(String),
    Int(i64),
    Float(f64),
    Char(char),
    Timestamp(u64), // Nanoseconds since epoch
}

impl FixValue {
    pub fn to_string(&self) -> String {
        match self {
            Self::String(s) => s.clone(),
            Self::Int(i) => i.to_string(),
            Self::Float(f) => format!("{:.8}", f),
            Self::Char(c) => c.to_string(),
            Self::Timestamp(ts) => format_timestamp(*ts),
        }
    }
}

fn format_timestamp(ns: u64) -> String {
    // FIX timestamp format: YYYYMMDD-HH:MM:SS.sss
    let secs = ns / 1_000_000_000;
    let millis = (ns % 1_000_000_000) / 1_000_000;
    
    let datetime: SystemTime = UNIX_EPOCH + Duration::from_secs(secs);
    // Simplified - in production use proper date formatting
    format!("{}{:03}", secs, millis)
}

/// A single FIX field
#[derive(Debug, Clone)]
pub struct FixField {
    pub tag: usize,
    pub value: FixValue,
}

impl FixField {
    pub fn new(tag: usize, value: FixValue) -> Self {
        Self { tag, value }
    }
    
    pub fn to_fix_string(&self) -> String {
        format!("{}={}\x01", self.tag, self.value.to_string())
    }
}

/// Complete FIX message
#[derive(Debug, Clone)]
pub struct FixMessage {
    pub fields: HashMap<usize, FixValue>,
    pub msg_type: FixMsgType,
}

impl FixMessage {
    pub fn new(msg_type: FixMsgType) -> Self {
        let mut fields = HashMap::new();
        fields.insert(fix_tags::MSG_TYPE, FixValue::Char(msg_type.to_char()));
        
        Self { fields, msg_type }
    }
    
    pub fn add_field(&mut self, tag: usize, value: FixValue) {
        self.fields.insert(tag, value);
    }
    
    pub fn get_field(&self, tag: usize) -> Option<&FixValue> {
        self.fields.get(&tag)
    }
    
    /// Serialize to FIX wire format
    pub fn serialize(&self, begin_string: &str, sender: &str, target: &str, seq_num: u32) -> String {
        let mut body = String::new();
        
        // Add standard header fields
        body.push_str(&format!("{}={}\x01", fix_tags::SENDER_COMP_ID, sender));
        body.push_str(&format!("{}={}\x01", fix_tags::TARGET_COMP_ID, target));
        body.push_str(&format!("{}={}\x01", fix_tags::MSG_SEQ_NUM, seq_num));
        
        // Add current timestamp
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        body.push_str(&format!("{}={}\x01", fix_tags::SENDING_TIME, format_timestamp(now)));
        
        // Add all message fields (except MsgType which is already set)
        let mut sorted_tags: Vec<_> = self.fields.iter().collect();
        sorted_tags.sort_by_key(|(tag, _)| *tag);
        
        for (tag, value) in sorted_tags {
            if *tag != fix_tags::MSG_TYPE {
                body.push_str(&format!("{}={}\x01", tag, value.to_string()));
            }
        }
        
        // Calculate body length (everything after BodyLength tag up to but not including CheckSum)
        let body_length = body.len();
        
        // Build complete message
        let mut message = String::new();
        message.push_str(&format!("{}={}\x01", fix_tags::BEGIN_STRING, begin_string));
        message.push_str(&format!("{}={}\x01", fix_tags::BODY_LENGTH, body_length));
        message.push_str(&body);
        
        // Calculate and append checksum
        let checksum = calculate_checksum(&message);
        message.push_str(&format!("{}={:03}", fix_tags::CHECK_SUM, checksum));
        
        message
    }
    
    /// Parse from FIX wire format
    pub fn parse(fix_str: &str) -> Result<Self, FixError> {
        let fields = parse_fields(fix_str)?;
        
        // Extract MsgType
        let msg_type_char = fields.get(&fix_tags::MSG_TYPE)
            .and_then(|v| match v {
                FixValue::Char(c) => Some(*c),
                _ => None,
            })
            .ok_or(FixError::MissingField(fix_tags::MSG_TYPE))?;
        
        let msg_type = FixMsgType::from_char(msg_type_char)
            .ok_or(FixError::UnknownMessageType(msg_type_char))?;
        
        Ok(Self { fields, msg_type })
    }
    
    /// Verify checksum
    pub fn verify_checksum(&self, fix_str: &str) -> bool {
        if let Some(expected) = self.fields.get(&fix_tags::CHECK_SUM) {
            if let FixValue::Int(expected_cs) = expected {
                let calculated = calculate_checksum(fix_str);
                return calculated == *expected_cs as u8;
            }
        }
        false
    }
}

/// Calculate FIX checksum (sum of all bytes mod 256)
fn calculate_checksum(message: &str) -> u8 {
    message.bytes().fold(0u8, |acc, b| acc.wrapping_add(b))
}

/// Parse FIX fields from wire format
fn parse_fields(fix_str: &str) -> Result<HashMap<usize, FixValue>, FixError> {
    let mut fields = HashMap::new();
    
    for field_str in fix_str.split('\x01').filter(|s| !s.is_empty()) {
        if let Some(eq_pos) = field_str.find('=') {
            let tag: usize = field_str[..eq_pos].parse()
                .map_err(|_| FixError::InvalidTag(field_str[..eq_pos].to_string()))?;
            
            let value_str = &field_str[eq_pos + 1..];
            let value = parse_value(tag, value_str)?;
            
            fields.insert(tag, value);
        }
    }
    
    Ok(fields)
}

/// Parse a FIX field value based on tag type
fn parse_value(tag: usize, value_str: &str) -> Result<FixValue, FixError> {
    // Determine type based on standard FIX tag conventions
    let value = match tag {
        t if t == fix_tags::MSG_TYPE => FixValue::Char(
            value_str.chars().next().ok_or(FixError::InvalidValue(value_str.to_string()))?
        ),
        t if matches!(t, fix_tags::BODY_LENGTH | fix_tags::MSG_SEQ_NUM | fix_tags::ORDER_QTY) => {
            FixValue::Int(value_str.parse().map_err(|_| FixError::InvalidValue(value_str.to_string()))?)
        }
        t if matches!(t, fix_tags::PRICE | fix_tags::MD_ENTRY_PRICE) => {
            FixValue::Float(value_str.parse().map_err(|_| FixError::InvalidValue(value_str.to_string()))?)
        }
        t if t == fix_tags::CHECK_SUM => {
            FixValue::Int(value_str.parse().map_err(|_| FixError::InvalidValue(value_str.to_string()))?)
        }
        _ => FixValue::String(value_str.to_string()),
    };
    
    Ok(value)
}

/// FIX protocol errors
#[derive(Debug, Clone)]
pub enum FixError {
    InvalidFormat(String),
    MissingField(usize),
    UnknownMessageType(char),
    InvalidTag(String),
    InvalidValue(String),
    ChecksumMismatch,
    SequenceGap(u32, u32), // Expected, Received
    ConnectionError(String),
}

impl std::fmt::Display for FixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFormat(s) => write!(f, "Invalid FIX format: {}", s),
            Self::MissingField(tag) => write!(f, "Missing required field: {}", tag),
            Self::UnknownMessageType(c) => write!(f, "Unknown message type: {}", c),
            Self::InvalidTag(s) => write!(f, "Invalid tag: {}", s),
            Self::InvalidValue(s) => write!(f, "Invalid value: {}", s),
            Self::ChecksumMismatch => write!(f, "Checksum mismatch"),
            Self::SequenceGap(exp, recv) => write!(f, "Sequence gap: expected {}, received {}", exp, recv),
            Self::ConnectionError(s) => write!(f, "Connection error: {}", s),
        }
    }
}

impl std::error::Error for FixError {}

/// FIX session state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixSessionState {
    Disconnected,
    Connecting,
    LoggedOn,
    LoggingOut,
    Resetting,
}

/// FIX session configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixSessionConfig {
    pub begin_string: String,
    pub sender_comp_id: String,
    pub target_comp_id: String,
    pub heartbeat_interval_secs: u32,
    pub reconnect_delay_secs: u32,
    pub reset_on_disconnect: bool,
    pub persist_messages: bool,
}

impl Default for FixSessionConfig {
    fn default() -> Self {
        Self {
            begin_string: "FIX.4.4".to_string(),
            sender_comp_id: "TRADER".to_string(),
            target_comp_id: "BROKER".to_string(),
            heartbeat_interval_secs: 30,
            reconnect_delay_secs: 5,
            reset_on_disconnect: false,
            persist_messages: true,
        }
    }
}

/// FIX Session manager
pub struct FixSession {
    config: FixSessionConfig,
    state: RwLock<FixSessionState>,
    outgoing_seq_num: RwLock<u32>,
    incoming_seq_num: RwLock<u32>,
    last_message_time: RwLock<Option<Instant>>,
    test_request_id: RwLock<Option<String>>,
}

impl FixSession {
    pub fn new(config: FixSessionConfig) -> Self {
        Self {
            config,
            state: RwLock::new(FixSessionState::Disconnected),
            outgoing_seq_num: RwLock::new(1),
            incoming_seq_num: RwLock::new(1),
            last_message_time: RwLock::new(None),
            test_request_id: RwLock::new(None),
        }
    }
    
    /// Create a logon message
    pub fn create_logon(&self) -> FixMessage {
        let mut msg = FixMessage::new(FixMsgType::Logon);
        msg.add_field(fix_tags::ENCRYPT_METHOD, FixValue::String("NONE".to_string()));
        msg.add_field(fix_tags::HEART_BT_INT, FixValue::Int(self.config.heartbeat_interval_secs as i64));
        msg
    }
    
    /// Create a heartbeat message
    pub fn create_heartbeat(&self) -> FixMessage {
        FixMessage::new(FixMsgType::Heartbeat)
    }
    
    /// Create a test request message
    pub fn create_test_request(&self, test_id: &str) -> FixMessage {
        let mut msg = FixMessage::new(FixMsgType::TestRequest);
        msg.add_field(fix_tags::TEST_MSG_ID, FixValue::String(test_id.to_string()));
        *self.test_request_id.write() = Some(test_id.to_string());
        msg
    }
    
    /// Create a logout message
    pub fn create_logout(&self, reason: Option<&str>) -> FixMessage {
        let mut msg = FixMessage::new(FixMsgType::Logout);
        if let Some(r) = reason {
            msg.add_field(fix_tags::TEXT, FixValue::String(r.to_string()));
        }
        msg
    }
    
    /// Create a new order single message
    pub fn create_new_order_single(
        &self,
        client_order_id: &str,
        symbol: &str,
        side: char, // '1'=Buy, '2'=Sell
        order_type: char, // '1'=Market, '2'=Limit
        quantity: f64,
        price: Option<f64>,
        time_in_force: char, // '0'=Day, '1'=GTC, etc.
    ) -> FixMessage {
        let mut msg = FixMessage::new(FixMsgType::NewOrderSingle);
        
        msg.add_field(fix_tags::CLIENT_ORDER_ID, FixValue::String(client_order_id.to_string()));
        msg.add_field(fix_tags::SYMBOL, FixValue::String(symbol.to_string()));
        msg.add_field(fix_tags::SIDE, FixValue::Char(side));
        msg.add_field(fix_tags::ORD_TYPE, FixValue::Char(order_type));
        msg.add_field(fix_tags::ORDER_QTY, FixValue::Float(quantity));
        msg.add_field(fix_tags::TIME_IN_FORCE, FixValue::Char(time_in_force));
        
        if let Some(p) = price {
            msg.add_field(fix_tags::PRICE, FixValue::Float(p));
        }
        
        msg
    }
    
    /// Create an order cancel request
    pub fn create_order_cancel(
        &self,
        orig_client_order_id: &str,
        symbol: &str,
        side: char,
    ) -> FixMessage {
        let mut msg = FixMessage::new(FixMsgType::OrderCancelRequest);
        
        msg.add_field(fix_tags::CLIENT_ORDER_ID, FixValue::String(orig_client_order_id.to_string()));
        msg.add_field(fix_tags::ORIG_CL_ORD_ID, FixValue::String(orig_client_order_id.to_string()));
        msg.add_field(fix_tags::SYMBOL, FixValue::String(symbol.to_string()));
        msg.add_field(fix_tags::SIDE, FixValue::Char(side));
        
        msg
    }
    
    /// Get next outgoing sequence number
    pub fn next_outgoing_seq(&self) -> u32 {
        let mut seq = self.outgoing_seq_num.write();
        let current = *seq;
        *seq += 1;
        current
    }
    
    /// Process incoming sequence number
    pub fn process_incoming_seq(&self, seq_num: u32) -> Result<(), FixError> {
        let mut expected = self.incoming_seq_num.write();
        
        if seq_num < *expected {
            // Duplicate or resend
            Ok(())
        } else if seq_num == *expected {
            *expected += 1;
            Ok(())
        } else {
            // Gap detected
            Err(FixError::SequenceGap(*expected, seq_num))
        }
    }
    
    /// Get current session state
    pub fn get_state(&self) -> FixSessionState {
        *self.state.read()
    }
    
    /// Set session state
    pub fn set_state(&self, state: FixSessionState) {
        info!("FIX session state changed: {:?}", state);
        *self.state.write() = state;
        
        if state == FixSessionState::Disconnected || state == FixSessionState::LoggedOn {
            *self.last_message_time.write() = Some(Instant::now());
        }
    }
    
    /// Check if heartbeat timeout occurred
    pub fn check_heartbeat_timeout(&self) -> bool {
        if let Some(last_time) = *self.last_message_time.read() {
            let elapsed = last_time.elapsed().as_secs();
            return elapsed > (self.config.heartbeat_interval_secs * 2) as u64;
        }
        false
    }
}

// Additional standard FIX tags not defined earlier
impl fix_tags {
    pub const ENCRYPT_METHOD: usize = 98;
    pub const HEART_BT_INT: usize = 108;
    pub const ORIG_CL_ORD_ID: usize = 41;
    pub const TEXT: usize = 58;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_msg_type_conversion() {
        assert_eq!(FixMsgType::from_char('8'), Some(FixMsgType::ExecutionReport));
        assert_eq!(FixMsgType::from_char('D'), Some(FixMsgType::NewOrderSingle));
        assert_eq!(FixMsgType::from_char('Z'), None);
    }
    
    #[test]
    fn test_checksum_calculation() {
        let test_msg = "8=FIX.4.4\x019=100\x01";
        let checksum = calculate_checksum(test_msg);
        assert!(checksum > 0);
    }
    
    #[test]
    fn test_fix_message_creation() {
        let msg = FixMessage::new(FixMsgType::Heartbeat);
        assert_eq!(msg.msg_type, FixMsgType::Heartbeat);
    }
    
    #[test]
    fn test_session_config() {
        let config = FixSessionConfig::default();
        assert_eq!(config.begin_string, "FIX.4.4");
        assert_eq!(config.heartbeat_interval_secs, 30);
    }
}
