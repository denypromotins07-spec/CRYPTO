//! Local UDP Multicast for Zero-Copy Internal Data Broadcasting
//! 
//! Broadcasts parsed market data internally from the main network thread to multiple 
//! worker threads (strategy, risk, logging) with zero-copy semantics, eliminating 
//! the need for multiple parsing instances.
//!
//! Target Hardware: AMD Ryzen AI 5 + AMD Radeon GPU + 16GB RAM (8GB cap)

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;
use parking_lot::RwLock;
use log::{info, warn, error, debug};
use serde::{Serialize, Deserialize};

use crate::types::{Symbol, Price, Quantity, Timestamp, Side};

/// Configuration for UDP multicast
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UdpMulticastConfig {
    /// Multicast group address (224.0.0.0 - 239.255.255.255)
    pub multicast_addr: String,
    /// Port for multicast
    pub port: u16,
    /// Network interface to bind to
    pub interface_addr: String,
    /// Time-to-live for multicast packets
    pub ttl: u32,
    /// Enable loopback (receive own messages)
    pub loopback: bool,
    /// Buffer size in bytes
    pub buffer_size: usize,
}

impl Default for UdpMulticastConfig {
    fn default() -> Self {
        Self {
            // Use a private multicast range
            multicast_addr: "239.255.100.1".to_string(),
            port: 9000,
            interface_addr: "0.0.0.0".to_string(),
            ttl: 1, // Don't leave local network
            loopback: true,
            buffer_size: 65536,
        }
    }
}

/// Market data message format for multicast
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MarketDataPacket {
    /// Message type identifier
    pub msg_type: u8,
    /// Symbol ID (mapped from Symbol enum)
    pub symbol_id: u16,
    /// Padding for alignment
    pub _padding: u16,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Last price
    pub last_price: f64,
    /// Last quantity
    pub last_qty: f64,
    /// Bid price (best)
    pub bid_price: f64,
    /// Bid quantity
    pub bid_qty: f64,
    /// Ask price (best)
    pub ask_price: f64,
    /// Ask quantity
    pub ask_qty: f64,
    /// Sequence number
    pub sequence: u64,
    /// Checksum for integrity
    pub checksum: u32,
}

impl MarketDataPacket {
    pub const SIZE: usize = std::mem::size_of::<Self>();
    
    pub fn new() -> Self {
        Self {
            msg_type: 0,
            symbol_id: 0,
            _padding: 0,
            timestamp_ns: 0,
            last_price: 0.0,
            last_qty: 0.0,
            bid_price: 0.0,
            bid_qty: 0.0,
            ask_price: 0.0,
            ask_qty: 0.0,
            sequence: 0,
            checksum: 0,
        }
    }
    
    /// Calculate simple checksum
    pub fn calculate_checksum(&self) -> u32 {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                Self::SIZE - 4, // Exclude checksum field
            )
        };
        
        bytes.iter().fold(0u32, |acc, &b| acc.wrapping_add(b as u32))
    }
    
    /// Verify checksum
    pub fn verify_checksum(&self) -> bool {
        self.checksum == self.calculate_checksum()
    }
    
    /// Set checksum
    pub fn set_checksum(&mut self) {
        self.checksum = self.calculate_checksum();
    }
    
    /// Convert from tick data
    pub fn from_tick(
        symbol: Symbol,
        timestamp_ns: u64,
        price: f64,
        qty: f64,
        side: Side,
        sequence: u64,
    ) -> Self {
        let mut packet = Self::new();
        packet.msg_type = 1; // Tick message
        packet.symbol_id = symbol as u16;
        packet.timestamp_ns = timestamp_ns;
        packet.last_price = price;
        packet.last_qty = qty;
        packet.sequence = sequence;
        packet.set_checksum();
        packet
    }
    
    /// Convert from order book snapshot
    pub fn from_orderbook(
        symbol: Symbol,
        timestamp_ns: u64,
        bid_price: f64,
        bid_qty: f64,
        ask_price: f64,
        ask_qty: f64,
        sequence: u64,
    ) -> Self {
        let mut packet = Self::new();
        packet.msg_type = 2; // Order book message
        packet.symbol_id = symbol as u16;
        packet.timestamp_ns = timestamp_ns;
        packet.bid_price = bid_price;
        packet.bid_qty = bid_qty;
        packet.ask_price = ask_price;
        packet.ask_qty = ask_qty;
        packet.sequence = sequence;
        packet.set_checksum();
        packet
    }
}

/// UDP Multicast sender for broadcasting market data
pub struct UdpMulticastSender {
    config: UdpMulticastConfig,
    socket: Option<std::net::UdpSocket>,
    sequence_counter: RwLock<u64>,
}

impl UdpMulticastSender {
    /// Create and initialize multicast sender
    pub fn new(config: UdpMulticastConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let addr: SocketAddr = SocketAddr::new(
            IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            config.port,
        );
        
        let socket = std::net::UdpSocket::bind(addr)?;
        
        // Set TTL
        socket.set_ttl(config.ttl)?;
        
        // Set buffer sizes
        socket.set_send_buffer_size(config.buffer_size)?;
        
        info!(
            "UDP Multicast sender initialized on {}:{} (TTL={})",
            config.multicast_addr, config.port, config.ttl
        );
        
        Ok(Self {
            config,
            socket: Some(socket),
            sequence_counter: RwLock::new(0),
        })
    }
    
    /// Join multicast group
    pub fn join_group(&self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref socket) = self.socket {
            let multicast_ip: Ipv4Addr = self.config.multicast_addr.parse()?;
            let interface = Ipv4Addr::UNSPECIFIED;
            
            socket.join_multicast_v4(&multicast_ip, &interface)?;
            info!("Joined multicast group {}", self.config.multicast_addr);
        }
        
        Ok(())
    }
    
    /// Send market data packet
    pub fn send(&self, mut packet: MarketDataPacket) -> Result<usize, Box<dyn std::error::Error>> {
        if let Some(ref socket) = self.socket {
            // Update sequence number
            let mut seq = self.sequence_counter.write();
            *seq += 1;
            packet.sequence = *seq;
            packet.set_checksum();
            
            let multicast_addr: SocketAddr = SocketAddr::new(
                self.config.multicast_addr.parse::<IpAddr>()?,
                self.config.port,
            );
            
            // Zero-copy send using raw bytes
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    &packet as *const MarketDataPacket as *const u8,
                    MarketDataPacket::SIZE,
                )
            };
            
            let sent = socket.send_to(bytes, multicast_addr)?;
            debug!("Sent multicast packet: seq={}, symbol={}", packet.sequence, packet.symbol_id);
            
            Ok(sent)
        } else {
            Err("Socket not initialized".into())
        }
    }
    
    /// Broadcast tick data
    pub fn broadcast_tick(
        &self,
        symbol: Symbol,
        timestamp_ns: u64,
        price: f64,
        qty: f64,
        side: Side,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let packet = MarketDataPacket::from_tick(symbol, timestamp_ns, price, qty, side, 0);
        self.send(packet)
    }
    
    /// Broadcast order book update
    pub fn broadcast_orderbook(
        &self,
        symbol: Symbol,
        timestamp_ns: u64,
        bid_price: f64,
        bid_qty: f64,
        ask_price: f64,
        ask_qty: f64,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let packet = MarketDataPacket::from_orderbook(
            symbol, timestamp_ns, bid_price, bid_qty, ask_price, ask_qty, 0
        );
        self.send(packet)
    }
    
    /// Close the socket
    pub fn close(&mut self) {
        self.socket = None;
        info!("UDP Multicast sender closed");
    }
}

/// UDP Multicast receiver for subscribing to market data
pub struct UdpMulticastReceiver {
    config: UdpMulticastConfig,
    socket: Option<std::net::UdpSocket>,
    running: RwLock<bool>,
    received_count: RwLock<u64>,
    dropped_count: RwLock<u64>,
}

impl UdpMulticastReceiver {
    /// Create and initialize multicast receiver
    pub fn new(config: UdpMulticastConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let addr: SocketAddr = SocketAddr::new(
            IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            config.port,
        );
        
        let socket = std::net::UdpSocket::bind(addr)?;
        
        // Set receive buffer size (important for high throughput)
        socket.set_recv_buffer_size(config.buffer_size)?;
        
        // Set read timeout
        socket.set_read_timeout(Some(Duration::from_millis(100)))?;
        
        info!(
            "UDP Multicast receiver initialized on port {} (buffer={}KB)",
            config.port, config.buffer_size / 1024
        );
        
        Ok(Self {
            config,
            socket: Some(socket),
            running: RwLock::new(false),
            received_count: RwLock::new(0),
            dropped_count: RwLock::new(0),
        })
    }
    
    /// Join multicast group
    pub fn join_group(&self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref socket) = self.socket {
            let multicast_ip: Ipv4Addr = self.config.multicast_addr.parse()?;
            let interface = Ipv4Addr::UNSPECIFIED;
            
            socket.join_multicast_v4(&multicast_ip, &interface)?;
            info!("Joined multicast group {}", self.config.multicast_addr);
        }
        
        Ok(())
    }
    
    /// Receive a single packet (blocking with timeout)
    pub fn receive(&self) -> Option<MarketDataPacket> {
        if let Some(ref socket) = self.socket {
            let mut buffer = vec![0u8; MarketDataPacket::SIZE];
            
            match socket.recv_from(&mut buffer) {
                Ok((len, _addr)) => {
                    if len != MarketDataPacket::SIZE {
                        warn!("Received packet of unexpected size: {}", len);
                        *self.dropped_count.write() += 1;
                        return None;
                    }
                    
                    // Zero-copy interpretation of received bytes
                    let packet = unsafe {
                        *(buffer.as_ptr() as *const MarketDataPacket)
                    };
                    
                    // Verify checksum
                    if !packet.verify_checksum() {
                        warn!("Checksum verification failed for packet seq={}", packet.sequence);
                        *self.dropped_count.write() += 1;
                        return None;
                    }
                    
                    *self.received_count.write() += 1;
                    debug!("Received multicast packet: seq={}, symbol={}", packet.sequence, packet.symbol_id);
                    
                    Some(packet)
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => None,
                Err(e) => {
                    error!("Error receiving multicast packet: {}", e);
                    None
                }
            }
        } else {
            None
        }
    }
    
    /// Start continuous reception in background
    pub fn start_receiver<F>(&self, mut handler: F) 
    where
        F: FnMut(MarketDataPacket) + Send + 'static,
    {
        *self.running.write() = true;
        
        let socket_opt = self.socket.clone();
        let running = self.running.clone();
        let dropped = self.dropped_count.clone();
        let received = self.received_count.clone();
        
        std::thread::spawn(move || {
            let socket = match socket_opt {
                Some(s) => s,
                None => return,
            };
            
            let mut buffer = vec![0u8; MarketDataPacket::SIZE];
            
            while *running.read() {
                match socket.recv_from(&mut buffer) {
                    Ok((len, _addr)) => {
                        if len != MarketDataPacket::SIZE {
                            *dropped.write() += 1;
                            continue;
                        }
                        
                        let packet = unsafe {
                            *(buffer.as_ptr() as *const MarketDataPacket)
                        };
                        
                        if packet.verify_checksum() {
                            *received.write() += 1;
                            handler(packet);
                        } else {
                            *dropped.write() += 1;
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => continue,
                    Err(e) => {
                        error!("Receive error: {}", e);
                        break;
                    }
                }
            }
        });
    }
    
    /// Stop receiver
    pub fn stop(&self) {
        *self.running.write() = false;
        info!("UDP Multicast receiver stopped");
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> UdpStats {
        UdpStats {
            received: *self.received_count.read(),
            dropped: *self.dropped_count.read(),
            drop_rate: {
                let total = *self.received_count.read() + *self.dropped_count.read();
                if total > 0 {
                    *self.dropped_count.read() as f64 / total as f64
                } else {
                    0.0
                }
            },
        }
    }
    
    /// Leave multicast group
    pub fn leave_group(&self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref socket) = self.socket {
            let multicast_ip: Ipv4Addr = self.config.multicast_addr.parse()?;
            let interface = Ipv4Addr::UNSPECIFIED;
            
            socket.leave_multicast_v4(&multicast_ip, &interface)?;
            info!("Left multicast group {}", self.config.multicast_addr);
        }
        
        Ok(())
    }
    
    /// Close the socket
    pub fn close(&mut self) {
        self.stop();
        self.socket = None;
        info!("UDP Multicast receiver closed");
    }
}

/// Statistics for UDP multicast
#[derive(Debug, Clone)]
pub struct UdpStats {
    pub received: u64,
    pub dropped: u64,
    pub drop_rate: f64,
}

/// Shared memory ring buffer for zero-copy inter-thread communication
pub struct ZeroCopyRingBuffer<T> {
    buffer: Arc<RwLock<Vec<T>>>,
    write_index: Arc<RwLock<usize>>,
    read_index: Arc<RwLock<usize>>,
    capacity: usize,
}

impl<T: Clone + Default> ZeroCopyRingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        let buffer = vec![T::default(); capacity];
        
        Self {
            buffer: Arc::new(RwLock::new(buffer)),
            write_index: Arc::new(RwLock::new(0)),
            read_index: Arc::new(RwLock::new(0)),
            capacity,
        }
    }
    
    /// Write an item (producer side)
    pub fn write(&self, item: T) -> bool {
        let mut write_idx = self.write_index.write();
        let read_idx = *self.read_index.read();
        
        // Check if buffer is full
        let next_write = (*write_idx + 1) % self.capacity;
        if next_write == read_idx {
            return false; // Buffer full
        }
        
        let mut buffer = self.buffer.write();
        buffer[*write_idx] = item;
        *write_idx = next_write;
        
        true
    }
    
    /// Read an item (consumer side)
    pub fn read(&self) -> Option<T> {
        let mut read_idx = self.read_index.write();
        let write_idx = *self.write_index.read();
        
        // Check if buffer is empty
        if *read_idx == write_idx {
            return None;
        }
        
        let buffer = self.buffer.read();
        let item = buffer[*read_idx].clone();
        *read_idx = (*read_idx + 1) % self.capacity;
        
        Some(item)
    }
    
    /// Get current fill level
    pub fn fill_level(&self) -> usize {
        let write_idx = *self.write_index.read();
        let read_idx = *self.read_index.read();
        
        if write_idx >= read_idx {
            write_idx - read_idx
        } else {
            self.capacity - read_idx + write_idx
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_packet_size() {
        assert_eq!(MarketDataPacket::SIZE, 96); // 8+8+8*6+8+4 = 96 bytes
    }
    
    #[test]
    fn test_checksum() {
        let mut packet = MarketDataPacket::new();
        packet.symbol_id = 1;
        packet.last_price = 50000.0;
        packet.set_checksum();
        
        assert!(packet.verify_checksum());
        
        // Modify data
        packet.last_price = 50001.0;
        assert!(!packet.verify_checksum());
    }
    
    #[test]
    fn test_ring_buffer() {
        let buffer = ZeroCopyRingBuffer::<i32>::new(10);
        
        assert!(buffer.write(1));
        assert!(buffer.write(2));
        assert!(buffer.write(3));
        
        assert_eq!(buffer.fill_level(), 3);
        assert_eq!(buffer.read(), Some(1));
        assert_eq!(buffer.read(), Some(2));
        assert_eq!(buffer.fill_level(), 1);
    }
}
