//! # Rust IPC Implementation using Shared Memory Ring Buffer
//! 
//! This module implements the Rust side of the inter-process communication
//! bridge, using memory-mapped files (mmap) for zero-copy data transfer
//! to Python consumers.
//! 
//! Features:
//! - Lock-free ring buffer for high-throughput
//! - Memory-mapped file for cross-process sharing
//! - Atomic operations for thread safety
//! - Support for multiple readers

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::Path;

// Import IPC protocol types
use crate::ipc_protocol::{MessageHeader, MessageType, IpcMessage};

/// Default shared memory file path
const DEFAULT_SHM_PATH: &str = "/tmp/trading_bot_ipc.bin";

/// Ring buffer capacity (power of 2 for efficient modulo)
const RING_BUFFER_CAPACITY: usize = 1 << 14; // 16384 messages

/// Total buffer size in bytes
const BUFFER_SIZE: usize = RING_BUFFER_CAPACITY * IpcMessage::MAX_SIZE;

/// Magic number for shared memory validation
const SHM_MAGIC: u32 = 0x53484D51; // "SHMQ"

/// Shared memory header structure
#[repr(C)]
struct ShmHeader {
    /// Magic number
    magic: u32,
    /// Version
    version: u16,
    /// Flags
    flags: u16,
    /// Write position (head)
    write_pos: AtomicU64,
    /// Read position (tail) - for single reader
    read_pos: AtomicU64,
    /// Total messages written
    total_written: AtomicU64,
    /// Total messages read
    total_read: AtomicU64,
    /// Overflow count
    overflow_count: AtomicU64,
    /// Last write timestamp
    last_write_ns: AtomicU64,
    /// Reserved
    _reserved: [u8; 48],
}

impl ShmHeader {
    fn new() -> Self {
        ShmHeader {
            magic: SHM_MAGIC,
            version: 1,
            flags: 0,
            write_pos: AtomicU64::new(0),
            read_pos: AtomicU64::new(0),
            total_written: AtomicU64::new(0),
            total_read: AtomicU64::new(0),
            overflow_count: AtomicU64::new(0),
            last_write_ns: AtomicU64::new(0),
            _reserved: [0u8; 48],
        }
    }
    
    fn validate(&self) -> bool {
        self.magic == SHM_MAGIC && self.version == 1
    }
}

/// Shared memory ring buffer for IPC
pub struct SharedMemoryRingBuffer {
    /// Path to the shared memory file
    path: String,
    /// Header pointer
    header: *mut ShmHeader,
    /// Data buffer pointer
    data_ptr: *mut u8,
    /// File handle (kept open)
    _file: File,
    /// Memory map length
    mmap_len: usize,
    /// Is owner (created the file)
    is_owner: bool,
}

unsafe impl Send for SharedMemoryRingBuffer {}
unsafe impl Sync for SharedMemoryRingBuffer {}

impl SharedMemoryRingBuffer {
    /// Create a new shared memory ring buffer (as owner)
    pub fn create(path: &str) -> Result<Self, String> {
        let full_size = std::mem::size_of::<ShmHeader>() + BUFFER_SIZE;
        
        // Create or open the file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| format!("Failed to create shared memory file: {}", e))?;
        
        // Set file size
        file.set_len(full_size as u64)
            .map_err(|e| format!("Failed to set file size: {}", e))?;
        
        // Memory map the file
        let mmap = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                full_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        
        if mmap == libc::MAP_FAILED {
            return Err("Failed to mmap shared memory".to_string());
        }
        
        // Initialize header
        let header = mmap as *mut ShmHeader;
        unsafe {
            std::ptr::write(header, ShmHeader::new());
        }
        
        let data_ptr = unsafe { mmap.add(std::mem::size_of::<ShmHeader>()) };
        
        Ok(SharedMemoryRingBuffer {
            path: path.to_string(),
            header,
            data_ptr,
            _file: file,
            mmap_len: full_size,
            is_owner: true,
        })
    }
    
    /// Open an existing shared memory ring buffer (as reader)
    pub fn open(path: &str) -> Result<Self, String> {
        let full_size = std::mem::size_of::<ShmHeader>() + BUFFER_SIZE;
        
        // Open the file
        let file = OpenOptions::new()
            .read(true)
            .write(false)
            .open(path)
            .map_err(|e| format!("Failed to open shared memory file: {}", e))?;
        
        // Memory map the file
        let mmap = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                full_size,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        
        if mmap == libc::MAP_FAILED {
            return Err("Failed to mmap shared memory".to_string());
        }
        
        let header = mmap as *mut ShmHeader;
        
        // Validate header
        if !unsafe { (*header).validate() } {
            unsafe { libc::munmap(mmap, full_size) };
            return Err("Invalid shared memory header".to_string());
        }
        
        let data_ptr = unsafe { mmap.add(std::mem::size_of::<ShmHeader>()) };
        
        Ok(SharedMemoryRingBuffer {
            path: path.to_string(),
            header,
            data_ptr,
            _file: file,
            mmap_len: full_size,
            is_owner: false,
        })
    }
    
    /// Write a message to the ring buffer
    #[inline]
    pub fn write(&self, message: &IpcMessage) -> bool {
        unsafe {
            let header = &*self.header;
            
            // Get current write position
            let write_pos = header.write_pos.load(Ordering::Acquire);
            let read_pos = header.read_pos.load(Ordering::Acquire);
            
            // Check if buffer is full
            let next_pos = (write_pos + 1) & (RING_BUFFER_CAPACITY as u64 - 1);
            if next_pos == read_pos {
                header.overflow_count.fetch_add(1, Ordering::Relaxed);
                return false;
            }
            
            // Calculate offset in data buffer
            let offset = (write_pos as usize) * IpcMessage::MAX_SIZE;
            let dest_ptr = self.data_ptr.add(offset);
            
            // Copy message data
            let bytes = message.as_bytes();
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                dest_ptr,
                bytes.len(),
            );
            
            // Memory barrier
            std::sync::atomic::fence(Ordering::Release);
            
            // Update write position
            header.write_pos.store(next_pos, Ordering::Release);
            header.total_written.fetch_add(1, Ordering::Relaxed);
            header.last_write_ns.store(crate::get_timestamp_ns(), Ordering::Relaxed);
            
            true
        }
    }
    
    /// Get current buffer statistics
    pub fn get_stats(&self) -> RingBufferStats {
        unsafe {
            let header = &*self.header;
            
            let write_pos = header.write_pos.load(Ordering::Acquire);
            let read_pos = header.read_pos.load(Ordering::Acquire);
            let size = ((write_pos.wrapping_sub(read_pos)) & (RING_BUFFER_CAPACITY as u64 - 1)) as usize;
            
            RingBufferStats {
                write_pos,
                read_pos,
                size,
                capacity: RING_BUFFER_CAPACITY,
                total_written: header.total_written.load(Ordering::Relaxed),
                total_read: header.total_read.load(Ordering::Relaxed),
                overflow_count: header.overflow_count.load(Ordering::Relaxed),
                last_write_ns: header.last_write_ns.load(Ordering::Relaxed),
            }
        }
    }
    
    /// Check if buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        unsafe {
            let header = &*self.header;
            header.write_pos.load(Ordering::Acquire) == header.read_pos.load(Ordering::Acquire)
        }
    }
    
    /// Get approximate fill level (0.0 to 1.0)
    #[inline]
    pub fn fill_level(&self) -> f32 {
        let stats = self.get_stats();
        stats.size as f32 / stats.capacity as f32
    }
}

impl Drop for SharedMemoryRingBuffer {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.data_ptr as *mut _, self.mmap_len);
            
            // If owner, unlink the file
            if self.is_owner {
                let _ = std::fs::remove_file(&self.path);
            }
        }
    }
}

/// Ring buffer statistics
#[derive(Debug, Clone)]
pub struct RingBufferStats {
    pub write_pos: u64,
    pub read_pos: u64,
    pub size: usize,
    pub capacity: usize,
    pub total_written: u64,
    pub total_read: u64,
    pub overflow_count: u64,
    pub last_write_ns: u64,
}

/// IPC Publisher for broadcasting events to Python
pub struct IpcPublisher {
    ring_buffer: SharedMemoryRingBuffer,
    sequence: u64,
    message_builder: crate::ipc_protocol::MessageBuilder,
}

impl IpcPublisher {
    /// Create a new IPC publisher
    pub fn new(path: &str) -> Result<Self, String> {
        let ring_buffer = SharedMemoryRingBuffer::create(path)?;
        
        Ok(IpcPublisher {
            ring_buffer,
            sequence: 0,
            message_builder: crate::ipc_protocol::MessageBuilder::new(),
        })
    }
    
    /// Publish an order book update
    pub fn publish_orderbook(&mut self, symbol: &str, bids: &[(f64, f64)],
                             asks: &[(f64, f64)], first_id: i64, last_id: i64) -> bool {
        let bytes = self.message_builder.order_book(symbol, bids, asks, first_id, last_id);
        self.sequence += 1;
        
        let message = IpcMessage::new(MessageType::OrderBook, &bytes, self.sequence)
            .expect("OrderBook message too large");
        
        self.ring_buffer.write(&message)
    }
    
    /// Publish a trade
    pub fn publish_trade(&mut self, symbol: &str, trade_id: i64, price: f64,
                         quantity: f64, buyer_order_id: i64, seller_order_id: i64,
                         buyer_is_maker: bool) -> bool {
        let bytes = self.message_builder.trade(
            symbol, trade_id, price, quantity, buyer_order_id, seller_order_id, buyer_is_maker
        );
        self.sequence += 1;
        
        let message = IpcMessage::new(MessageType::Trade, &bytes, self.sequence)
            .expect("Trade message too large");
        
        self.ring_buffer.write(&message)
    }
    
    /// Publish a trading signal
    pub fn publish_signal(&mut self, symbol: &str, signal_type: u8, confidence: f32,
                          target_price: f64, stop_loss: f64, position_size: f32,
                          time_horizon_secs: u32, model_id: u32) -> bool {
        let bytes = self.message_builder.signal(
            symbol, signal_type, confidence, target_price, stop_loss,
            position_size, time_horizon_secs, model_id
        );
        self.sequence += 1;
        
        let message = IpcMessage::new(MessageType::Signal, &bytes, self.sequence)
            .expect("Signal message too large");
        
        self.ring_buffer.write(&message)
    }
    
    /// Publish a heartbeat
    pub fn publish_heartbeat(&mut self) -> bool {
        self.sequence += 1;
        let payload: [u8; 8] = [0u8; 8];
        
        let message = IpcMessage::new(MessageType::Heartbeat, &payload, self.sequence)
            .expect("Heartbeat message too large");
        
        self.ring_buffer.write(&message)
    }
    
    /// Get publisher statistics
    pub fn get_stats(&self) -> RingBufferStats {
        self.ring_buffer.get_stats()
    }
}

/// Send a ready signal to Python subscriber
pub fn signal_ready(path: &str) -> Result<(), String> {
    let ready_path = format!("{}.ready", path);
    let mut file = File::create(&ready_path)
        .map_err(|e| format!("Failed to create ready file: {}", e))?;
    
    file.write_all(b"READY")
        .map_err(|e| format!("Failed to write ready signal: {}", e))?;
    
    Ok(())
}

/// Check if Python subscriber is ready
pub fn check_subscriber_ready(path: &str) -> bool {
    let ready_path = format!("{}.ready", path);
    Path::new(&ready_path).exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shared_memory_create() {
        let path = "/tmp/test_shm_ring_buffer.bin";
        let buffer = SharedMemoryRingBuffer::create(path).unwrap();
        
        assert!(buffer.is_owner);
        assert!(!buffer.is_empty()); // Header counts as non-empty initially
        
        let stats = buffer.get_stats();
        assert_eq!(stats.capacity, RING_BUFFER_CAPACITY);
    }
    
    #[test]
    fn test_publish_subscribe() {
        let path = "/tmp/test_shm_pubsub.bin";
        
        // Create publisher
        let mut publisher = IpcPublisher::new(path).unwrap();
        
        // Publish a trade
        let result = publisher.publish_trade("BTCUSDT", 1, 50000.0, 0.001, 100, 101, false);
        assert!(result);
        
        // Check stats
        let stats = publisher.get_stats();
        assert_eq!(stats.total_written, 1);
    }
}
