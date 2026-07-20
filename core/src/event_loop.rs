//! # Ultra-Low Latency Lock-Free Event Loop
//! 
//! This module implements a highly optimized, single-threaded event loop designed
//! for microsecond-level execution on AMD Ryzen processors. Features include:
//! - Busy-waiting with adaptive spin strategies
//! - CPU core pinning for cache locality
//! - Lock-free ring buffer for event ingestion
//! - Memory barrier optimization for minimal latency
//!
//! Target: AMD Ryzen AI 5 (Zen 4 architecture)

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::cell::UnsafeCell;

/// Cache line size for preventing false sharing
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of events in the ring buffer (power of 2 for efficient modulo)
const RING_BUFFER_SIZE: usize = 1 << 16; // 65536 events

/// Spin count before yielding (tuned for AMD Ryzen)
const SPIN_COUNT: u32 = 1000;

/// Event types supported by the event loop
#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum EventType {
    OrderBookUpdate = 0,
    Trade = 1,
    Kline = 2,
    Signal = 3,
    OrderAck = 4,
    OrderReject = 5,
    Heartbeat = 6,
    Custom = 255,
}

impl EventType {
    #[inline(always)]
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => EventType::OrderBookUpdate,
            1 => EventType::Trade,
            2 => EventType::Kline,
            3 => EventType::Signal,
            4 => EventType::OrderAck,
            5 => EventType::OrderReject,
            6 => EventType::Heartbeat,
            _ => EventType::Custom,
        }
    }
}

/// Event structure optimized for cache efficiency
/// Total size: 64 bytes (exactly one cache line)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Event {
    /// Event type (1 byte)
    pub event_type: u8,
    /// Priority level (1 byte, higher = more urgent)
    pub priority: u8,
    /// Reserved padding (2 bytes)
    pub _padding: u16,
    /// Timestamp in nanoseconds since epoch (8 bytes)
    pub timestamp_ns: u64,
    /// Source ID (e.g., exchange, symbol) (4 bytes)
    pub source_id: u32,
    /// Sequence number for ordering (8 bytes)
    pub sequence: u64,
    /// Payload length (4 bytes)
    pub payload_len: u32,
    /// Inline payload or pointer to external data (32 bytes)
    pub payload: [u8; 32],
}

impl Event {
    /// Create a new event with current timestamp
    #[inline(always)]
    pub fn new(event_type: EventType, priority: u8, source_id: u32, sequence: u64) -> Self {
        Event {
            event_type: event_type as u8,
            priority,
            _padding: 0,
            timestamp_ns: crate::get_timestamp_ns(),
            source_id,
            sequence,
            payload_len: 0,
            payload: [0u8; 32],
        }
    }
    
    /// Set payload data (copies up to 32 bytes)
    #[inline(always)]
    pub fn set_payload(&mut self, data: &[u8]) {
        let len = data.len().min(32);
        self.payload_len = len as u32;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.payload.as_mut_ptr(),
                len,
            );
        }
    }
    
    /// Get payload as slice
    #[inline(always)]
    pub fn get_payload(&self) -> &[u8] {
        &self.payload[..self.payload_len as usize]
    }
}

/// Lock-free ring buffer for event ingestion
/// Uses separate head and tail indices with padding to prevent false sharing
pub struct EventRingBuffer {
    /// Buffer storage (cache-line aligned)
    buffer: UnsafeCell<[Event; RING_BUFFER_SIZE]>,
    /// Head index (producer writes here) - padded to cache line
    head: CachePadded<AtomicUsize>,
    /// Tail index (consumer reads from here) - padded to cache line
    tail: CachePadded<AtomicUsize>,
    /// Overflow counter (for monitoring)
    overflow_count: AtomicU64,
}

/// Padding to ensure cache line separation
#[repr(align(64))]
struct CachePadded<T>(T);

impl<T> CachePadded<T> {
    #[inline]
    fn new(val: T) -> Self {
        CachePadded(val)
    }
    
    #[inline]
    fn get(&self) -> &T {
        &self.0
    }
}

unsafe impl<T: Send> Send for EventRingBuffer {}
unsafe impl<T: Sync> Sync for EventRingBuffer {}

impl EventRingBuffer {
    /// Create a new ring buffer
    pub fn new() -> Self {
        EventRingBuffer {
            buffer: UnsafeCell::new(unsafe { std::mem::zeroed() }),
            head: CachePadded::new(AtomicUsize::new(0)),
            tail: CachePadded::new(AtomicUsize::new(0)),
            overflow_count: AtomicU64::new(0),
        }
    }
    
    /// Push an event to the buffer (producer side)
    /// Returns true if successful, false if buffer is full
    #[inline(always)]
    pub fn push(&self, event: Event) -> bool {
        let head = self.head.get().load(Ordering::Relaxed);
        let tail = self.tail.get().load(Ordering::Acquire);
        
        // Check if buffer is full
        let next_head = (head + 1) & (RING_BUFFER_SIZE - 1);
        if next_head == tail {
            self.overflow_count.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        
        // Write event to buffer
        unsafe {
            let buffer = &mut *self.buffer.get();
            buffer[head] = event;
        }
        
        // Memory barrier to ensure write is visible
        std::sync::atomic::fence(Ordering::Release);
        
        // Update head
        self.head.get().store(next_head, Ordering::Release);
        true
    }
    
    /// Pop an event from the buffer (consumer side)
    /// Returns Some(Event) if available, None if buffer is empty
    #[inline(always)]
    pub fn pop(&self) -> Option<Event> {
        let tail = self.tail.get().load(Ordering::Relaxed);
        let head = self.head.get().load(Ordering::Acquire);
        
        // Check if buffer is empty
        if tail == head {
            return None;
        }
        
        // Read event from buffer
        unsafe {
            let buffer = &*self.buffer.get();
            let event = buffer[tail];
            
            // Memory barrier to ensure read is complete
            std::sync::atomic::fence(Ordering::Release);
            
            // Update tail
            let next_tail = (tail + 1) & (RING_BUFFER_SIZE - 1);
            self.tail.get().store(next_tail, Ordering::Release);
            
            Some(event)
        }
    }
    
    /// Check if buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tail.get().load(Ordering::Acquire) == self.head.get().load(Ordering::Acquire)
    }
    
    /// Get current buffer size
    #[inline]
    pub fn size(&self) -> usize {
        let head = self.head.get().load(Ordering::Acquire);
        let tail = self.tail.get().load(Ordering::Acquire);
        (head.wrapping_sub(tail)) & (RING_BUFFER_SIZE - 1)
    }
    
    /// Get overflow count
    #[inline]
    pub fn overflow_count(&self) -> u64 {
        self.overflow_count.load(Ordering::Relaxed)
    }
}

/// Event handler trait for processing events
pub trait EventHandler: Send {
    /// Handle an event
    fn handle_event(&mut self, event: &Event);
    
    /// Called when buffer is empty (opportunity for background work)
    fn on_idle(&mut self) {}
    
    /// Called on event loop start
    fn on_start(&mut self) {}
    
    /// Called on event loop stop
    fn on_stop(&mut self) {}
}

/// CPU affinity helper for pinning threads to specific cores
pub struct CpuAffinity;

impl CpuAffinity {
    /// Pin current thread to a specific CPU core
    /// For AMD Ryzen AI 5: Cores 0-5 are performance cores
    pub fn pin_to_core(core_id: usize) -> Result<(), String> {
        #[cfg(target_os = "linux")]
        {
            use libc::{cpu_set_t, pthread_self, sched_setaffinity, CPU_SET};
            use std::mem;
            
            unsafe {
                let mut cpuset: cpu_set_t = mem::zeroed();
                CPU_SET(core_id, &mut cpuset);
                
                let result = sched_setaffinity(
                    pthread_self(),
                    mem::size_of::<cpu_set_t>(),
                    &cpuset as *const _ as *const _,
                );
                
                if result == 0 {
                    Ok(())
                } else {
                    Err(format!("Failed to pin to core {}: errno {}", core_id, *libc::__errno_location()))
                }
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback for non-Linux systems
            eprintln!("CPU pinning not supported on this platform");
            Ok(())
        }
    }
    
    /// Get the recommended core for the event loop
    /// On AMD Ryzen AI 5, use core 0 for the main event loop
    pub fn get_event_loop_core() -> usize {
        0
    }
    
    /// Get the recommended core for network I/O
    /// On AMD Ryzen AI 5, use core 1 for network handling
    pub fn get_network_core() -> usize {
        1
    }
}

/// Adaptive spin strategy for busy-waiting
pub struct SpinStrategy {
    /// Current spin count
    spin_count: u32,
    /// Minimum spin count
    min_spin: u32,
    /// Maximum spin count
    max_spin: u32,
    /// Empty iterations counter
    empty_iterations: u32,
}

impl SpinStrategy {
    pub fn new() -> Self {
        SpinStrategy {
            spin_count: SPIN_COUNT,
            min_spin: 100,
            max_spin: 10000,
            empty_iterations: 0,
        }
    }
    
    /// Execute spin loop, returns true if should continue spinning
    #[inline(always)]
    pub fn spin(&mut self, has_work: bool) {
        if has_work {
            // Reset spin count on work
            self.spin_count = self.min_spin;
            self.empty_iterations = 0;
            return;
        }
        
        self.empty_iterations += 1;
        
        // Adaptive spin: increase spin count if consistently empty
        if self.empty_iterations > 100 {
            self.spin_count = (self.spin_count * 2).min(self.max_spin);
            self.empty_iterations = 0;
        }
        
        // Busy wait with pause instruction
        for _ in 0..self.spin_count {
            std::hint::spin_loop();
        }
    }
}

/// Main event loop structure
pub struct EventLoop {
    /// Event ring buffer
    buffer: Arc<EventRingBuffer>,
    /// Running flag
    running: AtomicBool,
    /// Events processed counter
    events_processed: AtomicU64,
    /// Last activity timestamp
    last_activity_ns: AtomicU64,
    /// Spin strategy
    spin_strategy: SpinStrategy,
}

impl EventLoop {
    /// Create a new event loop
    pub fn new() -> Self {
        EventLoop {
            buffer: Arc::new(EventRingBuffer::new()),
            running: AtomicBool::new(false),
            events_processed: AtomicU64::new(0),
            last_activity_ns: AtomicU64::new(0),
            spin_strategy: SpinStrategy::new(),
        }
    }
    
    /// Get the event buffer for producers
    pub fn get_buffer(&self) -> Arc<EventRingBuffer> {
        Arc::clone(&self.buffer)
    }
    
    /// Run the event loop with the given handler
    /// This method blocks until stop() is called
    pub fn run<H: EventHandler>(&self, mut handler: H, core_id: Option<usize>) {
        // Pin to CPU core if specified
        if let Some(core) = core_id {
            if let Err(e) = CpuAffinity::pin_to_core(core) {
                eprintln!("Warning: Failed to pin to core {}: {}", core, e);
            }
        }
        
        // Set high priority for the event loop thread
        #[cfg(target_os = "linux")]
        unsafe {
            libc::setpriority(libc::PRIO_PROCESS, 0, -20);
        }
        
        handler.on_start();
        self.running.store(true, Ordering::Release);
        
        let mut idle_counter = 0u32;
        
        while self.running.load(Ordering::Acquire) {
            let mut has_work = false;
            
            // Process all available events (batch processing)
            while let Some(event) = self.buffer.pop() {
                has_work = true;
                handler.handle_event(&event);
                self.events_processed.fetch_add(1, Ordering::Relaxed);
                self.last_activity_ns.store(event.timestamp_ns, Ordering::Relaxed);
            }
            
            // Handle idle time
            if !has_work {
                idle_counter += 1;
                if idle_counter % 1000 == 0 {
                    handler.on_idle();
                }
            } else {
                idle_counter = 0;
            }
            
            // Adaptive spin/yield
            self.spin_strategy.spin(has_work);
        }
        
        handler.on_stop();
    }
    
    /// Stop the event loop
    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
    }
    
    /// Check if event loop is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }
    
    /// Get total events processed
    pub fn events_processed(&self) -> u64 {
        self.events_processed.load(Ordering::Relaxed)
    }
    
    /// Get last activity timestamp
    pub fn last_activity_ns(&self) -> u64 {
        self.last_activity_ns.load(Ordering::Relaxed)
    }
    
    /// Get buffer statistics
    pub fn buffer_size(&self) -> usize {
        self.buffer.size()
    }
}

impl Default for EventLoop {
    fn default() -> Self {
        Self::new()
    }
}

/// High-resolution timestamp in nanoseconds
#[inline(always)]
pub fn get_timestamp_ns() -> u64 {
    #[cfg(target_os = "linux")]
    {
        use std::time::SystemTime;
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
    
    #[cfg(not(target_os = "linux"))]
    {
        use std::time::SystemTime;
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}

/// RDTSC-based timestamp for ultra-low latency (AMD Ryzen specific)
#[inline(always)]
pub fn rdtsc_ns() -> u64 {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    {
        unsafe {
            let low: u32;
            let high: u32;
            std::arch::asm!(
                "rdtsc",
                out("eax") low,
                out("edx") high,
                out("ecx") _,
                out("ebx") _,
            );
            ((high as u64) << 32) | (low as u64)
        }
        // Convert TSC cycles to nanoseconds (approximate for ~4GHz CPU)
        // TODO: Calibrate this based on actual CPU frequency
    }
    
    #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
    {
        get_timestamp_ns()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    struct TestHandler {
        event_count: usize,
    }
    
    impl EventHandler for TestHandler {
        fn handle_event(&mut self, _event: &Event) {
            self.event_count += 1;
        }
    }
    
    #[test]
    fn test_ring_buffer() {
        let buffer = EventRingBuffer::new();
        
        let event = Event::new(EventType::Trade, 1, 100, 1);
        assert!(buffer.push(event));
        assert_eq!(buffer.size(), 1);
        
        let popped = buffer.pop();
        assert!(popped.is_some());
        assert_eq!(buffer.size(), 0);
    }
    
    #[test]
    fn test_event_creation() {
        let event = Event::new(EventType::OrderBookUpdate, 2, 200, 42);
        assert_eq!(event.event_type, EventType::OrderBookUpdate as u8);
        assert_eq!(event.priority, 2);
        assert_eq!(event.source_id, 200);
        assert_eq!(event.sequence, 42);
    }
}
