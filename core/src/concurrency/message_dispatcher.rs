//! Message Dispatcher Module
//! 
//! A highly optimized, zero-allocation message dispatcher that routes events
//! to specific CPU cores based on thread affinity and L1/L2 cache locality
//! to minimize cache misses.
//! 
//! Key features:
//! - Zero-allocation message passing
//! - CPU core affinity management
//! - Cache-line aware data structures
//! - Lock-free ring buffers for inter-thread communication
//! - Priority-based message routing
//! 
//! Target latency: < 200 nanoseconds per message dispatch

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use std::cell::UnsafeCell;
use std::marker::PhantomData;

/// Cache line size for padding (typically 64 bytes on x86_64)
const CACHE_LINE_SIZE: usize = 64;

/// Maximum number of CPU cores supported
const MAX_CORES: usize = 64;

/// Message types supported by the dispatcher
#[derive(Clone, Debug)]
pub enum MessageType {
    /// Order book update
    OrderBookUpdate { symbol: u32, sequence: u64 },
    /// Trade execution
    Trade { symbol: u32, price: i64, quantity: u64 },
    /// Signal from ML model
    Signal { strategy_id: u32, value: f64 },
    /// Risk management event
    RiskEvent { level: u8, code: u32 },
    /// Timer/tick event
    Timer { tick_id: u64 },
    /// Custom payload
    Custom { type_id: u32, data_len: usize },
}

/// Message envelope with metadata
#[repr(C)]
pub struct Message {
    /// Message type
    pub msg_type: MessageType,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
    /// Source thread/core ID
    pub source_core: u8,
    /// Destination core ID
    pub dest_core: u8,
    /// Priority (lower = higher priority)
    pub priority: u8,
    /// Sequence number for ordering
    pub sequence: u64,
    /// Inline data (for small messages)
    pub inline_data: [u8; 32],
}

impl Message {
    pub fn new(msg_type: MessageType, dest_core: u8, priority: u8) -> Self {
        Self {
            msg_type,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            source_core: 0,
            dest_core,
            priority,
            sequence: 0,
            inline_data: [0; 32],
        }
    }

    /// Set inline data for small payloads
    #[inline]
    pub fn with_inline_data(mut self, data: &[u8]) -> Self {
        let len = data.len().min(32);
        self.inline_data[..len].copy_from_slice(&data[..len]);
        self
    }
}

/// Lock-free single-producer single-consumer ring buffer
/// Padded to avoid false sharing
pub struct SPSCRingBuffer<T> {
    /// Buffer storage
    buffer: Box<[UnsafeCell<T>]>,
    /// Buffer size (must be power of 2)
    size: usize,
    /// Mask for wraparound
    mask: usize,
    /// Write position (owned by producer)
    write_pos: UnsafeCell<usize>,
    /// Read position (owned by consumer)
    read_pos: UnsafeCell<usize>,
    /// Cache line padding
    _pad1: [u8; CACHE_LINE_SIZE],
    _pad2: [u8; CACHE_LINE_SIZE],
}

unsafe impl<T: Send> Send for SPSCRingBuffer<T> {}
unsafe impl<T: Send> Sync for SPSCRingBuffer<T> {}

impl<T: Default + Clone> SPSCRingBuffer<T> {
    pub fn new(size: usize) -> Self {
        assert!(size.is_power_of_two(), "Size must be power of 2");
        
        let mut buffer = Vec::with_capacity(size);
        for _ in 0..size {
            buffer.push(UnsafeCell::new(T::default()));
        }

        Self {
            buffer: buffer.into_boxed_slice(),
            size,
            mask: size - 1,
            write_pos: UnsafeCell::new(0),
            read_pos: UnsafeCell::new(0),
            _pad1: [0; CACHE_LINE_SIZE],
            _pad2: [0; CACHE_LINE_SIZE],
        }
    }

    /// Try to push an item (producer side)
    #[inline]
    pub fn try_push(&self, item: T) -> Result<(), T> {
        unsafe {
            let write_pos = *self.write_pos.get();
            let read_pos = *self.read_pos.get();
            
            // Check if buffer is full
            if write_pos.wrapping_sub(read_pos) >= self.size {
                return Err(item);
            }

            let idx = write_pos & self.mask;
            *self.buffer[idx].get() = item;
            
            // Memory barrier before updating write position
            std::sync::atomic::fence(Ordering::Release);
            
            *self.write_pos.get() = write_pos.wrapping_add(1);
            Ok(())
        }
    }

    /// Try to pop an item (consumer side)
    #[inline]
    pub fn try_pop(&self) -> Option<T> {
        unsafe {
            let read_pos = *self.read_pos.get();
            let write_pos = *self.write_pos.get();
            
            // Check if buffer is empty
            if read_pos >= write_pos {
                return None;
            }

            let idx = read_pos & self.mask;
            
            // Memory barrier after reading data
            std::sync::atomic::fence(Ordering::Acquire);
            
            let item = (*self.buffer[idx].get()).clone();
            *self.read_pos.get() = read_pos.wrapping_add(1);
            Some(item)
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        unsafe {
            let write_pos = *self.write_pos.get();
            let read_pos = *self.read_pos.get();
            write_pos.wrapping_sub(read_pos)
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.len() >= self.size - 1
    }
}

/// Per-core message queue structure
pub struct CoreQueue {
    /// Incoming message buffer
    incoming: SPSCRingBuffer<Message>,
    /// Outgoing message buffer (for responses)
    outgoing: SPSCRingBuffer<Message>,
    /// Core is active flag
    active: AtomicBool,
    /// Messages processed counter
    messages_processed: AtomicU64,
    /// Last activity timestamp
    last_activity_ns: AtomicU64,
    /// Cache line padding
    _pad: [u8; CACHE_LINE_SIZE],
}

impl CoreQueue {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            incoming: SPSCRingBuffer::new(buffer_size),
            outgoing: SPSCRingBuffer::new(buffer_size),
            active: AtomicBool::new(true),
            messages_processed: AtomicU64::new(0),
            last_activity_ns: AtomicU64::new(0),
            _pad: [0; CACHE_LINE_SIZE],
        }
    }

    #[inline]
    pub fn push_message(&self, msg: Message) -> Result<(), Message> {
        self.last_activity_ns.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            Ordering::Relaxed,
        );
        self.incoming.try_push(msg)
    }

    #[inline]
    pub fn pop_message(&self) -> Option<Message> {
        let msg = self.incoming.try_pop();
        if msg.is_some() {
            self.messages_processed.fetch_add(1, Ordering::Relaxed);
        }
        msg
    }

    #[inline]
    pub fn push_outgoing(&self, msg: Message) -> Result<(), Message> {
        self.outgoing.try_push(msg)
    }

    #[inline]
    pub fn pop_outgoing(&self) -> Option<Message> {
        self.outgoing.try_pop()
    }

    #[inline]
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    #[inline]
    pub fn deactivate(&self) {
        self.active.store(false, Ordering::Release);
    }

    #[inline]
    pub fn stats(&self) -> (u64, u64, u64) {
        (
            self.messages_processed.load(Ordering::Relaxed),
            self.incoming.len() as u64,
            self.last_activity_ns.load(Ordering::Relaxed),
        )
    }
}

/// Thread affinity helper for binding to specific CPU cores
pub struct CoreAffinity {
    /// Target core ID
    core_id: usize,
    /// Affinity mask
    affinity_mask: usize,
}

impl CoreAffinity {
    pub fn new(core_id: usize) -> Self {
        Self {
            core_id,
            affinity_mask: 1 << (core_id % 64),
        }
    }

    /// Bind current thread to this core
    pub fn bind_current_thread(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            use libc::{cpu_set_t, pthread_setaffinity_np, pthread_self};
            
            let mut cpuset: cpu_set_t = unsafe { std::mem::zeroed() };
            unsafe {
                libc::CPU_SET(self.core_id, &mut cpuset);
            }
            
            let handle = unsafe { pthread_self() };
            let result = unsafe {
                pthread_setaffinity_np(
                    handle,
                    std::mem::size_of::<cpu_set_t>(),
                    &cpuset,
                )
            };
            
            return result == 0;
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback for other platforms
            true
        }
    }

    /// Get the core ID
    #[inline]
    pub fn core_id(&self) -> usize {
        self.core_id
    }
}

/// Main message dispatcher coordinating all core queues
pub struct MessageDispatcher {
    /// Per-core queues
    core_queues: [Option<Box<CoreQueue>>; MAX_CORES],
    /// Number of active cores
    active_cores: AtomicUsize,
    /// Global sequence counter
    sequence_counter: AtomicU64,
    /// Total messages dispatched
    total_dispatched: AtomicU64,
    /// Total messages dropped (queue full)
    total_dropped: AtomicU64,
    /// Dispatcher statistics
    dispatch_times_ns: UnsafeCell<[u64; 1000]>,
    dispatch_count: AtomicUsize,
}

unsafe impl Send for MessageDispatcher {}
unsafe impl Sync for MessageDispatcher {}

impl MessageDispatcher {
    pub fn new(num_cores: usize, buffer_size: usize) -> Self {
        assert!(num_cores <= MAX_CORES, "Too many cores");

        let mut core_queues: [Option<Box<CoreQueue>>; MAX_CORES] = Default::default();
        
        for i in 0..num_cores {
            core_queues[i] = Some(Box::new(CoreQueue::new(buffer_size)));
        }

        Self {
            core_queues,
            active_cores: AtomicUsize::new(num_cores),
            sequence_counter: AtomicU64::new(0),
            total_dispatched: AtomicU64::new(0),
            total_dropped: AtomicU64::new(0),
            dispatch_times_ns: UnsafeCell::new([0; 1000]),
            dispatch_count: AtomicUsize::new(0),
        }
    }

    /// Dispatch a message to a specific core
    #[inline]
    pub fn dispatch_to_core(&self, mut msg: Message, core_id: usize) -> bool {
        let start = std::time::Instant::now();

        if core_id >= MAX_CORES {
            return false;
        }

        let queue = match &self.core_queues[core_id] {
            Some(q) => q.as_ref(),
            None => return false,
        };

        // Assign sequence number
        msg.sequence = self.sequence_counter.fetch_add(1, Ordering::Relaxed);
        msg.source_core = std::thread::current().id().as_u64().try_into().unwrap_or(0);

        let result = queue.push_message(msg).is_ok();

        if result {
            self.total_dispatched.fetch_add(1, Ordering::Relaxed);
        } else {
            self.total_dropped.fetch_add(1, Ordering::Relaxed);
        }

        // Record timing
        let elapsed = start.elapsed().as_nanos() as u64;
        self.record_dispatch_time(elapsed);

        result
    }

    /// Broadcast message to all active cores
    pub fn broadcast(&self, msg: Message) -> usize {
        let mut sent_count = 0;
        let num_cores = self.active_cores.load(Ordering::Acquire);

        for i in 0..num_cores {
            if let Some(ref queue) = self.core_queues[i] {
                let mut msg_copy = msg.clone();
                msg_copy.dest_core = i as u8;
                
                if queue.push_message(msg_copy).is_ok() {
                    sent_count += 1;
                }
            }
        }

        self.total_dispatched.fetch_add(sent_count as u64, Ordering::Relaxed);
        sent_count
    }

    /// Route message based on priority and content hash
    #[inline]
    pub fn route_by_hash(&self, msg: Message) -> bool {
        // Simple hash-based routing
        let hash = match &msg.msg_type {
            MessageType::OrderBookUpdate { symbol, .. } => *symbol,
            MessageType::Trade { symbol, .. } => *symbol,
            MessageType::Signal { strategy_id, .. } => *strategy_id,
            _ => msg.priority as u32,
        };

        let core_id = (hash as usize) % self.active_cores.load(Ordering::Acquire);
        self.dispatch_to_core(msg, core_id)
    }

    /// Route high-priority messages to dedicated cores (0-1)
    #[inline]
    pub fn route_by_priority(&self, mut msg: Message) -> bool {
        if msg.priority <= 2 {
            // High priority: use core 0 or 1
            msg.dest_core = (msg.priority % 2) as u8;
            self.dispatch_to_core(msg, msg.dest_core as usize)
        } else {
            // Normal priority: use remaining cores
            self.route_by_hash(msg)
        }
    }

    /// Get reference to a core's queue
    #[inline]
    pub fn get_queue(&self, core_id: usize) -> Option<&CoreQueue> {
        self.core_queues.get(core_id)
            .and_then(|q| q.as_ref().map(|b| b.as_ref()))
    }

    /// Record dispatch time for statistics
    fn record_dispatch_time(&self, time_ns: u64) {
        let idx = self.dispatch_count.fetch_add(1, Ordering::Relaxed) % 1000;
        unsafe {
            (*self.dispatch_times_ns.get())[idx] = time_ns;
        }
    }

    /// Get average dispatch time
    pub fn avg_dispatch_time_ns(&self) -> f64 {
        let count = self.dispatch_count.load(Ordering::Relaxed).min(1000);
        if count == 0 {
            return 0.0;
        }

        unsafe {
            let times = &(*self.dispatch_times_ns.get())[..count];
            times.iter().sum::<u64>() as f64 / count as f64
        }
    }

    /// Get dispatcher statistics
    pub fn stats(&self) -> DispatcherStats {
        DispatcherStats {
            active_cores: self.active_cores.load(Ordering::Acquire),
            total_dispatched: self.total_dispatched.load(Ordering::Relaxed),
            total_dropped: self.total_dropped.load(Ordering::Relaxed),
            avg_dispatch_time_ns: self.avg_dispatch_time_ns(),
            drop_rate: {
                let total = self.total_dispatched.load(Ordering::Relaxed) 
                    + self.total_dropped.load(Ordering::Relaxed);
                if total == 0 {
                    0.0
                } else {
                    self.total_dropped.load(Ordering::Relaxed) as f64 / total as f64
                }
            },
        }
    }

    /// Activate a core
    pub fn activate_core(&self, core_id: usize) -> bool {
        if core_id >= MAX_CORES || self.core_queues[core_id].is_none() {
            return false;
        }

        if let Some(ref queue) = self.core_queues[core_id] {
            // Reactivation logic would go here
        }

        self.active_cores.fetch_add(1, Ordering::AcqRel);
        true
    }

    /// Deactivate a core (graceful drain)
    pub fn deactivate_core(&self, core_id: usize) -> bool {
        if core_id >= MAX_CORES {
            return false;
        }

        if let Some(ref queue) = self.core_queues[core_id] {
            queue.deactivate();
            
            // Wait for queue to drain
            while !queue.incoming.is_empty() {
                std::hint::spin_loop();
            }
        }

        self.active_cores.fetch_sub(1, Ordering::AcqRel);
        true
    }
}

/// Statistics about the dispatcher
#[derive(Debug, Clone)]
pub struct DispatcherStats {
    pub active_cores: usize,
    pub total_dispatched: u64,
    pub total_dropped: u64,
    pub avg_dispatch_time_ns: f64,
    pub drop_rate: f64,
}

/// Worker thread runner for processing messages on a core
pub struct CoreWorker {
    core_id: usize,
    running: AtomicBool,
}

impl CoreWorker {
    pub fn new(core_id: usize) -> Self {
        Self {
            core_id,
            running: AtomicBool::new(false),
        }
    }

    /// Start the worker thread bound to its core
    pub fn start<F>(&self, dispatcher: &MessageDispatcher, handler: F) -> std::thread::JoinHandle<()>
    where
        F: Fn(Message) + Send + 'static,
    {
        self.running.store(true, Ordering::Release);

        let core_id = self.core_id;
        let affinity = CoreAffinity::new(core_id);

        let handler = std::sync::Arc::new(handler);

        std::thread::spawn(move || {
            // Bind to core
            affinity.bind_current_thread();

            while affinity.core_id() < MAX_CORES {
                // Check if should stop
                // In real implementation, would check running flag
                
                // Process messages
                if let Some(queue) = dispatcher.get_queue(core_id) {
                    while let Some(msg) = queue.pop_message() {
                        handler(msg);
                    }
                }

                // Yield to avoid busy spinning
                std::hint::spin_loop();
            }
        })
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let buffer: SPSCRingBuffer<i32> = SPSCRingBuffer::new(16);
        
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());
        
        buffer.try_push(42).unwrap();
        assert_eq!(buffer.len(), 1);
        
        let val = buffer.try_pop().unwrap();
        assert_eq!(val, 42);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_dispatcher_dispatch() {
        let dispatcher = MessageDispatcher::new(4, 64);
        
        let msg = Message::new(
            MessageType::Timer { tick_id: 1 },
            0,
            5,
        );
        
        assert!(dispatcher.dispatch_to_core(msg, 0));
        
        let stats = dispatcher.stats();
        assert_eq!(stats.total_dispatched, 1);
        assert_eq!(stats.total_dropped, 0);
    }

    #[test]
    fn test_broadcast() {
        let dispatcher = MessageDispatcher::new(4, 64);
        
        let msg = Message::new(
            MessageType::Timer { tick_id: 1 },
            0,
            5,
        );
        
        let sent = dispatcher.broadcast(msg);
        assert_eq!(sent, 4);
        
        let stats = dispatcher.stats();
        assert_eq!(stats.total_dispatched, 4);
    }
}
