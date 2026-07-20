//! # Ultra-Low Latency Crypto Trading Bot - Rust Core Engine
//! 
//! This is the main entry point for the Rust execution engine, designed for
//! microsecond-level trading on AMD Ryzen AI 5 processors.
//! 
//! Features:
//! - Custom memory allocator to prevent GC pauses and reduce syscalls
//! - CPU core pinning for optimal cache utilization
//! - Lock-free event loop for maximum throughput
//! - Integration with Python via IPC bridge
//! 
//! Memory Budget: Strictly capped at 8GB total system usage
//! Target Latency: < 10 microseconds for order execution

#![no_std]
#![cfg_attr(not(test), no_main)]

// Conditional compilation for std support (needed for some dependencies)
#[cfg(not(target_os = "none"))]
extern crate std;

#[cfg(not(target_os = "none"))]
extern crate alloc;

// Module declarations
pub mod allocator;
pub mod event_loop;
pub mod network;

use core::panic::PanicInfo;

#[cfg(not(target_os = "none"))]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(not(target_os = "none"))]
use std::sync::Arc;
#[cfg(not(target_os = "none"))]
use std::thread;
#[cfg(not(target_os = "none"))]
use std::time::Duration;

// Re-export key types for external use
pub use allocator::{CustomAllocator, GlobalCustomAllocator, ZeroCopyBuffer};
pub use event_loop::{Event, EventLoop, EventHandler, EventType, get_timestamp_ns};

/// System-wide configuration constants
pub mod config {
    /// Maximum RAM usage in bytes (8GB cap)
    pub const MAX_RAM_BYTES: usize = 8 * 1024 * 1024 * 1024;
    
    /// Number of CPU cores available on AMD Ryzen AI 5
    pub const NUM_CPU_CORES: usize = 6;
    
    /// Core reserved for OS and background tasks
    pub const OS_RESERVED_CORE: usize = 5;
    
    /// Core for main event loop
    pub const EVENT_LOOP_CORE: usize = 0;
    
    /// Core for network I/O
    pub const NETWORK_CORE: usize = 1;
    
    /// Core for IPC bridge
    pub const IPC_CORE: usize = 2;
    
    /// Watchdog timeout in seconds
    pub const WATCHDOG_TIMEOUT_SECS: u64 = 30;
    
    /// Heartbeat interval in milliseconds
    pub const HEARTBEAT_INTERVAL_MS: u64 = 100;
}

/// Global shutdown flag
static SHUTDOWN_FLAG: AtomicBool = AtomicBool::new(false);

/// System uptime counter (nanoseconds)
static UPTIME_NS: AtomicU64 = AtomicU64::new(0);

/// Events processed counter
static EVENTS_PROCESSED: AtomicU64 = AtomicU64::new(0);

/// Main entry point for the Rust engine
#[cfg(not(target_os = "none"))]
fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   QUANTUM TRADING ENGINE - Rust Core v1.0.0             ║");
    println!("║   Ultra-Low Latency Execution for Binance               ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();
    
    // Initialize the custom allocator
    println!("[INIT] Initializing custom memory allocator...");
    let allocator = CustomAllocator::init();
    println!("[INIT] Allocator initialized. Arena size: {} MB", 
             allocator::ARENA_SIZE / (1024 * 1024));
    
    // Log system configuration
    println!();
    println!("[CONFIG] System Configuration:");
    println!("  - Max RAM: {} GB", config::MAX_RAM_BYTES / (1024 * 1024 * 1024));
    println!("  - CPU Cores: {}", config::NUM_CPU_CORES);
    println!("  - Event Loop Core: {}", config::EVENT_LOOP_CORE);
    println!("  - Network Core: {}", config::NETWORK_CORE);
    println!("  - IPC Core: {}", config::IPC_CORE);
    println!();
    
    // Create the main event loop
    println!("[INIT] Creating event loop...");
    let event_loop = Arc::new(EventLoop::new());
    let buffer = event_loop.get_buffer();
    println!("[INIT] Event loop created. Buffer capacity: {} events", 
             1 << 16); // RING_BUFFER_SIZE
    println!();
    
    // Spawn the network listener thread
    println!("[INIT] Spawning network listener on core {}...", config::NETWORK_CORE);
    let network_buffer = Arc::clone(&buffer);
    let network_handle = thread::spawn(move || {
        // Pin to network core
        if let Err(e) = event_loop::CpuAffinity::pin_to_core(config::NETWORK_CORE) {
            eprintln!("[WARN] Failed to pin network thread: {}", e);
        }
        
        println!("[NETWORK] Network listener started");
        
        // Simulate receiving market data events
        let mut sequence = 0u64;
        while !SHUTDOWN_FLAG.load(Ordering::Relaxed) {
            // Create a mock trade event
            let mut event = Event::new(EventType::Trade, 1, 1001, sequence);
            event.set_payload(b"BTCUSDT:BUY:0.001");
            
            if network_buffer.push(event) {
                sequence += 1;
            }
            
            // Small delay to simulate real market data rate
            thread::sleep(Duration::from_micros(100));
        }
        
        println!("[NETWORK] Network listener stopped. Total events: {}", sequence);
    });
    
    // Spawn the IPC bridge thread
    println!("[INIT] Spawning IPC bridge on core {}...", config::IPC_CORE);
    let ipc_buffer = Arc::clone(&buffer);
    let ipc_handle = thread::spawn(move || {
        // Pin to IPC core
        if let Err(e) = event_loop::CpuAffinity::pin_to_core(config::IPC_CORE) {
            eprintln!("[WARN] Failed to pin IPC thread: {}", e);
        }
        
        println!("[IPC] IPC bridge started");
        
        // Forward events to Python (simulated)
        let mut forwarded = 0u64;
        while !SHUTDOWN_FLAG.load(Ordering::Relaxed) {
            if let Some(_event) = ipc_buffer.pop() {
                forwarded += 1;
                // In production, this would write to shared memory
                // for Python to consume
            }
            thread::yield_now();
        }
        
        println!("[IPC] IPC bridge stopped. Events forwarded: {}", forwarded);
    });
    
    // Run the main event loop on the designated core
    println!("[INIT] Starting event loop on core {}...", config::EVENT_LOOP_CORE);
    
    struct MainEventHandler {
        events_processed: u64,
        last_heartbeat: u64,
    }
    
    impl event_loop::EventHandler for MainEventHandler {
        fn handle_event(&mut self, event: &Event) {
            self.events_processed += 1;
            EVENTS_PROCESSED.store(self.events_processed, Ordering::Relaxed);
            
            // Process based on event type
            match EventType::from_u8(event.event_type) {
                EventType::Trade => {
                    // Handle trade event
                    let payload = event.get_payload();
                    // In production: update order book, trigger strategies
                }
                EventType::OrderBookUpdate => {
                    // Handle order book update
                }
                EventType::Heartbeat => {
                    self.last_heartbeat = event.timestamp_ns;
                }
                _ => {}
            }
        }
        
        fn on_idle(&mut self) {
            // Background work during idle periods
            // E.g., flush logs, update statistics
        }
        
        fn on_start(&mut self) {
            println!("[EVENT_LOOP] Event loop handler started");
        }
        
        fn on_stop(&mut self) {
            println!("[EVENT_LOOP] Event loop handler stopped. Processed: {}", self.events_processed);
        }
    }
    
    // Start the event loop (this blocks until stop() is called)
    let handler = MainEventHandler {
        events_processed: 0,
        last_heartbeat: 0,
    };
    
    // Run event loop in a separate thread so we can handle signals
    let event_loop_clone = Arc::clone(&event_loop);
    let main_handle = thread::spawn(move || {
        event_loop_clone.run(handler, Some(config::EVENT_LOOP_CORE));
    });
    
    // Send heartbeat events periodically
    println!("[INIT] System ready. Sending heartbeats...");
    println!();
    
    let start_time = get_timestamp_ns();
    UPTIME_NS.store(start_time, Ordering::Release);
    
    // Main loop for monitoring and signal handling
    let mut iteration = 0u64;
    while !SHUTDOWN_FLAG.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_millis(config::HEARTBEAT_INTERVAL_MS));
        
        iteration += 1;
        
        // Send heartbeat event
        let heartbeat = Event::new(EventType::Heartbeat, 0, 0, iteration);
        if buffer.push(heartbeat) {
            // Heartbeat sent
        }
        
        // Print status every 10 seconds
        if iteration % 100 == 0 {
            let uptime_ns = UPTIME_NS.load(Ordering::Relaxed);
            let elapsed_secs = (get_timestamp_ns() - uptime_ns) / 1_000_000_000;
            let events = EVENTS_PROCESSED.load(Ordering::Relaxed);
            let events_per_sec = if elapsed_secs > 0 { events / elapsed_secs } else { 0 };
            
            println!("[STATUS] Uptime: {}s | Events: {} | Rate: {} events/sec | Buffer: {}",
                     elapsed_secs, events, events_per_sec, buffer.size());
        }
        
        // Check for shutdown condition (for demo, shutdown after 30 seconds)
        if iteration >= 300 {
            println!();
            println!("[SHUTDOWN] Demo complete. Initiating graceful shutdown...");
            SHUTDOWN_FLAG.store(true, Ordering::Release);
        }
    }
    
    // Stop the event loop
    event_loop.stop();
    
    // Wait for threads to finish
    println!("[SHUTDOWN] Waiting for threads to terminate...");
    let _ = network_handle.join();
    let _ = ipc_handle.join();
    let _ = main_handle.join();
    
    // Final statistics
    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║                    SHUTDOWN COMPLETE                     ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    let final_events = EVENTS_PROCESSED.load(Ordering::Relaxed);
    println!("║  Total Events Processed: {:>28} ║", final_events);
    let uptime_ns = UPTIME_NS.load(Ordering::Relaxed);
    let elapsed_secs = (get_timestamp_ns() - uptime_ns) / 1_000_000_000;
    println!("║  Total Runtime: {:>35} ║", format!("{} seconds", elapsed_secs));
    if elapsed_secs > 0 {
        println!("║  Avg Throughput: {:>34} ║", format!("{} events/sec", final_events / elapsed_secs));
    }
    println!("╚══════════════════════════════════════════════════════════╝");
}

/// Panic handler for production builds
#[cfg(not(target_os = "none"))]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    eprintln!();
    eprintln!("╔══════════════════════════════════════════════════════════╗");
    eprintln!("║                    PANIC DETECTED                        ║");
    eprintln!("╚══════════════════════════════════════════════════════════╝");
    eprintln!("Location: {}", info);
    
    // Set shutdown flag to trigger graceful termination
    SHUTDOWN_FLAG.store(true, Ordering::Release);
    
    // Abort the process
    #[cfg(not(target_os = "none"))]
    std::process::abort();
    
    #[cfg(target_os = "none")]
    loop {}
}

/// Get current uptime in nanoseconds
#[inline]
pub fn get_uptime_ns() -> u64 {
    UPTIME_NS.load(Ordering::Relaxed)
}

/// Get total events processed
#[inline]
pub fn get_events_processed() -> u64 {
    EVENTS_PROCESSED.load(Ordering::Relaxed)
}

/// Request system shutdown
#[inline]
pub fn request_shutdown() {
    SHUTDOWN_FLAG.store(true, Ordering::Release);
}

/// Check if shutdown is requested
#[inline]
pub fn is_shutdown_requested() -> bool {
    SHUTDOWN_FLAG.load(Ordering::Relaxed)
}
