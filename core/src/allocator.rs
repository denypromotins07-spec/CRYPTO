//! # Custom Lock-Free Bump Allocator and Memory Pool
//! 
//! This module implements a custom memory allocator designed for ultra-low latency
//! trading systems. It prevents GC pauses and OS allocation syscalls by using:
//! - Thread-local bump allocators for fast path allocations
//! - Pre-allocated memory pools for fixed-size objects
//! - Zero-copy semantics where possible
//! - CPU cache-line alignment to prevent false sharing
//!
//! Target: AMD Ryzen AI 5 with 16GB RAM (system capped at 8GB)

use std::alloc::{self, Layout};
use std::cell::RefCell;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Once;

/// Cache line size for AMD Ryzen processors (typically 64 bytes)
const CACHE_LINE_SIZE: usize = 64;

/// Maximum size for pool-allocated objects (larger objects fall back to system allocator)
const MAX_POOL_OBJECT_SIZE: usize = 4096;

/// Pre-allocated arena size per thread (256MB per thread, max 8 threads = 2GB total)
const ARENA_SIZE: usize = 256 * 1024 * 1024;

/// Number of size classes in the memory pool (powers of 2 from 8 to 4096)
const NUM_SIZE_CLASSES: usize = 9;

/// Thread-local bump allocator state
struct BumpAllocator {
    /// Current position in the arena
    current: AtomicUsize,
    /// Base pointer to the arena
    arena: NonNull<u8>,
    /// Arena size
    arena_size: usize,
}

unsafe impl Send for BumpAllocator {}
unsafe impl Sync for BumpAllocator {}

impl BumpAllocator {
    /// Create a new bump allocator with a pre-allocated arena
    fn new() -> Self {
        // Allocate aligned memory for the arena
        let layout = Layout::from_size_align(ARENA_SIZE, CACHE_LINE_SIZE)
            .expect("Invalid layout");
        
        let ptr = unsafe { alloc::alloc(layout) };
        
        if ptr.is_null() {
            panic!("Failed to allocate arena memory");
        }
        
        BumpAllocator {
            current: AtomicUsize::new(0),
            arena: NonNull::new(ptr).unwrap(),
            arena_size: ARENA_SIZE,
        }
    }
    
    /// Allocate memory from the bump allocator (lock-free fast path)
    #[inline(always)]
    fn allocate(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
        // Ensure alignment is a power of 2
        debug_assert!(align.is_power_of_two());
        
        // Calculate aligned size
        let aligned_size = (size + align - 1) & !(align - 1);
        
        // Try to claim space atomically
        let mut current = self.current.load(Ordering::Relaxed);
        
        loop {
            let new_current = current + aligned_size;
            
            // Check if we have enough space
            if new_current > self.arena_size {
                // Arena exhausted, fall back to system allocator
                return None;
            }
            
            // Try to atomically update the current position
            match self.current.compare_exchange_weak(
                current,
                new_current,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Successfully claimed space
                    let ptr = unsafe { self.arena.as_ptr().add(current) };
                    
                    // Ensure proper alignment
                    let aligned_ptr = ((ptr as usize + align - 1) & !(align - 1)) as *mut u8;
                    
                    return NonNull::new(aligned_ptr);
                }
                Err(actual) => current = actual,
            }
        }
    }
    
    /// Reset the allocator (call only when no other threads are allocating)
    #[inline]
    fn reset(&self) {
        self.current.store(0, Ordering::Release);
    }
}

/// Memory pool for fixed-size allocations
struct MemoryPool {
    /// Size class index
    size_class: usize,
    /// Object size for this class
    object_size: usize,
    /// Free list head (atomic for lock-free access)
    free_list: AtomicUsize,
    /// Pool storage (pre-allocated)
    storage: NonNull<u8>,
    /// Number of objects in the pool
    num_objects: usize,
}

unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}

impl MemoryPool {
    /// Create a new memory pool for objects of a specific size
    fn new(size_class: usize, object_size: usize) -> Self {
        let num_objects = ARENA_SIZE / object_size;
        let total_size = num_objects * object_size;
        
        let layout = Layout::from_size_align(total_size, CACHE_LINE_SIZE)
            .expect("Invalid layout");
        
        let ptr = unsafe { alloc::alloc(layout) };
        
        if ptr.is_null() {
            panic!("Failed to allocate pool memory");
        }
        
        // Initialize free list (each slot points to the next)
        for i in 0..num_objects {
            let slot_ptr = unsafe { ptr.add(i * object_size) };
            let next_idx = if i < num_objects - 1 { i + 1 } else { 0 };
            unsafe {
                ptr::write(slot_ptr as *mut usize, next_idx);
            }
        }
        
        MemoryPool {
            size_class,
            object_size,
            free_list: AtomicUsize::new(1), // Start at index 1 (0 means empty)
            storage: NonNull::new(ptr).unwrap(),
            num_objects,
        }
    }
    
    /// Allocate an object from the pool (lock-free)
    #[inline(always)]
    fn allocate(&self) -> Option<NonNull<u8>> {
        loop {
            let head = self.free_list.load(Ordering::Acquire);
            
            if head == 0 {
                // Pool exhausted
                return None;
            }
            
            // Get the next free index from the current head
            let head_idx = head - 1;
            let slot_ptr = unsafe { self.storage.as_ptr().add(head_idx * self.object_size) };
            let next_idx = unsafe { ptr::read(slot_ptr as *const usize) };
            
            // Try to atomically update the free list
            if self.free_list.compare_exchange_weak(
                head,
                next_idx + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ).is_ok() {
                return NonNull::new(slot_ptr);
            }
        }
    }
    
    /// Return an object to the pool (lock-free)
    #[inline(always)]
    fn deallocate(&self, ptr: NonNull<u8>) {
        let ptr_addr = ptr.as_ptr() as usize;
        let storage_addr = self.storage.as_ptr() as usize;
        let idx = (ptr_addr - storage_addr) / self.object_size;
        
        loop {
            let head = self.free_list.load(Ordering::Acquire);
            unsafe {
                ptr::write(ptr.as_ptr() as *mut usize, head);
            }
            
            if self.free_list.compare_exchange_weak(
                head,
                idx + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ).is_ok() {
                return;
            }
        }
    }
}

/// Global allocator state
static mut GLOBAL_ALLOCATOR: Option<&'static CustomAllocator> = None;
static INIT_ONCE: Once = Once::new();

/// Main custom allocator combining bump allocator and memory pools
pub struct CustomAllocator {
    /// Thread-local bump allocator
    bump: BumpAllocator,
    /// Memory pools for different size classes
    pools: [MemoryPool; NUM_SIZE_CLASSES],
}

impl CustomAllocator {
    /// Initialize the global custom allocator
    pub fn init() -> &'static Self {
        unsafe {
            INIT_ONCE.call_once(|| {
                // Calculate size classes (powers of 2 from 8 to 4096)
                let mut pools = Vec::with_capacity(NUM_SIZE_CLASSES);
                for i in 0..NUM_SIZE_CLASSES {
                    let size = 8 << i; // 8, 16, 32, 64, 128, 256, 512, 1024, 2048
                    pools.push(MemoryPool::new(i, size));
                }
                
                let pools_array: [MemoryPool; NUM_SIZE_CLASSES] = pools.try_into()
                    .expect("Failed to create pools array");
                
                GLOBAL_ALLOCATOR = Some(Box::leak(Box::new(CustomAllocator {
                    bump: BumpAllocator::new(),
                    pools: pools_array,
                })));
            });
            
            GLOBAL_ALLOCATOR.unwrap()
        }
    }
    
    /// Get the size class for a given allocation size
    #[inline(always)]
    fn get_size_class(&self, size: usize) -> Option<usize> {
        if size == 0 || size > MAX_POOL_OBJECT_SIZE {
            return None;
        }
        
        // Find the smallest power of 2 that fits the size
        let class = (size - 1).leading_zeros() as usize;
        let class = 31 - class; // Position of highest set bit
        
        // Adjust for our size class range (8 to 4096)
        if class < 3 {
            Some(0) // Round up to minimum size class (8 bytes)
        } else if class > 11 {
            None // Too large for pool
        } else {
            Some(class - 3) // Offset to start at 0
        }
    }
    
    /// Allocate memory using the custom allocator
    #[inline(always)]
    pub fn allocate(&self, size: usize, align: usize) -> *mut u8 {
        // Try pool allocation first for small objects
        if let Some(class) = self.get_size_class(size) {
            if let Some(ptr) = self.pools[class].allocate() {
                return ptr.as_ptr();
            }
        }
        
        // Fall back to bump allocator
        if let Some(ptr) = self.bump.allocate(size, align) {
            return ptr.as_ptr();
        }
        
        // Ultimate fallback to system allocator
        let layout = Layout::from_size_align(size, align)
            .expect("Invalid layout");
        unsafe { alloc::alloc(layout) }
    }
    
    /// Deallocate memory
    #[inline(always)]
    pub fn deallocate(&self, ptr: *mut u8, size: usize, align: usize) {
        // Try to return to pool if it's a pooled allocation
        if let Some(class) = self.get_size_class(size) {
            let pool_start = self.pools[class].storage.as_ptr() as usize;
            let pool_end = pool_start + (self.pools[class].num_objects * self.pools[class].object_size);
            let ptr_addr = ptr as usize;
            
            if ptr_addr >= pool_start && ptr_addr < pool_end {
                if let Some(non_null) = NonNull::new(ptr) {
                    self.pools[class].deallocate(non_null);
                    return;
                }
            }
        }
        
        // Check if it's from the bump allocator (can't individually free, must reset)
        let bump_start = self.bump.arena.as_ptr() as usize;
        let bump_end = bump_start + self.bump.arena_size;
        let ptr_addr = ptr as usize;
        
        if ptr_addr >= bump_start && ptr_addr < bump_end {
            // Bump allocator doesn't support individual deallocation
            // Memory will be reclaimed on reset
            return;
        }
        
        // Fall back to system allocator
        let layout = Layout::from_size_align(size, align)
            .expect("Invalid layout");
        unsafe { alloc::dealloc(ptr, layout) }
    }
    
    /// Reset the bump allocator (call during quiescent periods)
    #[inline]
    pub fn reset_bump(&self) {
        self.bump.reset();
    }
}

/// Global allocator implementation for std::alloc
pub struct GlobalCustomAllocator;

unsafe impl std::alloc::GlobalAlloc for GlobalCustomAllocator {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if let Some(allocator) = GLOBAL_ALLOCATOR {
            allocator.allocate(layout.size(), layout.align())
        } else {
            alloc::alloc(layout)
        }
    }
    
    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if let Some(allocator) = GLOBAL_ALLOCATOR {
            allocator.deallocate(ptr, layout.size(), layout.align());
        } else {
            alloc::dealloc(ptr, layout);
        }
    }
}

/// Thread-local allocator wrapper for zero-contention allocations
#[derive(Default)]
pub struct ThreadLocalAllocator {
    local_buffer: RefCell<Vec<u8>>,
}

impl ThreadLocalAllocator {
    pub const fn new() -> Self {
        ThreadLocalAllocator {
            local_buffer: RefCell::new(Vec::new()),
        }
    }
    
    /// Allocate from thread-local buffer (no synchronization needed)
    #[inline]
    pub fn allocate_local(&self, size: usize) -> *mut u8 {
        let mut buffer = self.local_buffer.borrow_mut();
        let current_len = buffer.len();
        buffer.resize(current_len + size, 0);
        unsafe { buffer.as_mut_ptr().add(current_len) }
    }
}

/// Zero-copy buffer for network data
pub struct ZeroCopyBuffer {
    data: NonNull<u8>,
    capacity: usize,
    len: usize,
}

unsafe impl Send for ZeroCopyBuffer {}
unsafe impl Sync for ZeroCopyBuffer {}

impl ZeroCopyBuffer {
    /// Create a new zero-copy buffer
    pub fn new(capacity: usize) -> Self {
        let allocator = CustomAllocator::init();
        let ptr = allocator.allocate(capacity, CACHE_LINE_SIZE);
        
        ZeroCopyBuffer {
            data: NonNull::new(ptr).unwrap(),
            capacity,
            len: 0,
        }
    }
    
    /// Get a slice of the buffer data
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr(), self.len) }
    }
    
    /// Get a mutable slice of the buffer data
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_ptr(), self.len) }
    }
    
    /// Append data to the buffer (zero-copy if possible)
    #[inline]
    pub fn append(&mut self, data: &[u8]) {
        let new_len = self.len + data.len();
        if new_len > self.capacity {
            panic!("ZeroCopyBuffer overflow");
        }
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), self.data.as_ptr().add(self.len), data.len());
        }
        self.len = new_len;
    }
    
    /// Clear the buffer (doesn't deallocate, just resets length)
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }
}

impl Drop for ZeroCopyBuffer {
    fn drop(&mut self) {
        let allocator = CustomAllocator::init();
        allocator.deallocate(self.data.as_ptr(), self.capacity, CACHE_LINE_SIZE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bump_allocator() {
        let allocator = BumpAllocator::new();
        let ptr1 = allocator.allocate(100, 8);
        let ptr2 = allocator.allocate(200, 8);
        
        assert!(ptr1.is_some());
        assert!(ptr2.is_some());
        assert!(ptr2.unwrap().as_ptr() > ptr1.unwrap().as_ptr());
    }
    
    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new(0, 64);
        let ptr1 = pool.allocate();
        let ptr2 = pool.allocate();
        
        assert!(ptr1.is_some());
        assert!(ptr2.is_some());
        
        if let Some(p) = ptr1 {
            pool.deallocate(p);
        }
        
        // Should be able to allocate again after deallocation
        let ptr3 = pool.allocate();
        assert!(ptr3.is_some());
    }
}
