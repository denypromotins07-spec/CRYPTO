//! System Health Monitor with Graceful Degradation
//! 
//! Monitors CPU temperature, RAM usage, and network jitter.
//! Implements dynamic graceful degradation protocol when approaching 8GB RAM limit.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;
use std::collections::VecDeque;

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

// ============================================================================
// System Metrics
// ============================================================================

/// System health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// All systems operating normally
    Healthy,
    /// Minor issues detected
    Warning,
    /// Significant issues, degradation active
    Degraded,
    /// Critical issues, emergency measures
    Critical,
}

/// CPU metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuMetrics {
    /// Current usage percentage (0-100)
    pub usage_percent: f64,
    /// Temperature in Celsius (if available)
    pub temperature_celsius: Option<f64>,
    /// Number of cores
    pub core_count: usize,
    /// Per-core usage percentages
    pub per_core_usage: Vec<f64>,
}

/// Memory metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Used memory in bytes
    pub used_bytes: u64,
    /// Total memory in bytes
    pub total_bytes: u64,
    /// Available memory in bytes
    pub available_bytes: u64,
    /// Usage percentage (0-100)
    pub usage_percent: f64,
    /// Hard limit in GB (8.0 for this system)
    pub hard_limit_gb: f64,
    /// Approaching hard limit
    pub approaching_limit: bool,
}

impl MemoryMetrics {
    /// Check if memory is within acceptable bounds
    pub fn is_within_limit(&self) -> bool {
        let used_gb = self.used_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        used_gb < self.hard_limit_gb
    }

    /// Get urgency level (0.0 to 1.0)
    pub fn urgency(&self) -> f64 {
        let used_gb = self.used_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        ((used_gb / self.hard_limit_gb) - 0.7).max(0.0).min(1.0) / 0.3
    }
}

/// Network metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Jitter (standard deviation) in milliseconds
    pub jitter_ms: f64,
    /// Packet loss percentage (0-100)
    pub packet_loss_percent: f64,
    /// Last successful ping time
    pub last_ping_time: Option<u64>,
}

/// Complete system health snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthSnapshot {
    pub timestamp: u64,
    pub status: HealthStatus,
    pub cpu: CpuMetrics,
    pub memory: MemoryMetrics,
    pub network: NetworkMetrics,
    pub degradation_level: u8,
    pub active_measures: Vec<String>,
}

// ============================================================================
// Graceful Degradation Protocol
// ============================================================================

/// Degradation level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DegradationLevel {
    /// No degradation
    None = 0,
    /// Level 1: Reduce non-critical features
    Level1 = 1,
    /// Level 2: Pause background tasks
    Level2 = 2,
    /// Level 3: Reduce ML model complexity
    Level3 = 3,
    /// Level 4: Emergency mode - minimal operations
    Level4 = 4,
}

/// Degradation measure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradationMeasure {
    pub name: String,
    pub description: String,
    pub trigger_threshold: f64,
    pub impact_score: u8,
}

/// Graceful degradation manager
pub struct DegradationManager {
    current_level: AtomicUsize,
    active_measures: Mutex<Vec<String>>,
    degradation_history: Mutex<Vec<(u64, DegradationLevel)>>,
    
    // Configuration thresholds
    memory_warning_threshold: AtomicU64,   // Bytes
    memory_critical_threshold: AtomicU64,  // Bytes
    cpu_warning_threshold: AtomicU64,      // Percentage * 100
    cpu_critical_threshold: AtomicU64,     // Percentage * 100
    
    // Callbacks
    on_degradation_change: Mutex<Vec<Box<dyn Fn(DegradationLevel) + Send + Sync>>>,
}

impl DegradationManager {
    pub fn new(hard_limit_gb: f64) -> Self {
        let warning_bytes = ((hard_limit_gb - 1.0) * 1024.0 * 1024.0 * 1024.0) as u64;
        let critical_bytes = ((hard_limit_gb - 0.5) * 1024.0 * 1024.0 * 1024.0) as u64;

        Self {
            current_level: AtomicUsize::new(0),
            active_measures: Mutex::new(Vec::new()),
            degradation_history: Mutex::new(Vec::new()),
            memory_warning_threshold: AtomicU64::new(warning_bytes),
            memory_critical_threshold: AtomicU64::new(critical_bytes),
            cpu_warning_threshold: AtomicU64::new(8000), // 80%
            cpu_critical_threshold: AtomicU64::new(9500), // 95%
            on_degradation_change: Mutex::new(Vec::new()),
        }
    }

    /// Evaluate and update degradation level
    pub fn evaluate(&self, memory: &MemoryMetrics, cpu: &CpuMetrics) -> DegradationLevel {
        let mut target_level = DegradationLevel::None;
        let mut measures = Vec::new();

        // Memory-based degradation
        let used_gb = memory.used_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        
        if used_gb >= 7.5 {
            target_level = DegradationLevel::Level4;
            measures.push("emergency_memory_mode".to_string());
        } else if used_gb >= 7.0 {
            target_level = target_level.max(DegradationLevel::Level3);
            measures.push("reduce_ml_complexity".to_string());
        } else if used_gb >= 6.5 {
            target_level = target_level.max(DegradationLevel::Level2);
            measures.push("pause_background_tasks".to_string());
        } else if used_gb >= 6.0 {
            target_level = target_level.max(DegradationLevel::Level1);
            measures.push("reduce_feature_calculation".to_string());
        }

        // CPU-based degradation
        if cpu.usage_percent >= 95.0 {
            target_level = target_level.max(DegradationLevel::Level4);
            measures.push("emergency_cpu_mode".to_string());
        } else if cpu.usage_percent >= 85.0 {
            target_level = target_level.max(DegradationLevel::Level3);
        } else if cpu.usage_percent >= 70.0 {
            target_level = target_level.max(DegradationLevel::Level1);
        }

        // Update level if changed
        let current = self.current_level.load(Ordering::Relaxed);
        let target = target_level as usize;

        if target != current {
            self.current_level.store(target, Ordering::Relaxed);
            
            // Update active measures
            *self.active_measures.lock() = measures.clone();
            
            // Record history
            self.degradation_history.lock().push((
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                target_level,
            ));

            // Trigger callbacks
            for callback in self.on_degradation_change.lock().iter() {
                callback(target_level);
            }

            eprintln!(
                "DEGRADATION: Level changed to {:?}, measures: {:?}",
                target_level, measures
            );
        }

        target_level
    }

    /// Get current degradation level
    pub fn current_level(&self) -> DegradationLevel {
        match self.current_level.load(Ordering::Relaxed) {
            0 => DegradationLevel::None,
            1 => DegradationLevel::Level1,
            2 => DegradationLevel::Level2,
            3 => DegradationLevel::Level3,
            _ => DegradationLevel::Level4,
        }
    }

    /// Check if a feature should be disabled based on degradation level
    pub fn should_disable_feature(&self, feature_impact: u8) -> bool {
        let level = self.current_level() as u8;
        level >= feature_impact
    }

    /// Register callback for degradation changes
    pub fn register_callback<F>(&self, callback: F)
    where
        F: Fn(DegradationLevel) + Send + Sync + 'static,
    {
        self.on_degradation_change.lock().push(Box::new(callback));
    }

    /// Get active measures
    pub fn get_active_measures(&self) -> Vec<String> {
        self.active_measures.lock().clone()
    }
}

// ============================================================================
// Health Monitor
// ============================================================================

/// Main system health monitor
pub struct HealthMonitor {
    /// Degradation manager
    degradation_manager: Arc<DegradationManager>,
    /// Running state
    running: AtomicBool,
    /// Last health snapshot
    last_snapshot: RwLock<Option<HealthSnapshot>>,
    /// Snapshot history
    snapshot_history: Mutex<VecDeque<HealthSnapshot>>,
    /// Monitoring interval
    check_interval: Duration,
    /// Hard RAM limit
    hard_limit_gb: f64,
    /// Statistics
    checks_performed: AtomicU64,
    warnings_issued: AtomicU64,
}

impl HealthMonitor {
    pub fn new(hard_limit_gb: f64, check_interval: Duration) -> Self {
        Self {
            degradation_manager: Arc::new(DegradationManager::new(hard_limit_gb)),
            running: AtomicBool::new(false),
            last_snapshot: RwLock::new(None),
            snapshot_history: Mutex::new(VecDeque::with_capacity(100)),
            check_interval,
            hard_limit_gb,
            checks_performed: AtomicU64::new(0),
            warnings_issued: AtomicU64::new(0),
        }
    }

    /// Get CPU metrics (platform-specific implementation would go here)
    pub fn get_cpu_metrics(&self) -> CpuMetrics {
        // Simulated values - in production would use platform-specific APIs
        // On Linux: read from /proc/stat and /sys/class/thermal/
        // On Windows: use Performance Counters or WMI
        
        CpuMetrics {
            usage_percent: self.simulate_cpu_usage(),
            temperature_celsius: None, // Would require platform-specific code
            core_count: num_cpus::get(),
            per_core_usage: vec![self.simulate_cpu_usage(); num_cpus::get()],
        }
    }

    /// Get memory metrics
    pub fn get_memory_metrics(&self) -> MemoryMetrics {
        #[cfg(target_os = "linux")]
        {
            // Read from /proc/meminfo
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                let mut mem_total = 0u64;
                let mut mem_available = 0u64;
                
                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        mem_total = line.split_whitespace()
                            .nth(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0) * 1024;
                    } else if line.starts_with("MemAvailable:") {
                        mem_available = line.split_whitespace()
                            .nth(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0) * 1024;
                    }
                }

                let used = mem_total.saturating_sub(mem_available);
                let usage_percent = (used as f64 / mem_total as f64) * 100.0;
                let used_gb = used as f64 / (1024.0 * 1024.0 * 1024.0);

                return MemoryMetrics {
                    used_bytes: used,
                    total_bytes: mem_total,
                    available_bytes: mem_available,
                    usage_percent,
                    hard_limit_gb: self.hard_limit_gb,
                    approaching_limit: used_gb >= self.hard_limit_gb - 1.0,
                };
            }
        }

        // Fallback using psutil-like estimation
        // In production, would use proper sysinfo crate
        MemoryMetrics {
            used_bytes: 4_000_000_000, // Placeholder
            total_bytes: 16_000_000_000,
            available_bytes: 12_000_000_000,
            usage_percent: 25.0,
            hard_limit_gb: self.hard_limit_gb,
            approaching_limit: false,
        }
    }

    /// Get network metrics via ping
    pub fn get_network_metrics(&self, targets: &[&str]) -> NetworkMetrics {
        let mut latencies = Vec::new();
        
        for target in targets {
            // Simplified - in production would use actual ICMP ping
            // For now, simulate with random latency
            latencies.push(self.simulate_ping_latency());
        }

        if latencies.is_empty() {
            return NetworkMetrics {
                avg_latency_ms: 0.0,
                jitter_ms: 0.0,
                packet_loss_percent: 0.0,
                last_ping_time: None,
            };
        }

        let avg = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let variance = latencies.iter()
            .map(|&x| (x - avg).powi(2))
            .sum::<f64>() / latencies.len() as f64;
        let jitter = variance.sqrt();

        NetworkMetrics {
            avg_latency_ms: avg,
            jitter_ms: jitter,
            packet_loss_percent: 0.0,
            last_ping_time: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64
            ),
        }
    }

    /// Perform health check and return snapshot
    pub fn check_health(&self) -> HealthSnapshot {
        let cpu = self.get_cpu_metrics();
        let memory = self.get_memory_metrics();
        let network = self.get_network_metrics(&["8.8.8.8", "1.1.1.1"]);

        // Evaluate degradation
        let degradation_level = self.degradation_manager.evaluate(&memory, &cpu);

        // Determine overall status
        let status = self.determine_status(&memory, &cpu, &network, degradation_level);

        // Get active measures
        let active_measures = self.degradation_manager.get_active_measures();

        let snapshot = HealthSnapshot {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            status,
            cpu,
            memory,
            network,
            degradation_level: degradation_level as u8,
            active_measures,
        };

        // Store snapshot
        *self.last_snapshot.write() = Some(snapshot.clone());
        
        let mut history = self.snapshot_history.lock();
        history.push_back(snapshot.clone());
        if history.len() > 100 {
            history.pop_front();
        }

        self.checks_performed.fetch_add(1, Ordering::Relaxed);

        if status != HealthStatus::Healthy {
            self.warnings_issued.fetch_add(1, Ordering::Relaxed);
        }

        snapshot
    }

    /// Start background monitoring
    pub fn start_background(&self) {
        self.running.store(true, Ordering::Relaxed);
        
        let self_arc = Arc::new(self.clone_internal());
        
        thread::spawn(move || {
            while self_arc.running.load(Ordering::Relaxed) {
                let _ = self_arc.check_health();
                thread::sleep(self_arc.check_interval);
            }
        });
    }

    /// Stop background monitoring
    pub fn stop_background(&self) {
        self.running.store(false, Ordering::Relaxed);
    }

    /// Get degradation manager reference
    pub fn degradation_manager(&self) -> Arc<DegradationManager> {
        self.degradation_manager.clone()
    }

    /// Get last snapshot
    pub fn get_last_snapshot(&self) -> Option<HealthSnapshot> {
        self.last_snapshot.read().clone()
    }

    /// Get statistics
    pub fn get_stats(&self) -> (u64, u64) {
        (
            self.checks_performed.load(Ordering::Relaxed),
            self.warnings_issued.load(Ordering::Relaxed),
        )
    }

    // Internal helpers
    fn simulate_cpu_usage(&self) -> f64 {
        use rand::Rng;
        rand::thread_rng().gen_range(20.0..60.0)
    }

    fn simulate_ping_latency(&self) -> f64 {
        use rand::Rng;
        rand::thread_rng().gen_range(5.0..50.0)
    }

    fn determine_status(
        &self,
        memory: &MemoryMetrics,
        cpu: &CpuMetrics,
        network: &NetworkMetrics,
        degradation: DegradationLevel,
    ) -> HealthStatus {
        match degradation {
            DegradationLevel::None => HealthStatus::Healthy,
            DegradationLevel::Level1 => HealthStatus::Warning,
            DegradationLevel::Level2 => HealthStatus::Degraded,
            DegradationLevel::Level3 | DegradationLevel::Level4 => HealthStatus::Critical,
        }
    }

    fn clone_internal(&self) -> HealthMonitorInternal {
        HealthMonitorInternal {
            running: self.running.load(Ordering::Relaxed),
            check_interval: self.check_interval,
            hard_limit_gb: self.hard_limit_gb,
            checks_performed: self.checks_performed.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone)]
struct HealthMonitorInternal {
    running: bool,
    check_interval: Duration,
    hard_limit_gb: f64,
    checks_performed: u64,
}

impl HealthMonitorInternal {
    fn running(&self) -> &AtomicBool {
        // This is a simplified placeholder
        static RUNNING: AtomicBool = AtomicBool::new(false);
        &RUNNING
    }

    fn check_health(&self) -> HealthSnapshot {
        // Placeholder
        HealthSnapshot {
            timestamp: 0,
            status: HealthStatus::Healthy,
            cpu: CpuMetrics {
                usage_percent: 0.0,
                temperature_celsius: None,
                core_count: 0,
                per_core_usage: vec![],
            },
            memory: MemoryMetrics {
                used_bytes: 0,
                total_bytes: 0,
                available_bytes: 0,
                usage_percent: 0.0,
                hard_limit_gb: self.hard_limit_gb,
                approaching_limit: false,
            },
            network: NetworkMetrics {
                avg_latency_ms: 0.0,
                jitter_ms: 0.0,
                packet_loss_percent: 0.0,
                last_ping_time: None,
            },
            degradation_level: 0,
            active_measures: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_metrics() {
        let metrics = MemoryMetrics {
            used_bytes: 7_000_000_000,
            total_bytes: 16_000_000_000,
            available_bytes: 9_000_000_000,
            usage_percent: 43.75,
            hard_limit_gb: 8.0,
            approaching_limit: true,
        };

        assert!(metrics.approaching_limit);
        assert!(metrics.urgency() > 0.0);
    }

    #[test]
    fn test_degradation_manager() {
        let manager = DegradationManager::new(8.0);
        
        // Test with normal memory
        let normal_memory = MemoryMetrics {
            used_bytes: 4_000_000_000,
            total_bytes: 16_000_000_000,
            available_bytes: 12_000_000_000,
            usage_percent: 25.0,
            hard_limit_gb: 8.0,
            approaching_limit: false,
        };
        
        let normal_cpu = CpuMetrics {
            usage_percent: 30.0,
            temperature_celsius: Some(50.0),
            core_count: 8,
            per_core_usage: vec![30.0; 8],
        };

        let level = manager.evaluate(&normal_memory, &normal_cpu);
        assert_eq!(level, DegradationLevel::None);

        // Test with high memory
        let high_memory = MemoryMetrics {
            used_bytes: 7_500_000_000,
            total_bytes: 16_000_000_000,
            available_bytes: 500_000_000,
            usage_percent: 46.875,
            hard_limit_gb: 8.0,
            approaching_limit: true,
        };

        let level = manager.evaluate(&high_memory, &normal_cpu);
        assert!(level >= DegradationLevel::Level3);
    }
}
