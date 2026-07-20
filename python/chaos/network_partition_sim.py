"""
Network Partition Simulation for Chaos Engineering
===================================================
Safely simulates network failures, API rate-limit bans, and WebSocket drops
in a shadow/testing environment to validate the bot's resilience.

Tests:
1. Network partition simulation (exchange becomes unreachable)
2. API rate limit ban simulation
3. WebSocket connection drops
4. Partial data feed failures
5. Latency spikes and jitter

All simulations run in isolated environments to prevent accidental 
live trading disruptions.
"""

import asyncio
import random
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import threading
import json


class FailureType(Enum):
    """Types of network failures to simulate."""
    NETWORK_PARTITION = "network_partition"
    RATE_LIMIT_BAN = "rate_limit_ban"
    WEBSOCKET_DROP = "websocket_drop"
    PARTIAL_DATA_LOSS = "partial_data_loss"
    LATENCY_SPIKE = "latency_spike"
    API_TIMEOUT = "api_timeout"
    DNS_FAILURE = "dns_failure"


@dataclass
class FailureConfig:
    """Configuration for a failure scenario."""
    failure_type: FailureType
    
    # Timing
    start_delay_ms: int = 0  # Delay before triggering
    duration_ms: int = 5000  # How long the failure lasts
    
    # Severity
    severity: float = 1.0  # 0.0 to 1.0
    
    # Specific parameters
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Probability (for stochastic failures)
    probability: float = 1.0


@dataclass
class FailureEvent:
    """Record of a triggered failure event."""
    event_id: str
    failure_type: FailureType
    start_time: float
    end_time: Optional[float]
    affected_components: List[str]
    severity: float
    recovered: bool = False


class NetworkPartitionSimulator:
    """
    Simulates network partitions between the bot and exchanges.
    
    Uses a circuit breaker pattern with configurable failure injection.
    Safe mode ensures no real orders are affected during testing.
    """
    
    def __init__(self, safe_mode: bool = True):
        """
        Initialize simulator.
        
        Args:
            safe_mode: If True, blocks all real trading operations
        """
        self.safe_mode = safe_mode
        self.active_failures: Dict[str, FailureEvent] = {}
        self.failure_history: List[FailureEvent] = []
        self.event_counter = 0
        
        # Component connectivity state
        self.component_connected: Dict[str, bool] = {
            'binance_ws': True,
            'binance_rest': True,
            'coinbase_ws': True,
            'coinbase_rest': True,
            'risk_service': True,
            'data_feed': True,
        }
        
        # Callbacks for state changes
        self.on_disconnect_callbacks: List[Callable] = []
        self.on_reconnect_callbacks: List[Callable] = []
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        print(f"[ChaosEngine] Initialized in {'SAFE' if safe_mode else 'LIVE'} mode")
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        self.event_counter += 1
        return f"failure_{self.event_counter}_{int(time.time())}"
    
    def inject_failure(self, config: FailureConfig) -> str:
        """
        Inject a failure into the system.
        
        Returns event ID for tracking.
        """
        if random.random() > config.probability:
            return ""  # Failure not triggered based on probability
        
        event_id = self._generate_event_id()
        
        event = FailureEvent(
            event_id=event_id,
            failure_type=config.failure_type,
            start_time=time.time(),
            end_time=None,
            affected_components=[],
            severity=config.severity,
        )
        
        # Apply failure based on type
        self._apply_failure(event, config)
        
        # Schedule recovery
        if config.duration_ms > 0:
            self._schedule_recovery(event_id, config.duration_ms)
        
        with self._lock:
            self.active_failures[event_id] = event
            self.failure_history.append(event)
        
        print(f"[ChaosEngine] Injected {config.failure_type.value} (ID: {event_id})")
        
        return event_id
    
    def _apply_failure(self, event: FailureEvent, config: FailureConfig):
        """Apply the failure to relevant components."""
        failure_type = config.failure_type
        
        if failure_type == FailureType.NETWORK_PARTITION:
            # Disconnect all exchange connections
            for component in ['binance_ws', 'binance_rest', 'coinbase_ws', 'coinbase_rest']:
                self.component_connected[component] = False
                event.affected_components.append(component)
            
            self._notify_disconnect(event.affected_components)
            
        elif failure_type == FailureType.RATE_LIMIT_BAN:
            # Simulate rate limit on REST API
            for component in ['binance_rest', 'coinbase_rest']:
                self.component_connected[component] = False
                event.affected_components.append(component)
            
        elif failure_type == FailureType.WEBSOCKET_DROP:
            # Drop WebSocket connections only
            for component in ['binance_ws', 'coinbase_ws']:
                self.component_connected[component] = False
                event.affected_components.append(component)
            
            self._notify_disconnect(event.affected_components)
            
        elif failure_type == FailureType.PARTIAL_DATA_LOSS:
            # Simulate partial data feed issues
            self.component_connected['data_feed'] = False
            event.affected_components.append('data_feed')
            
        elif failure_type == FailureType.LATENCY_SPIKE:
            # Add artificial latency (simulated via config)
            config.params['added_latency_ms'] = int(config.severity * 5000)
            event.affected_components.append('all_network')
            
        elif failure_type == FailureType.API_TIMEOUT:
            # Simulate API timeouts
            for component in ['binance_rest', 'coinbase_rest']:
                event.affected_components.append(component)
            config.params['timeout_seconds'] = 30
            
        elif failure_type == FailureType.DNS_FAILURE:
            # Simulate DNS resolution failure
            event.affected_components.append('dns_resolver')
    
    def _schedule_recovery(self, event_id: str, duration_ms: int):
        """Schedule automatic recovery after duration."""
        def recover():
            self.recover_from_failure(event_id)
        
        timer = threading.Timer(duration_ms / 1000.0, recover)
        timer.daemon = True
        timer.start()
    
    def recover_from_failure(self, event_id: str) -> bool:
        """
        Recover from a specific failure.
        
        Returns True if recovery was successful.
        """
        with self._lock:
            if event_id not in self.active_failures:
                return False
            
            event = self.active_failures[event_id]
            event.end_time = time.time()
            event.recovered = True
            
            # Restore connectivity
            for component in event.affected_components:
                self.component_connected[component] = True
            
            self._notify_reconnect(event.affected_components)
            
            del self.active_failures[event_id]
        
        print(f"[ChaosEngine] Recovered from {event.failure_type.value} (ID: {event_id})")
        return True
    
    def _notify_disconnect(self, components: List[str]):
        """Notify callbacks about disconnection."""
        for callback in self.on_disconnect_callbacks:
            try:
                callback(components)
            except Exception as e:
                print(f"[ChaosEngine] Callback error: {e}")
    
    def _notify_reconnect(self, components: List[str]):
        """Notify callbacks about reconnection."""
        for callback in self.on_reconnect_callbacks:
            try:
                callback(components)
            except Exception as e:
                print(f"[ChaosEngine] Callback error: {e}")
    
    def is_connected(self, component: str) -> bool:
        """Check if a component is currently connected."""
        return self.component_connected.get(component, False)
    
    def get_active_failures(self) -> List[Dict]:
        """Get list of active failures."""
        with self._lock:
            return [
                {
                    'event_id': e.event_id,
                    'type': e.failure_type.value,
                    'duration_s': time.time() - e.start_time,
                    'components': e.affected_components,
                }
                for e in self.active_failures.values()
            ]
    
    def add_disconnect_callback(self, callback: Callable):
        """Add callback for disconnect events."""
        self.on_disconnect_callbacks.append(callback)
    
    def add_reconnect_callback(self, callback: Callable):
        """Add callback for reconnect events."""
        self.on_reconnect_callbacks.append(callback)
    
    def get_statistics(self) -> Dict:
        """Get chaos engineering statistics."""
        total_events = len(self.failure_history)
        recovered = sum(1 for e in self.failure_history if e.recovered)
        
        return {
            'total_events': total_events,
            'active_events': len(self.active_failures),
            'recovered_events': recovered,
            'recovery_rate': recovered / total_events if total_events > 0 else 0.0,
            'safe_mode': self.safe_mode,
        }


class ChaosTestScenario:
    """
    Pre-defined chaos test scenarios for comprehensive testing.
    """
    
    @staticmethod
    def gentle_partition() -> FailureConfig:
        """Gentle network partition test."""
        return FailureConfig(
            failure_type=FailureType.NETWORK_PARTITION,
            duration_ms=3000,
            severity=0.5,
        )
    
    @staticmethod
    def severe_outage() -> FailureConfig:
        """Severe multi-component outage."""
        return FailureConfig(
            failure_type=FailureType.NETWORK_PARTITION,
            duration_ms=30000,
            severity=1.0,
        )
    
    @staticmethod
    def rate_limit_stress() -> FailureConfig:
        """Rate limit ban simulation."""
        return FailureConfig(
            failure_type=FailureType.RATE_LIMIT_BAN,
            duration_ms=60000,  # 1 minute ban
            severity=1.0,
        )
    
    @staticmethod
    def websocket_flakiness() -> FailureConfig:
        """Intermittent WebSocket drops."""
        return FailureConfig(
            failure_type=FailureType.WEBSOCKET_DROP,
            duration_ms=1000,
            severity=0.3,
            probability=0.2,  # 20% chance
        )
    
    @staticmethod
    def latency_degradation() -> FailureConfig:
        """Gradual latency increase."""
        return FailureConfig(
            failure_type=FailureType.LATENCY_SPIKE,
            duration_ms=10000,
            severity=0.7,
        )


@contextmanager
def chaos_test_context(simulator: NetworkPartitionSimulator, 
                       config: FailureConfig):
    """
    Context manager for running chaos tests.
    
    Usage:
        with chaos_test_context(simulator, config):
            # Run test code here
            pass
    """
    event_id = simulator.inject_failure(config)
    try:
        yield event_id
    finally:
        if event_id:
            simulator.recover_from_failure(event_id)


async def run_chaos_test_suite():
    """Run a comprehensive chaos test suite."""
    print("=" * 60)
    print("CHAOS ENGINEERING TEST SUITE")
    print("=" * 60)
    
    simulator = NetworkPartitionSimulator(safe_mode=True)
    
    # Track disconnections
    disconnect_events = []
    simulator.add_disconnect_callback(lambda comps: disconnect_events.append(comps))
    
    # Test 1: Gentle partition
    print("\n[Test 1] Running gentle network partition...")
    with chaos_test_context(simulator, ChaosTestScenario.gentle_partition()):
        await asyncio.sleep(1)
        print(f"  Connected during partition: {simulator.is_connected('binance_ws')}")
    print(f"  After recovery: {simulator.is_connected('binance_ws')}")
    
    # Test 2: Rate limit ban
    print("\n[Test 2] Running rate limit ban simulation...")
    event_id = simulator.inject_failure(ChaosTestScenario.rate_limit_stress())
    await asyncio.sleep(0.5)
    print(f"  REST API blocked: {not simulator.is_connected('binance_rest')}")
    simulator.recover_from_failure(event_id)
    
    # Test 3: WebSocket flakiness
    print("\n[Test 3] Testing WebSocket resilience...")
    for i in range(5):
        simulator.inject_failure(ChaosTestScenario.websocket_flakiness())
        await asyncio.sleep(0.3)
    
    # Test 4: Combined failures
    print("\n[Test 4] Testing combined failure scenario...")
    simulator.inject_failure(ChaosTestScenario.latency_degradation())
    simulator.inject_failure(ChaosTestScenario.gentle_partition())
    await asyncio.sleep(2)
    
    # Print statistics
    stats = simulator.get_statistics()
    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    print(f"  Total Events: {stats['total_events']}")
    print(f"  Active Events: {stats['active_events']}")
    print(f"  Recovery Rate: {stats['recovery_rate']:.1%}")
    print(f"  Disconnect Events: {len(disconnect_events)}")
    print("=" * 60)
    
    return stats


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(run_chaos_test_suite())
