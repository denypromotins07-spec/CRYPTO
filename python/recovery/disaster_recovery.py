"""
Disaster Recovery Protocol - Automated Emergency Response
==========================================================
Automated disaster recovery system that spins up emergency liquidation
scripts on backup cloud instances when the primary system fails.

Key capabilities:
1. Primary instance health monitoring with heartbeat
2. Automatic failover to backup cloud instance (AWS/GCP/Azure)
3. Emergency portfolio liquidation to protect capital
4. State recovery from last checkpoint
5. Post-mortem data collection for analysis

Designed to minimize losses during catastrophic failures while
maintaining strict security and access controls.
"""

import os
import json
import time
import hashlib
import hmac
import base64
import requests
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DisasterLevel(Enum):
    """Severity levels for disasters."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    CATASTROPHIC = "catastrophic"


@dataclass
class HealthStatus:
    """Health status of a component."""
    component_id: str
    is_healthy: bool
    last_heartbeat: float
    latency_ms: float
    error_count: int
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FailoverConfig:
    """Configuration for failover behavior."""
    # Heartbeat settings
    heartbeat_interval_s: int = 5
    heartbeat_timeout_s: int = 15
    consecutive_failures_threshold: int = 3
    
    # Cloud provider settings
    cloud_provider: str = "aws"  # aws, gcp, azure
    backup_region: str = "us-east-1"
    instance_type: str = "t3.micro"
    
    # Liquidation settings
    liquidation_threshold_dd: float = 0.15  # 15% drawdown triggers emergency
    liquidation_slippage_tolerance: float = 0.02  # 2% max slippage
    partial_liquidation_pct: float = 0.5  # Liquidate 50% initially
    
    # Security
    api_key_encrypted: bool = True
    require_2fa_confirm: bool = False
    
    # Notification
    alert_webhook_url: Optional[str] = None
    alert_email: Optional[str] = None


class HealthMonitor:
    """
    Monitors health of primary trading instance.
    
    Uses multiple health check methods:
    - Heartbeat pings
    - API response latency
    - Error rate tracking
    - Data feed freshness
    """
    
    def __init__(self, config: FailoverConfig):
        self.config = config
        self.component_status: Dict[str, HealthStatus] = {}
        self.failure_counts: Dict[str, int] = {}
        self.last_check_time: Dict[str, float] = {}
        
        # Callbacks
        self.on_failure_callbacks: List[callable] = []
        self.on_recovery_callbacks: List[callable] = []
        
        # Running state
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        logger.info("HealthMonitor initialized")
    
    def register_component(self, component_id: str):
        """Register a component for monitoring."""
        self.component_status[component_id] = HealthStatus(
            component_id=component_id,
            is_healthy=True,
            last_heartbeat=time.time(),
            latency_ms=0.0,
            error_count=0,
        )
        self.failure_counts[component_id] = 0
        self.last_check_time[component_id] = time.time()
    
    def report_heartbeat(self, component_id: str, latency_ms: float = 0.0):
        """Report successful heartbeat from component."""
        if component_id not in self.component_status:
            self.register_component(component_id)
        
        status = self.component_status[component_id]
        status.is_healthy = True
        status.last_heartbeat = time.time()
        status.latency_ms = latency_ms
        status.error_count = 0
        
        self.last_check_time[component_id] = time.time()
    
    def report_error(self, component_id: str, error: str):
        """Report error from component."""
        if component_id not in self.component_status:
            return
        
        status = self.component_status[component_id]
        status.error_count += 1
        status.details['last_error'] = error
        status.details['last_error_time'] = time.time()
        
        # Track consecutive failures
        self.failure_counts[component_id] = self.failure_counts.get(component_id, 0) + 1
        
        # Check if threshold exceeded
        if self.failure_counts[component_id] >= self.config.consecutive_failures_threshold:
            status.is_healthy = False
            self._on_component_failure(component_id)
    
    def _on_component_failure(self, component_id: str):
        """Handle component failure."""
        logger.warning(f"Component {component_id} failed health check")
        
        for callback in self.on_failure_callbacks:
            try:
                callback(component_id, self.get_disaster_level())
            except Exception as e:
                logger.error(f"Failure callback error: {e}")
    
    def get_disaster_level(self) -> DisasterLevel:
        """Calculate current disaster level based on health status."""
        healthy_count = sum(1 for s in self.component_status.values() if s.is_healthy)
        total_count = len(self.component_status)
        
        if total_count == 0:
            return DisasterLevel.NORMAL
        
        health_ratio = healthy_count / total_count
        
        if health_ratio == 1.0:
            return DisasterLevel.NORMAL
        elif health_ratio >= 0.8:
            return DisasterLevel.WARNING
        elif health_ratio >= 0.5:
            return DisasterLevel.CRITICAL
        elif health_ratio >= 0.2:
            return DisasterLevel.EMERGENCY
        else:
            return DisasterLevel.CATASTROPHIC
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            current_time = time.time()
            
            for component_id, status in list(self.component_status.items()):
                # Check for missed heartbeats
                time_since_heartbeat = current_time - status.last_heartbeat
                
                if time_since_heartbeat > self.config.heartbeat_timeout_s:
                    self.report_error(component_id, "Heartbeat timeout")
                elif time_since_heartbeat > self.config.heartbeat_interval_s * 2:
                    # Warning but not failure yet
                    status.details['heartbeat_delayed'] = True
            
            time.sleep(self.config.heartbeat_interval_s)
    
    def add_failure_callback(self, callback: callable):
        """Add callback for failure events."""
        self.on_failure_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: callable):
        """Add callback for recovery events."""
        self.on_recovery_callbacks.append(callback)


class EmergencyLiquidator:
    """
    Executes emergency portfolio liquidation.
    
    Designed to minimize losses during catastrophic events
    by quickly flattening positions across all exchanges.
    """
    
    def __init__(self, config: FailoverConfig, exchange_clients: Dict[str, Any]):
        self.config = config
        self.exchange_clients = exchange_clients
        self.liquidation_in_progress = False
        self.liquidation_results: Dict[str, Any] = {}
        
    def execute_emergency_liquidation(self, 
                                      portfolio_state: Dict,
                                      reason: str) -> Dict[str, Any]:
        """
        Execute emergency liquidation of all positions.
        
        Args:
            portfolio_state: Current portfolio positions and balances
            reason: Reason for liquidation
            
        Returns:
            Liquidation results summary
        """
        if self.liquidation_in_progress:
            logger.warning("Liquidation already in progress")
            return {'status': 'already_in_progress'}
        
        self.liquidation_in_progress = True
        logger.critical(f"EMERGENCY LIQUIDATION INITIATED: {reason}")
        
        results = {
            'start_time': time.time(),
            'reason': reason,
            'positions_liquidated': [],
            'total_proceeds': 0.0,
            'errors': [],
        }
        
        try:
            positions = portfolio_state.get('positions', {})
            
            for symbol, position in positions.items():
                if abs(position) < 1e-8:
                    continue
                
                try:
                    result = self._liquidate_position(symbol, position)
                    results['positions_liquidated'].append(result)
                    results['total_proceeds'] += result.get('proceeds', 0.0)
                except Exception as e:
                    logger.error(f"Liquidation error for {symbol}: {e}")
                    results['errors'].append({
                        'symbol': symbol,
                        'error': str(e)
                    })
            
            results['end_time'] = time.time()
            results['duration_s'] = results['end_time'] - results['start_time']
            results['status'] = 'completed'
            
            logger.info(f"Liquidation completed in {results['duration_s']:.2f}s")
            
        except Exception as e:
            logger.error(f"Liquidation failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        finally:
            self.liquidation_in_progress = False
            self.liquidation_results = results
        
        return results
    
    def _liquidate_position(self, symbol: str, position: float) -> Dict:
        """Liquidate a single position using market orders."""
        # Determine exchange and side
        side = 'sell' if position > 0 else 'buy'
        quantity = abs(position)
        
        # Use market order for immediate execution
        # In production, this would use actual exchange API
        result = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'order_type': 'market',
            'status': 'submitted',
            'proceeds': 0.0,  # Would be filled price * quantity
        }
        
        # Simulate execution (replace with actual API call)
        logger.info(f"Liquidating {quantity} {symbol} via market {side}")
        
        # In production:
        # client = self.exchange_clients.get(exchange)
        # order = client.create_market_order(symbol, side, quantity)
        # result['proceeds'] = order['filled'] * order['average']
        
        return result
    
    def execute_partial_liquidation(self, 
                                    portfolio_state: Dict,
                                    percentage: float) -> Dict:
        """
        Execute partial liquidation (reduce exposure by percentage).
        
        Used for graduated response to elevated risk levels.
        """
        percentage = min(max(percentage, 0.0), 1.0)
        
        reduced_positions = {}
        for symbol, position in portfolio_state.get('positions', {}).items():
            reduced_positions[symbol] = position * (1 - percentage)
        
        liquidation_qty = {}
        for symbol, position in portfolio_state.get('positions', {}).items():
            qty_to_liquidate = abs(position) * percentage
            if qty_to_liquidate > 1e-8:
                liquidation_qty[symbol] = qty_to_liquidate
        
        logger.info(f"Partial liquidation: reducing positions by {percentage*100:.1f}%")
        
        results = {'partial': True, 'reduction_pct': percentage}
        
        for symbol, qty in liquidation_qty.items():
            side = 'sell' if portfolio_state['positions'][symbol] > 0 else 'buy'
            result = self._liquidate_position(symbol, qty * (1 if side == 'sell' else -1))
            results.setdefault('liquidations', []).append(result)
        
        return results


class BackupInstanceManager:
    """
    Manages backup cloud instance for failover.
    
    Supports AWS, GCP, and Azure for geographic redundancy.
    """
    
    def __init__(self, config: FailoverConfig):
        self.config = config
        self.instance_id: Optional[str] = None
        self.instance_state: str = "stopped"
        self.last_health_check: float = 0
        
    def provision_backup_instance(self) -> bool:
        """Provision backup instance on cloud provider."""
        logger.info(f"Provisioning backup instance on {self.config.cloud_provider}")
        
        try:
            if self.config.cloud_provider == "aws":
                self._provision_aws()
            elif self.config.cloud_provider == "gcp":
                self._provision_gcp()
            elif self.config.cloud_provider == "azure":
                self._provision_azure()
            else:
                raise ValueError(f"Unsupported cloud provider: {self.config.cloud_provider}")
            
            self.instance_state = "running"
            logger.info("Backup instance provisioned successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to provision backup instance: {e}")
            return False
    
    def _provision_aws(self):
        """Provision AWS EC2 instance."""
        # In production, use boto3
        # import boto3
        # ec2 = boto3.client('ec2', region_name=self.config.backup_region)
        # response = ec2.run_instances(
        #     InstanceType=self.config.instance_type,
        #     ImageId='ami-xxxxxxxxx',  # Pre-configured AMI with bot
        #     MinCount=1,
        #     MaxCount=1,
        # )
        # self.instance_id = response['Instances'][0]['InstanceId']
        self.instance_id = "i-emergency-backup"
        
    def _provision_gcp(self):
        """Provision GCP Compute Engine instance."""
        # In production, use google-cloud-compute library
        self.instance_id = "emergency-backup-gcp"
        
    def _provision_azure(self):
        """Provision Azure VM instance."""
        # In production, use azure-mgmt-compute library
        self.instance_id = "emergency-backup-azure"
    
    def terminate_backup_instance(self):
        """Terminate backup instance."""
        if not self.instance_id:
            return
        
        logger.info(f"Terminating backup instance {self.instance_id}")
        
        # In production, call cloud provider API
        # aws: ec2.terminate_instances(InstanceIds=[self.instance_id])
        
        self.instance_id = None
        self.instance_state = "terminated"
    
    def deploy_emergency_script(self, script_path: str) -> bool:
        """Deploy emergency liquidation script to backup instance."""
        if not self.instance_id:
            logger.error("No backup instance available")
            return False
        
        logger.info(f"Deploying emergency script to {self.instance_id}")
        
        # In production, use SSH or cloud-init to deploy
        # This would copy the liquidation script and dependencies
        
        return True
    
    def trigger_emergency_mode(self) -> bool:
        """Trigger emergency mode on backup instance."""
        if not self.instance_id:
            return False
        
        logger.warning("TRIGGERING EMERGENCY MODE ON BACKUP")
        
        # In production, send signal to backup instance
        # via cloud provider's run-command or similar
        
        return True


class DisasterRecoverySystem:
    """
    Main disaster recovery orchestrator.
    
    Coordinates health monitoring, failover decisions, and
    emergency response actions.
    """
    
    def __init__(self, config: Optional[FailoverConfig] = None):
        self.config = config or FailoverConfig()
        
        # Initialize components
        self.health_monitor = HealthMonitor(self.config)
        self.liquidator: Optional[EmergencyLiquidator] = None
        self.backup_manager = BackupInstanceManager(self.config)
        
        # State
        self.current_level = DisasterLevel.NORMAL
        self.failover_triggered = False
        self.recovery_log: List[Dict] = []
        
        # Setup callbacks
        self.health_monitor.add_failure_callback(self._on_health_failure)
        
        logger.info("DisasterRecoverySystem initialized")
    
    def initialize(self, exchange_clients: Dict[str, Any], 
                   portfolio_state: Dict):
        """Initialize with live connections."""
        self.liquidator = EmergencyLiquidator(self.config, exchange_clients)
        
        # Register critical components for monitoring
        for exchange in exchange_clients.keys():
            self.health_monitor.register_component(f"exchange_{exchange}")
        
        self.health_monitor.register_component("data_feed")
        self.health_monitor.register_component("risk_engine")
        self.health_monitor.register_component("order_router")
        
        # Start monitoring
        self.health_monitor.start_monitoring()
    
    def _on_health_failure(self, component_id: str, level: DisasterLevel):
        """Handle health failure event."""
        logger.warning(f"Health failure: {component_id}, Level: {level.value}")
        
        self.current_level = level
        self._log_event('health_failure', {
            'component': component_id,
            'level': level.value,
        })
        
        # Escalate response based on severity
        if level == DisasterLevel.CRITICAL:
            self._initiate_graduated_response()
        elif level in [DisasterLevel.EMERGENCY, DisasterLevel.CATASTROPHIC]:
            self._trigger_full_failover()
    
    def _initiate_graduated_response(self):
        """Initiate graduated response to elevated risk."""
        logger.info("Initiating graduated response")
        
        # Reduce position limits
        # Pause new alpha signals
        # Prepare for potential liquidation
        
        self._log_event('graduated_response', {
            'actions': ['position_limits_reduced', 'alpha_paused']
        })
    
    def _trigger_full_failover(self):
        """Trigger full failover to backup."""
        if self.failover_triggered:
            return
        
        logger.critical("TRIGGERING FULL FAILOVER")
        self.failover_triggered = True
        
        # Provision backup if not already done
        if not self.backup_manager.instance_id:
            self.backup_manager.provision_backup_instance()
        
        # Deploy and trigger emergency script
        self.backup_manager.deploy_emergency_script("/path/to/emergency_liquidation.py")
        self.backup_manager.trigger_emergency_mode()
        
        # Execute local emergency liquidation
        if self.liquidator:
            portfolio_state = self._get_current_portfolio()
            self.liquidator.execute_emergency_liquidation(
                portfolio_state,
                reason=f"Failover triggered - {self.current_level.value}"
            )
        
        # Send alerts
        self._send_alerts()
        
        self._log_event('full_failover', {
            'level': self.current_level.value,
            'backup_instance': self.backup_manager.instance_id,
        })
    
    def _get_current_portfolio(self) -> Dict:
        """Get current portfolio state (from checkpoint or API)."""
        # In production, load from last checkpoint or query exchanges
        return {
            'positions': {},
            'cash': {},
            'timestamp': time.time(),
        }
    
    def _send_alerts(self):
        """Send disaster alerts via configured channels."""
        message = {
            'level': self.current_level.value,
            'timestamp': time.time(),
            'failover_triggered': self.failover_triggered,
        }
        
        # Webhook alert
        if self.config.alert_webhook_url:
            try:
                requests.post(
                    self.config.alert_webhook_url,
                    json=message,
                    timeout=5
                )
            except Exception as e:
                logger.error(f"Webhook alert failed: {e}")
        
        # Email alert (would use SMTP library)
        if self.config.alert_email:
            logger.info(f"Email alert would be sent to {self.config.alert_email}")
    
    def _log_event(self, event_type: str, details: Dict):
        """Log disaster recovery event."""
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'details': details,
        }
        self.recovery_log.append(event)
        
        # Also log to file for post-mortem
        try:
            with open('/var/log/disaster_recovery.log', 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception:
            pass
    
    def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down disaster recovery system")
        self.health_monitor.stop_monitoring()
        
        # Keep backup instance running for safety
        # self.backup_manager.terminate_backup_instance()
    
    def get_status(self) -> Dict:
        """Get current DR status."""
        return {
            'disaster_level': self.current_level.value,
            'failover_triggered': self.failover_triggered,
            'backup_instance': self.backup_manager.instance_id,
            'backup_state': self.backup_manager.instance_state,
            'healthy_components': sum(
                1 for s in self.health_monitor.component_status.values() 
                if s.is_healthy
            ),
            'total_components': len(self.health_monitor.component_status),
        }


# Convenience function
def create_disaster_recovery(config_dict: Optional[Dict] = None) -> DisasterRecoverySystem:
    """Factory function for creating DR system."""
    if config_dict:
        config = FailoverConfig(**config_dict)
    else:
        config = FailoverConfig()
    
    return DisasterRecoverySystem(config)


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("DISASTER RECOVERY SYSTEM - DEMONSTRATION")
    print("=" * 60)
    
    dr_system = create_disaster_recovery({
        'heartbeat_interval_s': 5,
        'heartbeat_timeout_s': 10,
        'cloud_provider': 'aws',
        'alert_webhook_url': None,
    })
    
    # Initialize with mock clients
    dr_system.initialize(
        exchange_clients={'binance': {}, 'coinbase': {}},
        portfolio_state={'positions': {'BTC': 1.0, 'ETH': 10.0}}
    )
    
    # Simulate some health reports
    dr_system.health_monitor.report_heartbeat('exchange_binance', latency_ms=50)
    dr_system.health_monitor.report_heartbeat('exchange_coinbase', latency_ms=45)
    dr_system.health_monitor.report_heartbeat('data_feed', latency_ms=10)
    
    # Get status
    status = dr_system.get_status()
    print(f"\nCurrent Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Simulate failure
    print("\nSimulating component failure...")
    for i in range(4):
        dr_system.health_monitor.report_error('exchange_binance', f'Simulated error {i}')
        time.sleep(0.1)
    
    # Final status
    status = dr_system.get_status()
    print(f"\nAfter Failure:")
    print(f"  Disaster Level: {status['disaster_level']}")
    print(f"  Failover Triggered: {status['failover_triggered']}")
    
    # Cleanup
    dr_system.shutdown()
    
    print("\n" + "=" * 60)
    print("Demonstration complete")
    print("=" * 60)
