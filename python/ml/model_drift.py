"""
Model Drift Detection System
Implements Page-Hinkley test, KL divergence, and Population Stability Index (PSI)
Automatically triggers model retraining or fallback to deterministic logic
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from scipy import stats
from scipy.special import rel_entr
import time
import threading


# ============================================================================
# PAGE-HINKLEY TEST FOR CONCEPT DRIFT
# ============================================================================

class PageHinkleyTest:
    """
    Page-Hinkley test for detecting concept drift in data streams.
    Monitors changes in the mean of prediction errors.
    """
    
    def __init__(
        self,
        delta: float = 0.005,
        threshold: float = 50.0,
        min_instances: int = 30,
        alpha: float = 1 - 0.0001
    ):
        """
        Initialize Page-Hinkley test.
        
        Args:
            delta: Allowable deviation magnitude
            threshold: Threshold for drift detection
            min_instances: Minimum instances before testing
            alpha: Forgetting factor for exponential weighting
        """
        self.delta = delta
        self.threshold = threshold
        self.min_instances = min_instances
        self.alpha = alpha
        
        # Running statistics
        self.x_mean = 0.0
        self.sum = 0.0
        self.n_instances = 0
        self.ph_max = 0.0
        self.ph_min = float('inf')
        
        # History for visualization
        self.history = deque(maxlen=1000)
        
    def update(self, x: float) -> bool:
        """
        Update test with new observation.
        Returns True if drift detected.
        """
        self.n_instances += 1
        
        # Update running mean with exponential weighting
        self.x_mean = self.x_mean + self.alpha * (x - self.x_mean)
        
        # Update cumulative sum
        self.sum = self.sum + (x - self.x_mean - self.delta)
        
        # Track max and min
        self.ph_max = max(self.ph_max, self.sum)
        self.ph_min = min(self.ph_min, self.sum)
        
        # Store history
        self.history.append({
            'n': self.n_instances,
            'sum': self.sum,
            'ph_range': self.ph_max - self.ph_min
        })
        
        # Check for drift
        if self.n_instances >= self.min_instances:
            ph_range = self.ph_max - self.ph_min
            if ph_range > self.threshold:
                return True
        
        return False
    
    def reset(self):
        """Reset the test statistics."""
        self.x_mean = 0.0
        self.sum = 0.0
        self.n_instances = 0
        self.ph_max = 0.0
        self.ph_min = float('inf')


# ============================================================================
# KL DIVERGENCE MONITOR
# ============================================================================

class KLDivergenceMonitor:
    """
    Monitor KL divergence between reference and current distributions.
    Detects distribution shifts in model predictions or features.
    """
    
    def __init__(
        self,
        n_bins: int = 20,
        threshold: float = 0.5,
        window_size: int = 500,
        smoothing_epsilon: float = 1e-10
    ):
        """
        Initialize KL divergence monitor.
        
        Args:
            n_bins: Number of bins for histogram
            threshold: KL divergence threshold for drift
            window_size: Size of sliding window
            smoothing_epsilon: Small value to prevent log(0)
        """
        self.n_bins = n_bins
        self.threshold = threshold
        self.window_size = window_size
        self.smoothing_epsilon = smoothing_epsilon
        
        # Reference distribution (baseline)
        self.reference_dist: Optional[np.ndarray] = None
        self.reference_set = False
        
        # Current window
        self.current_window = deque(maxlen=window_size)
        
        # Bin edges
        self.bin_edges: Optional[np.ndarray] = None
        
        # History
        self.kl_history = deque(maxlen=100)
        
    def set_reference(self, data: np.ndarray):
        """Set reference distribution from baseline data."""
        self.reference_dist, self.bin_edges = np.histogram(
            data, bins=self.n_bins, density=True
        )
        # Add smoothing
        self.reference_dist = self.reference_dist + self.smoothing_epsilon
        self.reference_dist = self.reference_dist / self.reference_dist.sum()
        self.reference_set = True
        
    def update(self, x: float) -> Optional[float]:
        """
        Update monitor with new observation.
        Returns KL divergence if calculable.
        """
        self.current_window.append(x)
        
        if len(self.current_window) < self.window_size // 2:
            return None
        
        if not self.reference_set:
            # Auto-set reference on first sufficient window
            self.set_reference(np.array(list(self.current_window)))
            return None
        
        # Calculate current distribution
        current_dist, _ = np.histogram(
            np.array(list(self.current_window)),
            bins=self.bin_edges,
            density=True
        )
        current_dist = current_dist + self.smoothing_epsilon
        current_dist = current_dist / current_dist.sum()
        
        # Calculate KL divergence
        kl_div = np.sum(rel_entr(current_dist, self.reference_dist))
        
        self.kl_history.append(kl_div)
        
        return kl_div
    
    def is_drifted(self) -> bool:
        """Check if current distribution has drifted from reference."""
        if not self.kl_history:
            return False
        
        return self.kl_history[-1] > self.threshold
    
    def get_drift_magnitude(self) -> float:
        """Get current drift magnitude."""
        if not self.kl_history:
            return 0.0
        
        return self.kl_history[-1]


# ============================================================================
# POPULATION STABILITY INDEX (PSI)
# ============================================================================

class PopulationStabilityIndex:
    """
    Calculate Population Stability Index for feature drift detection.
    Commonly used in finance for monitoring feature distributions.
    """
    
    # PSI interpretation thresholds
    PSI_STABLE = 0.1
    PSI_MODERATE = 0.2
    PSI_SIGNIFICANT = 0.25
    
    def __init__(
        self,
        n_bins: int = 10,
        window_size: int = 1000,
        check_interval: int = 100
    ):
        """
        Initialize PSI calculator.
        
        Args:
            n_bins: Number of bins for categorization
            window_size: Sliding window size
            check_interval: How often to calculate PSI
        """
        self.n_bins = n_bins
        self.window_size = window_size
        self.check_interval = check_interval
        
        # Reference distribution
        self.reference_dist: Optional[np.ndarray] = None
        self.reference_count = 0
        
        # Current window
        self.current_window = deque(maxlen=window_size)
        
        # PSI history per feature
        self.psi_history: Dict[str, deque] = {}
        
        # Last check count
        self.last_check = 0
        
    def set_reference(self, data: Dict[str, np.ndarray]):
        """Set reference distributions for multiple features."""
        self.reference_dist = {}
        
        for feature_name, feature_data in data.items():
            dist, _ = np.histogram(feature_data, bins=self.n_bins)
            # Add small epsilon to avoid division by zero
            dist = dist + 1
            self.reference_dist[feature_name] = dist.astype(float)
            self.reference_count = len(feature_data)
            
            # Initialize history
            self.psi_history[feature_name] = deque(maxlen=100)
    
    def update(self, feature_values: Dict[str, float]) -> Optional[Dict[str, float]]:
        """
        Update with new feature values.
        Returns PSI values if calculation triggered.
        """
        # Add to window
        for feature_name, value in feature_values.items():
            # Store as single-item array for consistency
            if feature_name not in self.current_window:
                self.current_window[feature_name] = deque(maxlen=self.window_size)
            self.current_window[feature_name].append(value)
        
        self.last_check += 1
        
        # Check if it's time to calculate PSI
        if self.last_check % self.check_interval != 0:
            return None
        
        if self.reference_dist is None:
            return None
        
        psi_values = {}
        
        for feature_name, ref_dist in self.reference_dist.items():
            if feature_name not in self.current_window:
                continue
            
            current_data = np.array(list(self.current_window[feature_name]))
            
            if len(current_data) < self.window_size // 2:
                continue
            
            # Calculate current distribution
            curr_dist, _ = np.histogram(current_data, bins=self.n_bins)
            curr_dist = curr_dist + 1  # Smoothing
            
            # Normalize to proportions
            ref_prop = ref_dist / ref_dist.sum()
            curr_prop = curr_dist.astype(float) / curr_dist.sum()
            
            # Calculate PSI
            psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
            
            psi_values[feature_name] = psi
            self.psi_history[feature_name].append(psi)
        
        return psi_values
    
    def get_stability_status(self, feature_name: str) -> str:
        """
        Get stability status for a feature.
        Returns: 'stable', 'moderate_shift', 'significant_shift'
        """
        if feature_name not in self.psi_history or not self.psi_history[feature_name]:
            return 'unknown'
        
        psi = self.psi_history[feature_name][-1]
        
        if psi < self.PSI_STABLE:
            return 'stable'
        elif psi < self.PSI_MODERATE:
            return 'moderate_shift'
        else:
            return 'significant_shift'


# ============================================================================
# COMPREHENSIVE DRIFT DETECTOR
# ============================================================================

class DriftDetectionResult:
    """Container for drift detection results."""
    
    def __init__(self):
        self.concept_drift_detected = False
        self.distribution_drift_detected = False
        self.feature_drifts: Dict[str, str] = {}
        self.drift_severity = 'none'  # none, low, medium, high
        self.recommended_action = 'none'  # none, retrain, fallback, pause
        self.metrics: Dict[str, float] = {}
        self.timestamp = time.time()


class ComprehensiveDriftDetector:
    """
    Main drift detection system combining multiple methods.
    Coordinates Page-Hinkley, KL divergence, and PSI monitors.
    """
    
    def __init__(
        self,
        ph_threshold: float = 50.0,
        kl_threshold: float = 0.5,
        psi_features: Optional[List[str]] = None,
        cooldown_period: int = 300  # seconds
    ):
        """
        Initialize comprehensive drift detector.
        
        Args:
            ph_threshold: Page-Hinkley threshold
            kl_threshold: KL divergence threshold
            psi_features: List of feature names for PSI monitoring
            cooldown_period: Minimum time between drift alerts
        """
        # Page-Hinkley for prediction errors
        self.ph_test = PageHinkleyTest(threshold=ph_threshold)
        
        # KL divergence for prediction distributions
        self.kl_monitor = KLDivergenceMonitor(threshold=kl_threshold)
        
        # PSI for feature monitoring
        self.psi_monitor = PopulationStabilityIndex()
        self.psi_features = psi_features or []
        
        # Cooldown management
        self.cooldown_period = cooldown_period
        self.last_drift_time = 0.0
        
        # Fallback mode
        self.fallback_mode = False
        self.fallback_triggers = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Callbacks
        self.on_drift_callbacks = []
        
        # History
        self.detection_history = deque(maxlen=500)
        
    def initialize_reference(
        self,
        reference_errors: np.ndarray,
        reference_predictions: np.ndarray,
        reference_features: Optional[Dict[str, np.ndarray]] = None
    ):
        """
        Initialize reference distributions from baseline data.
        
        Args:
            reference_errors: Baseline prediction errors
            reference_predictions: Baseline model predictions
            reference_features: Baseline feature values per feature name
        """
        with self.lock:
            # Reset all monitors
            self.ph_test.reset()
            
            # Set KL reference from predictions
            self.kl_monitor.set_reference(reference_predictions)
            
            # Set PSI reference from features
            if reference_features and self.psi_features:
                feature_data = {
                    k: v for k, v in reference_features.items()
                    if k in self.psi_features
                }
                self.psi_monitor.set_reference(feature_data)
    
    def update(
        self,
        error: float,
        prediction: float,
        features: Optional[Dict[str, float]] = None
    ) -> DriftDetectionResult:
        """
        Update all drift detectors with new observation.
        
        Args:
            error: Current prediction error
            prediction: Current model prediction
            features: Current feature values
        
        Returns:
            DriftDetectionResult with findings and recommendations
        """
        result = DriftDetectionResult()
        
        with self.lock:
            # Update Page-Hinkley test
            ph_drift = self.ph_test.update(error)
            result.concept_drift_detected = ph_drift
            
            # Update KL divergence monitor
            kl_div = self.kl_monitor.update(prediction)
            if kl_div is not None:
                result.distribution_drift_detected = self.kl_monitor.is_drifted()
                result.metrics['kl_divergence'] = kl_div
            
            # Update PSI monitor
            if features and self.psi_features:
                feature_subset = {
                    k: v for k, v in features.items()
                    if k in self.psi_features
                }
                psi_values = self.psi_monitor.update(feature_subset)
                
                if psi_values:
                    for feature_name, psi in psi_values.items():
                        result.feature_drifts[feature_name] = \
                            self.psi_monitor.get_stability_status(feature_name)
                        result.metrics[f'psi_{feature_name}'] = psi
            
            # Determine overall severity
            result.drift_severity = self._calculate_severity(result)
            
            # Determine recommended action
            result.recommended_action = self._get_recommendation(result)
            
            # Store metrics
            result.metrics['ph_sum'] = self.ph_test.sum
            result.metrics['ph_range'] = self.ph_test.ph_max - self.ph_test.ph_min
            
            # Check cooldown
            current_time = time.time()
            if result.drift_severity != 'none':
                if current_time - self.last_drift_time < self.cooldown_period:
                    # Still in cooldown, don't trigger action
                    result.recommended_action = 'monitor'
                else:
                    self.last_drift_time = current_time
                    self._trigger_callbacks(result)
            
            # Store in history
            self.detection_history.append(result)
        
        return result
    
    def _calculate_severity(self, result: DriftDetectionResult) -> str:
        """Calculate overall drift severity."""
        severity_score = 0
        
        if result.concept_drift_detected:
            severity_score += 2
        
        if result.distribution_drift_detected:
            severity_score += 2
        
        # Count significant feature drifts
        significant_drifts = sum(
            1 for status in result.feature_drifts.values()
            if status == 'significant_shift'
        )
        moderate_drifts = sum(
            1 for status in result.feature_drifts.values()
            if status == 'moderate_shift'
        )
        
        severity_score += significant_drifts * 2
        severity_score += moderate_drifts
        
        if severity_score == 0:
            return 'none'
        elif severity_score <= 2:
            return 'low'
        elif severity_score <= 4:
            return 'medium'
        else:
            return 'high'
    
    def _get_recommendation(self, result: DriftDetectionResult) -> str:
        """Get recommended action based on drift detection."""
        if result.drift_severity == 'none':
            return 'none'
        
        if result.drift_severity == 'high':
            self.fallback_triggers += 1
            if self.fallback_triggers >= 3:
                return 'pause'  # Multiple high severity events
            return 'fallback'
        
        if result.drift_severity == 'medium':
            return 'retrain'
        
        if result.drift_severity == 'low':
            return 'monitor'
        
        return 'none'
    
    def _trigger_callbacks(self, result: DriftDetectionResult):
        """Trigger registered callbacks on drift detection."""
        for callback in self.on_drift_callbacks:
            try:
                callback(result)
            except Exception as e:
                print(f"Error in drift callback: {e}")
    
    def register_callback(self, callback):
        """Register a callback function for drift events."""
        self.on_drift_callbacks.append(callback)
    
    def enable_fallback_mode(self):
        """Enable fallback to deterministic logic."""
        self.fallback_mode = True
        print("DRIFT DETECTOR: Fallback mode enabled")
    
    def disable_fallback_mode(self):
        """Disable fallback mode after retraining."""
        self.fallback_mode = False
        self.fallback_triggers = 0
        print("DRIFT DETECTOR: Fallback mode disabled")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection status."""
        return {
            'fallback_mode': self.fallback_mode,
            'fallback_triggers': self.fallback_triggers,
            'last_drift_time': self.last_drift_time,
            'cooldown_remaining': max(0, self.cooldown_period - (time.time() - self.last_drift_time)),
            'recent_drifts': len([
                r for r in self.detection_history
                if r.drift_severity != 'none'
            ])
        }


# ============================================================================
# AUTOMATED RETRAINING TRIGGER
# ============================================================================

class AutomatedRetrainingSystem:
    """
    Manages automated model retraining based on drift detection.
    Coordinates between drift detector and model training pipeline.
    """
    
    def __init__(
        self,
        drift_detector: ComprehensiveDriftDetector,
        retrain_func,
        fallback_func,
        check_interval: int = 60  # seconds
    ):
        """
        Initialize automated retraining system.
        
        Args:
            drift_detector: Drift detection instance
            retrain_func: Function to call for retraining
            fallback_func: Function to call for fallback mode
            check_interval: How often to check drift status
        """
        self.drift_detector = drift_detector
        self.retrain_func = retrain_func
        self.fallback_func = fallback_func
        self.check_interval = check_interval
        
        self.last_check = 0
        self.retrain_pending = False
        self.is_retraining = False
        
        # Register callback
        self.drift_detector.register_callback(self._on_drift_detected)
    
    def _on_drift_detected(self, result: DriftDetectionResult):
        """Handle drift detection event."""
        print(f"DRIFT DETECTED: Severity={result.drift_severity}, Action={result.recommended_action}")
        
        if result.recommended_action == 'retrain':
            self.retrain_pending = True
        elif result.recommended_action == 'fallback':
            self.fallback_func()
        elif result.recommended_action == 'pause':
            print("CRITICAL: Trading paused due to severe drift")
    
    def check_and_act(self):
        """Check if retraining is needed and execute."""
        current_time = time.time()
        
        if current_time - self.last_check < self.check_interval:
            return
        
        self.last_check = current_time
        
        if self.retrain_pending and not self.is_retraining:
            self.is_retraining = True
            
            try:
                print("Starting automated retraining...")
                self.retrain_func()
                self.retrain_pending = False
                
                # Disable fallback after successful retrain
                self.drift_detector.disable_fallback_mode()
                
            except Exception as e:
                print(f"Retraining failed: {e}")
                self.retrain_pending = True  # Retry next cycle
            
            finally:
                self.is_retraining = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the retraining system."""
        return {
            'retrain_pending': self.retrain_pending,
            'is_retraining': self.is_retraining,
            'last_check': self.last_check,
            'drift_summary': self.drift_detector.get_summary()
        }


# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

if __name__ == '__main__':
    np.random.seed(42)
    
    # Create drift detector
    detector = ComprehensiveDriftDetector(
        ph_threshold=50.0,
        kl_threshold=0.5,
        psi_features=['feature_1', 'feature_2', 'feature_3'],
        cooldown_period=10  # Short cooldown for testing
    )
    
    # Generate reference data (stable regime)
    n_reference = 1000
    reference_errors = np.abs(np.random.randn(n_reference)) * 0.1
    reference_predictions = np.random.randn(n_reference)
    reference_features = {
        f'feature_{i}': np.random.randn(n_reference)
        for i in range(1, 4)
    }
    
    # Initialize reference
    detector.initialize_reference(
        reference_errors=reference_errors,
        reference_predictions=reference_predictions,
        reference_features=reference_features
    )
    
    print("Reference initialized. Simulating data stream...")
    
    # Simulate data stream with drift injection
    n_samples = 500
    drift_injected = False
    
    for i in range(n_samples):
        # Generate normal data
        if i < 200 or not drift_injected:
            error = np.abs(np.random.randn()) * 0.1
            prediction = np.random.randn()
            features = {f'feature_{j}': np.random.randn() for j in range(1, 4)}
        else:
            # Inject drift (shift in mean and variance)
            drift_injected = True
            error = np.abs(np.random.randn()) * 0.3 + 0.2  # Higher error
            prediction = np.random.randn() + 0.5  # Shifted predictions
            features = {f'feature_{j}': np.random.randn() + 0.3 for j in range(1, 4)}
        
        # Update detector
        result = detector.update(
            error=error,
            prediction=prediction,
            features=features
        )
        
        # Print significant events
        if result.drift_severity != 'none' and i % 50 == 0:
            print(f"\nSample {i}:")
            print(f"  Severity: {result.drift_severity}")
            print(f"  Concept drift: {result.concept_drift_detected}")
            print(f"  Distribution drift: {result.distribution_drift_detected}")
            print(f"  Recommended action: {result.recommended_action}")
            print(f"  Feature drifts: {result.feature_drifts}")
    
    # Print final summary
    print("\n" + "="*50)
    print("FINAL SUMMARY:")
    print(detector.get_summary())
