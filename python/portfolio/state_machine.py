"""
state_machine.py
----------------
Robust Finite State Machine (FSM) for bot operational states.
Enforces strict transition rules to prevent accidental trading during
data desyncs or API outages.

States: IDLE, WARMUP, TRADING, LIQUIDATING, HALT, ERROR
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Set
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class BotState(Enum):
    """Operational states for the trading bot."""
    IDLE = auto()        # Initial state, no active processes
    WARMUP = auto()      # Loading data, initializing models
    TRADING = auto()     # Active trading
    LIQUIDATING = auto() # Closing all positions
    HALT = auto()        # Paused by user/system
    ERROR = auto()       # Error state, requires intervention


# Define valid state transitions
VALID_TRANSITIONS: Dict[BotState, Set[BotState]] = {
    BotState.IDLE: {BotState.WARMUP, BotState.HALT},
    BotState.WARMUP: {BotState.TRADING, BotState.HALT, BotState.ERROR},
    BotState.TRADING: {BotState.LIQUIDATING, BotState.HALT, BotState.ERROR},
    BotState.LIQUIDATING: {BotState.IDLE, BotState.ERROR},
    BotState.HALT: {BotState.IDLE, BotState.WARMUP, BotState.ERROR},
    BotState.ERROR: {BotState.IDLE, BotState.HALT},
}

STATE_DESCRIPTIONS: Dict[BotState, str] = {
    BotState.IDLE: "System idle, ready to start",
    BotState.WARMUP: "Warming up models and data feeds",
    BotState.TRADING: "Active trading enabled",
    BotState.LIQUIDATING: "Liquidating all positions",
    BotState.HALT: "System halted by user or safety trigger",
    BotState.ERROR: "Error state - manual intervention required",
}


@dataclass
class StateTransition:
    """Record of a state transition event."""
    from_state: BotState
    to_state: BotState
    timestamp: datetime
    reason: str
    triggered_by: str  # 'user', 'system', 'auto'


class BotStateMachine:
    """
    Finite State Machine for trading bot lifecycle management.
    
    Features:
    - Strict transition validation
    - Callback hooks for state changes
    - Transition history logging
    - Automatic safety triggers
    """

    def __init__(self, initial_state: BotState = BotState.IDLE):
        self._current_state = initial_state
        self._transition_history: List[StateTransition] = []
        self._max_history = 1000
        
        # Callbacks for state changes
        self._callbacks: Dict[BotState, List[Callable]] = {
            state: [] for state in BotState
        }
        
        # State flags
        self._is_transitioning = False
        self._last_transition_time: Optional[datetime] = None
        
        # Safety counters
        self._error_count = 0
        self._max_errors_before_halt = 3
        
        logger.info(f"State machine initialized: {self._current_state.name}")

    @property
    def current_state(self) -> BotState:
        """Get current state."""
        return self._current_state

    @property
    def is_trading(self) -> bool:
        """Check if bot is in trading state."""
        return self._current_state == BotState.TRADING

    @property
    def is_halted(self) -> bool:
        """Check if bot is halted."""
        return self._current_state in (BotState.HALT, BotState.ERROR)

    def can_transition_to(self, target_state: BotState) -> bool:
        """Check if transition to target state is valid."""
        allowed = VALID_TRANSITIONS.get(self._current_state, set())
        return target_state in allowed

    def get_allowed_transitions(self) -> Set[BotState]:
        """Get set of allowed next states."""
        return VALID_TRANSITIONS.get(self._current_state, set())

    async def transition(
        self,
        target_state: BotState,
        reason: str = "",
        triggered_by: str = "user"
    ) -> bool:
        """
        Attempt to transition to a new state.
        Returns True if successful, False if invalid transition.
        """
        if self._is_transitioning:
            logger.warning("Transition already in progress")
            return False
        
        if not self.can_transition_to(target_state):
            logger.error(
                f"Invalid transition: {self._current_state.name} -> {target_state.name}"
            )
            return False
        
        self._is_transitioning = True
        
        try:
            # Execute pre-transition hooks
            await self._execute_pre_hooks(target_state)
            
            # Record transition
            transition = StateTransition(
                from_state=self._current_state,
                to_state=target_state,
                timestamp=datetime.utcnow(),
                reason=reason,
                triggered_by=triggered_by
            )
            
            old_state = self._current_state
            self._current_state = target_state
            self._last_transition_time = datetime.utcnow()
            
            # Store in history
            self._transition_history.append(transition)
            if len(self._transition_history) > self._max_history:
                self._transition_history.pop(0)
            
            logger.info(
                f"State transition: {old_state.name} -> {target_state.name} | Reason: {reason}"
            )
            
            # Execute post-transition hooks
            await self._execute_post_hooks(target_state)
            
            # Notify callbacks
            await self._notify_callbacks(target_state)
            
            # Reset error count on successful transition to non-error state
            if target_state != BotState.ERROR:
                self._error_count = 0
            
            return True
            
        finally:
            self._is_transitioning = False

    async def _execute_pre_hooks(self, target_state: BotState) -> None:
        """Execute actions before state transition."""
        if target_state == BotState.LIQUIDATING:
            logger.info("Pre-liquidation hook: Preparing to close all positions")
            # Signal position manager to start liquidation
        
        elif target_state == BotState.HALT:
            logger.info("Pre-halt hook: Canceling pending orders")
            # Signal execution client to cancel orders
        
        elif target_state == BotState.ERROR:
            logger.critical("Pre-error hook: Emergency stop initiated")
            # Immediate order cancellation

    async def _execute_post_hooks(self, target_state: BotState) -> None:
        """Execute actions after state transition."""
        if target_state == BotState.WARMUP:
            logger.info("Post-warmup hook: Starting data ingestion")
            # Start data feeds
        
        elif target_state == BotState.TRADING:
            logger.info("Post-trading hook: Enabling strategy signals")
            # Enable strategy execution
        
        elif target_state == BotState.IDLE:
            logger.info("Post-idle hook: Cleanup complete")
            # Release resources

    def register_callback(self, state: BotState, callback: Callable) -> None:
        """Register a callback for when a state is entered."""
        if state in self._callbacks:
            self._callbacks[state].append(callback)
            logger.debug(f"Registered callback for {state.name}")

    async def _notify_callbacks(self, state: BotState) -> None:
        """Notify all registered callbacks for a state."""
        for callback in self._callbacks.get(state, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(state)
                else:
                    callback(state)
            except Exception as e:
                logger.error(f"Callback error for {state.name}: {e}")

    def record_error(self, error_msg: str) -> None:
        """
        Record an error. Auto-transitions to ERROR state if threshold exceeded.
        """
        self._error_count += 1
        logger.error(f"Error #{self._error_count}: {error_msg}")
        
        if self._error_count >= self._max_errors_before_halt:
            logger.critical(
                f"Error threshold exceeded ({self._max_errors_before_halt}). "
                f"Auto-transitioning to ERROR state."
            )
            asyncio.create_task(
                self.transition(
                    BotState.ERROR,
                    reason=f"Error threshold exceeded: {error_msg}",
                    triggered_by="auto"
                )
            )

    def clear_error(self) -> bool:
        """
        Clear error state and reset to IDLE.
        Only works if current state is ERROR.
        """
        if self._current_state == BotState.ERROR:
            asyncio.create_task(
                self.transition(BotState.IDLE, reason="Error cleared by user", triggered_by="user")
            )
            return True
        return False

    def request_liquidation(self, reason: str = "User requested") -> bool:
        """Request transition to liquidation state."""
        if self._current_state == BotState.TRADING:
            asyncio.create_task(
                self.transition(BotState.LIQUIDATING, reason=reason, triggered_by="user")
            )
            return True
        logger.warning(f"Cannot liquidate from state: {self._current_state.name}")
        return False

    def request_halt(self, reason: str = "User requested") -> bool:
        """Request transition to halt state."""
        if self.can_transition_to(BotState.HALT):
            asyncio.create_task(
                self.transition(BotState.HALT, reason=reason, triggered_by="user")
            )
            return True
        return False

    def request_start(self) -> bool:
        """Request start of trading (IDLE -> WARMUP -> TRADING)."""
        if self._current_state == BotState.IDLE:
            asyncio.create_task(
                self.transition(BotState.WARMUP, reason="Starting system", triggered_by="user")
            )
            return True
        return False

    def get_status(self) -> dict:
        """Get comprehensive status report."""
        return {
            'current_state': self._current_state.name,
            'state_description': STATE_DESCRIPTIONS[self._current_state],
            'is_trading': self.is_trading,
            'is_halted': self.is_halted,
            'allowed_transitions': [s.name for s in self.get_allowed_transitions()],
            'error_count': self._error_count,
            'last_transition': self._last_transition_time.isoformat() if self._last_transition_time else None,
            'recent_transitions': [
                {
                    'from': t.from_state.name,
                    'to': t.to_state.name,
                    'reason': t.reason,
                    'timestamp': t.timestamp.isoformat()
                }
                for t in self._transition_history[-10:]
            ]
        }

    def get_transition_history(self, limit: int = 100) -> List[dict]:
        """Get recent transition history."""
        return [
            {
                'from': t.from_state.name,
                'to': t.to_state.name,
                'reason': t.reason,
                'triggered_by': t.triggered_by,
                'timestamp': t.timestamp.isoformat()
            }
            for t in self._transition_history[-limit:]
        ]
