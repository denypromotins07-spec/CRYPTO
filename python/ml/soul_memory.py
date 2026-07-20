# python/ml/soul_memory.py
# =============================================================================
# STAGE 2 - CHAPTER 3 - FILE 3
# Focus: SOUL.md deterministic memory system for self-learning.
# NO LLMs - uses structured templates, decision trees, and rule-based NLP.
# =============================================================================

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import re


class MemoryType(Enum):
    """Types of memories stored in SOUL.md."""
    TRADE_OUTCOME = "trade_outcome"
    STRATEGY_UPDATE = "strategy_update"
    MISTAKE_PATTERN = "mistake_pattern"
    MARKET_REGIME = "market_regime"
    PARAMETER_TUNING = "parameter_tuning"
    RISK_EVENT = "risk_event"


class DecisionOutcome(Enum):
    """Outcome classification for decisions."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    BREAKEVEN = "breakeven"


@dataclass
class TradeMemory:
    """Structured memory of a trade outcome."""
    timestamp: str
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_percent: float
    holding_period_ms: int
    outcome: str  # success, failure, etc.
    market_conditions: Dict[str, float]  # RSI, volatility, etc.
    decision_factors: List[str]  # What triggered the trade
    lessons_learned: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class StrategyUpdate:
    """Record of strategy parameter updates."""
    timestamp: str
    update_type: str
    previous_params: Dict[str, Any]
    new_params: Dict[str, Any]
    trigger_reason: str
    expected_improvement: str
    validation_status: str = "pending"  # pending, validated, rejected


@dataclass
class MistakePattern:
    """Identified pattern of trading mistakes."""
    timestamp: str
    pattern_id: str
    pattern_description: str
    occurrence_count: int
    avg_loss_per_occurrence: float
    conditions: Dict[str, Any]  # When this mistake occurs
    correction_rule: str  # Rule to avoid this mistake
    status: str = "active"  # active, resolved


class SOULMemoryEngine:
    """
    Deterministic memory engine for the trading bot's self-learning system.
    
    Since NO LLMs are allowed, this uses:
    1. Structured templates for consistent formatting
    2. Statistical decision trees for pattern recognition
    3. Rule-based NLP for generating human-readable insights
    4. Hash-based deduplication to prevent redundant entries
    
    The SOUL.md file serves as episodic memory that the bot can reference
    to improve future decisions.
    """
    
    # Template strings for consistent formatting
    TRADE_TEMPLATE = """
## Trade #{id} - {outcome}

**Timestamp:** {timestamp}  
**Symbol:** {symbol}  
**Direction:** {direction}  
**Entry:** ${entry_price:.2f} → **Exit:** ${exit_price:.2f}  
**PnL:** ${pnl:.2f} ({pnl_percent:+.2f}%)  
**Holding Period:** {holding_period_ms}ms

### Market Conditions
{market_conditions_str}

### Decision Factors
{decision_factors_str}

### Lessons Learned
{lessons_str}

---
"""

    MISTAKE_TEMPLATE = """
## Mistake Pattern #{pattern_id}

**First Detected:** {timestamp}  
**Occurrences:** {count}  
**Avg Loss:** ${avg_loss:.2f}

**Description:** {description}

**Triggering Conditions:**
{conditions_str}

**Correction Rule:**
> {correction_rule}

---
"""

    def __init__(self, soul_path: str = "SOUL.md", max_memories: int = 10000):
        """
        Initialize the SOUL memory engine.
        
        Args:
            soul_path: Path to the SOUL.md file
            max_memories: Maximum number of memories to retain (FIFO eviction)
        """
        self.soul_path = soul_path
        self.max_memories = max_memories
        
        # In-memory storage
        self.trade_memories: List[TradeMemory] = []
        self.strategy_updates: List[StrategyUpdate] = []
        self.mistake_patterns: Dict[str, MistakePattern] = {}
        self.market_regimes: List[Dict] = []
        
        # Statistics for pattern detection
        self._win_count = 0
        self._loss_count = 0
        self._total_pnl = 0.0
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Hash set for deduplication
        self._memory_hashes: set = set()
        
        # Load existing SOUL.md if exists
        self._load_soul_file()
    
    def _load_soul_file(self):
        """Load existing memories from SOUL.md file."""
        if not os.path.exists(self.soul_path):
            # Create initial SOUL.md with header
            self._initialize_soul_file()
            return
        
        try:
            with open(self.soul_path, 'r') as f:
                content = f.read()
            
            # Parse existing memories (simplified parsing)
            # In production, would use more robust markdown parser
            self._parse_trade_memories(content)
            self._parse_mistake_patterns(content)
            
        except Exception as e:
            print(f"[SOUL] Error loading SOUL.md: {e}")
            self._initialize_soul_file()
    
    def _initialize_soul_file(self):
        """Create new SOUL.md with header."""
        header = """# SOUL - Self-Organizing Understanding Layer

> This file contains the episodic memory of the trading bot.
> All learnings are derived from actual trading experience using
> statistical analysis and deterministic pattern recognition.

**Created:** {created}  
**Last Updated:** {updated}  
**Total Trades Analyzed:** {trades}  
**Win Rate:** {win_rate:.2%}

---

## Table of Contents
- [Trade History](#trade-history)
- [Mistake Patterns](#mistake-patterns)
- [Strategy Updates](#strategy-updates)
- [Market Regimes](#market-regimes)

---

## Trade History

""".format(
            created=datetime.utcnow().isoformat(),
            updated=datetime.utcnow().isoformat(),
            trades=0,
            win_rate=0.0
        )
        
        with open(self.soul_path, 'w') as f:
            f.write(header)
    
    def _compute_hash(self, data: Dict) -> str:
        """Compute hash for deduplication."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def record_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        position_size: float,
        pnl: float,
        holding_period_ms: int,
        market_conditions: Dict[str, float],
        decision_factors: List[str],
    ) -> Optional[str]:
        """
        Record a trade outcome to SOUL memory.
        
        Returns the lesson ID if recorded, None if duplicate.
        """
        with self._lock:
            # Calculate outcome metrics
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100
            if direction == "short":
                pnl_percent = -pnl_percent
            
            # Classify outcome
            if pnl > 0:
                outcome = DecisionOutcome.SUCCESS.value
                self._win_count += 1
            elif pnl < 0:
                outcome = DecisionOutcome.FAILURE.value
                self._loss_count += 1
            else:
                outcome = DecisionOutcome.BREAKEVEN.value
            
            self._total_pnl += pnl
            
            # Generate lessons learned (rule-based)
            lessons = self._generate_lessons(
                outcome, pnl_percent, market_conditions, decision_factors
            )
            
            # Create trade memory
            memory = TradeMemory(
                timestamp=datetime.utcnow().isoformat(),
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                position_size=position_size,
                pnl=pnl,
                pnl_percent=pnl_percent,
                holding_period_ms=holding_period_ms,
                outcome=outcome,
                market_conditions=market_conditions,
                decision_factors=decision_factors,
                lessons_learned=lessons,
            )
            
            # Check for duplicates
            memory_hash = self._compute_hash(memory.to_dict())
            if memory_hash in self._memory_hashes:
                return None  # Duplicate, skip
            
            self._memory_hashes.add(memory_hash)
            
            # Add to memory list
            self.trade_memories.append(memory)
            
            # Evict old memories if exceeding limit
            if len(self.trade_memories) > self.max_memories:
                self.trade_memories = self.trade_memories[-self.max_memories:]
            
            # Append to SOUL.md
            self._append_trade_to_soul(memory)
            
            # Check for mistake patterns
            self._detect_mistake_patterns(memory)
            
            # Update statistics
            self._update_soul_stats()
            
            return f"TRADE_{len(self.trade_memories)}"
    
    def _generate_lessons(
        self,
        outcome: str,
        pnl_percent: float,
        market_conditions: Dict[str, float],
        decision_factors: List[str],
    ) -> List[str]:
        """
        Generate lessons learned using rule-based logic.
        NO LLM - uses predefined templates and thresholds.
        """
        lessons = []
        
        # Rule 1: Large losses
        if pnl_percent < -5.0:
            lessons.append(
                f"Large loss ({pnl_percent:.1f}%) - Review entry timing and stop-loss placement."
            )
        
        # Rule 2: Quick reversals (held too short)
        if outcome == DecisionOutcome.FAILURE.value:
            rsi = market_conditions.get('rsi', 50)
            if rsi > 70 or rsi < 30:
                lessons.append(
                    f"Trade entered at extreme RSI ({rsi:.1f}) - Consider waiting for mean reversion."
                )
        
        # Rule 3: Successful trades
        if outcome == DecisionOutcome.SUCCESS.value and pnl_percent > 3.0:
            lessons.append(
                f"Strong performance ({pnl_percent:.1f}%) - Analyze conditions for replication."
            )
        
        # Rule 4: High volatility
        volatility = market_conditions.get('volatility', 0)
        if volatility > 0.05:  # 5% volatility
            lessons.append(
                "High volatility environment - Position sizing was critical."
            )
        
        # Rule 5: Decision factor analysis
        if 'fomo' in [f.lower() for f in decision_factors]:
            lessons.append(
                "FOMO detected as factor - Implement cooling-off period after losses."
            )
        
        if not lessons:
            lessons.append("No significant patterns detected - Continue monitoring.")
        
        return lessons
    
    def _detect_mistake_patterns(self, trade: TradeMemory):
        """
        Detect recurring mistake patterns using statistical analysis.
        """
        if trade.outcome != DecisionOutcome.FAILURE.value:
            return
        
        # Look for similar failed trades
        similar_failures = [
            t for t in self.trade_memories[-100:]  # Last 100 trades
            if t.outcome == DecisionOutcome.FAILURE.value
            and t.symbol == trade.symbol
            and abs(t.pnl_percent - trade.pnl_percent) < 2.0  # Similar loss magnitude
        ]
        
        if len(similar_failures) >= 3:
            # Pattern detected
            pattern_id = f"MISTAKE_{hashlib.md5(trade.symbol.encode()).hexdigest()[:8]}"
            
            if pattern_id not in self.mistake_patterns:
                # Create new pattern
                avg_loss = sum(t.pnl for t in similar_failures) / len(similar_failures)
                
                pattern = MistakePattern(
                    timestamp=datetime.utcnow().isoformat(),
                    pattern_id=pattern_id,
                    pattern_description=f"Recurring losses on {trade.symbol} under similar conditions",
                    occurrence_count=len(similar_failures),
                    avg_loss_per_occurrence=avg_loss,
                    conditions={
                        'symbol': trade.symbol,
                        'avg_loss_pct': sum(t.pnl_percent for t in similar_failures) / len(similar_failures),
                    },
                    correction_rule=f"Reduce position size on {trade.symbol} by 50% until pattern is broken.",
                )
                
                self.mistake_patterns[pattern_id] = pattern
                self._append_mistake_to_soul(pattern)
            else:
                # Update existing pattern
                self.mistake_patterns[pattern_id].occurrence_count += 1
    
    def _append_trade_to_soul(self, memory: TradeMemory):
        """Append a trade memory to SOUL.md file."""
        trade_entry = self.TRADE_TEMPLATE.format(
            id=len(self.trade_memories),
            outcome=memory.outcome.upper(),
            timestamp=memory.timestamp,
            symbol=memory.symbol,
            direction=memory.direction,
            entry_price=memory.entry_price,
            exit_price=memory.exit_price,
            pnl=memory.pnl,
            pnl_percent=memory.pnl_percent,
            holding_period_ms=memory.holding_period_ms,
            market_conditions_str=self._format_dict(memory.market_conditions),
            decision_factors_str='\n'.join(f"- {f}" for f in memory.decision_factors),
            lessons_str='\n'.join(f"- {l}" for l in memory.lessons_learned),
        )
        
        with open(self.soul_path, 'a') as f:
            f.write(trade_entry)
    
    def _append_mistake_to_soul(self, pattern: MistakePattern):
        """Append a mistake pattern to SOUL.md file."""
        # Ensure Mistake Patterns section exists
        self._ensure_section_exists("Mistake Patterns")
        
        mistake_entry = self.MISTAKE_TEMPLATE.format(
            pattern_id=pattern.pattern_id,
            timestamp=pattern.timestamp,
            count=pattern.occurrence_count,
            avg_loss=pattern.avg_loss_per_occurrence,
            description=pattern.pattern_description,
            conditions_str=self._format_dict(pattern.conditions),
            correction_rule=pattern.correction_rule,
        )
        
        with open(self.soul_path, 'a') as f:
            f.write(mistake_entry)
    
    def _ensure_section_exists(self, section_name: str):
        """Ensure a section header exists in SOUL.md."""
        with open(self.soul_path, 'r') as f:
            content = f.read()
        
        if f"## {section_name}" not in content:
            with open(self.soul_path, 'a') as f:
                f.write(f"\n## {section_name}\n\n---\n\n")
    
    def _format_dict(self, d: Dict) -> str:
        """Format dictionary as markdown list."""
        return '\n'.join(f"- **{k}:** {v:.4f}" if isinstance(v, float) else f"- **{k}:** {v}" 
                        for k, v in d.items())
    
    def _update_soul_stats(self):
        """Update statistics header in SOUL.md."""
        total_trades = self._win_count + self._loss_count
        win_rate = self._win_count / total_trades if total_trades > 0 else 0.0
        
        # Read current file
        with open(self.soul_path, 'r') as f:
            content = f.read()
        
        # Update stats in header (simple regex replacement)
        content = re.sub(
            r'\*\*Total Trades Analyzed:\*\* \d+',
            f'**Total Trades Analyzed:** {total_trades}',
            content
        )
        content = re.sub(
            r'\*\*Win Rate:\*\* [\d.]+%',
            f'**Win Rate:** {win_rate:.2%}',
            content
        )
        content = re.sub(
            r'\*\*Last Updated:\*\* .+',
            f'**Last Updated:** {datetime.utcnow().isoformat()}',
            content
        )
        
        # Write back
        with open(self.soul_path, 'w') as f:
            f.write(content)
    
    def _parse_trade_memories(self, content: str):
        """Parse trade memories from SOUL.md content."""
        # Simplified parsing - in production would be more robust
        pass
    
    def _parse_mistake_patterns(self, content: str):
        """Parse mistake patterns from SOUL.md content."""
        pass
    
    def get_similar_trades(
        self,
        symbol: str,
        market_conditions: Dict[str, float],
        max_results: int = 10
    ) -> List[TradeMemory]:
        """
        Find historically similar trades for decision support.
        Uses simple distance metric on market conditions.
        """
        if not self.trade_memories:
            return []
        
        def distance(t: TradeMemory) -> float:
            """Calculate distance between current and historical conditions."""
            dist = 0.0
            for key, value in market_conditions.items():
                if key in t.market_conditions:
                    dist += abs(t.market_conditions[key] - value)
            return dist
        
        # Filter by symbol and sort by similarity
        similar = [t for t in self.trade_memories if t.symbol == symbol]
        similar.sort(key=distance)
        
        return similar[:max_results]
    
    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics from all memories."""
        total_trades = len(self.trade_memories)
        if total_trades == 0:
            return {'error': 'No trades recorded'}
        
        wins = [t for t in self.trade_memories if t.outcome == DecisionOutcome.SUCCESS.value]
        losses = [t for t in self.trade_memories if t.outcome == DecisionOutcome.FAILURE.value]
        
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
        
        return {
            'total_trades': total_trades,
            'win_count': len(wins),
            'loss_count': len(losses),
            'win_rate': len(wins) / total_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
            'total_pnl': sum(t.pnl for t in self.trade_memories),
            'active_mistake_patterns': len([p for p in self.mistake_patterns.values() if p.status == 'active']),
        }


# Singleton instance for global access
_soul_instance: Optional[SOULMemoryEngine] = None
_soul_lock = threading.Lock()


def get_soul_memory(soul_path: str = "SOUL.md") -> SOULMemoryEngine:
    """Get or create the singleton SOUL memory engine."""
    global _soul_instance
    
    with _soul_lock:
        if _soul_instance is None:
            _soul_instance = SOULMemoryEngine(soul_path)
        return _soul_instance


# Example usage
if __name__ == "__main__":
    # Initialize SOUL
    soul = get_soul_memory()
    
    # Record a sample trade
    trade_id = soul.record_trade(
        symbol="BTCUSDT",
        direction="long",
        entry_price=50000.0,
        exit_price=51500.0,
        position_size=0.1,
        pnl=150.0,
        holding_period_ms=3600000,
        market_conditions={'rsi': 55.0, 'volatility': 0.02},
        decision_factors=['macd_crossover', 'volume_spike'],
    )
    
    print(f"Recorded trade: {trade_id}")
    print(f"Stats: {soul.get_aggregate_stats()}")
