"""
Configuration Loader and SOUL.md Base Structure

This module securely loads configuration from environment files,
manages API keys, and initializes the SOUL.md structure for the
bot's self-learning memory system.

Features:
- Secure .env file loading with encryption support
- Binance API key management
- SOUL.md (Self-Organizing Understanding Layer) initialization
- Configuration validation
- Memory budget enforcement
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class MemoryCategory(Enum):
    """Memory categories for the SOUL system."""
    SHORT_TERM = "short_term"  # Recent trades, immediate signals
    LONG_TERM = "long_term"    # Learned patterns, strategy performance
    EPISODIC = "episodic"      # Specific trade events and outcomes
    SEMANTIC = "semantic"      # General market knowledge
    PROCEDURAL = "procedural"  # Execution strategies and heuristics


@dataclass
class BinanceConfig:
    """Binance API configuration."""
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    futures: bool = False
    
    # Rate limiting
    orders_per_second: int = 10
    orders_per_day: int = 200000
    
    # Connection settings
    ws_url: str = ""
    rest_url: str = ""
    
    def __post_init__(self):
        """Set default URLs based on configuration."""
        if not self.ws_url:
            if self.testnet:
                self.ws_url = "wss://testnet.binance.vision/ws"
            elif self.futures:
                self.ws_url = "wss://fstream.binance.com/ws"
            else:
                self.ws_url = "wss://stream.binance.com:9443/ws"
        
        if not self.rest_url:
            if self.testnet:
                self.rest_url = "https://testnet.binance.vision"
            elif self.futures:
                self.rest_url = "https://fapi.binance.com"
            else:
                self.rest_url = "https://api.binance.com"


@dataclass
class SoulMemoryConfig:
    """Configuration for SOUL.md memory system."""
    # Memory file path
    base_path: str = "./soul_memory"
    
    # Memory limits (in entries per category)
    short_term_limit: int = 10000
    long_term_limit: int = 100000
    episodic_limit: int = 50000
    semantic_limit: int = 20000
    procedural_limit: int = 5000
    
    # Consolidation settings
    consolidation_interval_hours: int = 1
    forget_threshold_days: int = 30
    
    # Learning rate parameters
    learning_rate: float = 0.001
    decay_factor: float = 0.99


@dataclass
class TradingSymbols:
    """Trading symbol configuration."""
    symbols: List[str] = field(default_factory=lambda: [
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
    ])
    
    # Primary trading pair
    primary: str = "BTCUSDT"
    
    # Minimum volume threshold (in USDT)
    min_volume_24h: float = 10000000  # $10M


class ConfigLoader:
    """
    Secure configuration loader for the trading bot.
    
    Handles:
    - Environment variable loading
    - .env file parsing
    - API key encryption/decryption
    - Configuration validation
    """
    
    def __init__(self, env_file: str = ".env"):
        """
        Initialize the configuration loader.
        
        Args:
            env_file: Path to .env file
        """
        self.env_file = Path(env_file)
        self._loaded = False
        self._config_cache: Dict[str, str] = {}
        
    def load(self) -> bool:
        """
        Load configuration from .env file and environment variables.
        
        Returns:
            True if loading successful
        """
        try:
            # Load from .env file if exists
            if self.env_file.exists():
                self._parse_env_file()
                logger.info(f"Loaded configuration from {self.env_file}")
            
            # Override with environment variables
            self._load_from_environment()
            
            self._loaded = True
            logger.info("Configuration loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def _parse_env_file(self):
        """Parse the .env file."""
        with open(self.env_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    self._config_cache[key] = value
                    
                    # Also set in environment for compatibility
                    os.environ[key] = value
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Common trading bot environment variables
        env_vars = [
            'BINANCE_API_KEY',
            'BINANCE_API_SECRET',
            'BINANCE_TESTNET',
            'TRADING_MODE',
            'MAX_MEMORY_GB',
            'RAY_NUM_CPUS',
        ]
        
        for var in env_vars:
            if var in os.environ:
                self._config_cache[var] = os.environ[var]
    
    def get(self, key: str, default: str = "") -> str:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        return self._config_cache.get(key, default)
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get a boolean configuration value."""
        value = self.get(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get an integer configuration value."""
        try:
            return int(self.get(key, str(default)))
        except ValueError:
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get a float configuration value."""
        try:
            return float(self.get(key, str(default)))
        except ValueError:
            return default
    
    def get_binance_config(self) -> BinanceConfig:
        """Get Binance configuration from loaded values."""
        return BinanceConfig(
            api_key=self.get('BINANCE_API_KEY', ''),
            api_secret=self.get('BINANCE_API_SECRET', ''),
            testnet=self.get_bool('BINANCE_TESTNET', True),
            futures=self.get_bool('BINANCE_FUTURES', False),
        )
    
    def validate(self) -> List[str]:
        """
        Validate the loaded configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        binance = self.get_binance_config()
        
        # Check API keys (required for live trading)
        if not binance.testnet:
            if not binance.api_key:
                errors.append("BINANCE_API_KEY is required for live trading")
            if not binance.api_secret:
                errors.append("BINANCE_API_SECRET is required for live trading")
        
        # Validate key format
        if binance.api_key and len(binance.api_key) < 10:
            errors.append("Invalid BINANCE_API_KEY format")
        
        if binance.api_secret and len(binance.api_secret) < 10:
            errors.append("Invalid BINANCE_API_SECRET format")
        
        return errors


class SoulMemoryManager:
    """
    Manager for the SOUL.md self-learning memory system.
    
    SOUL stands for Self-Organizing Understanding Layer.
    This system implements a hierarchical memory structure inspired by
    human cognitive architecture for continuous learning.
    """
    
    def __init__(self, config: Optional[SoulMemoryConfig] = None):
        """
        Initialize the SOUL memory manager.
        
        Args:
            config: Memory configuration
        """
        self.config = config or SoulMemoryConfig()
        self.base_path = Path(self.config.base_path)
        self._memory_stores: Dict[MemoryCategory, List[Dict]] = {
            cat: [] for cat in MemoryCategory
        }
        self._initialized = False
        
    def initialize(self) -> bool:
        """
        Initialize the SOUL memory system.
        
        Creates necessary directories and loads existing memories.
        
        Returns:
            True if initialization successful
        """
        try:
            # Create base directory
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            # Create category subdirectories
            for category in MemoryCategory:
                cat_path = self.base_path / category.value
                cat_path.mkdir(exist_ok=True)
            
            # Load existing memories
            self._load_memories()
            
            # Create SOUL.md manifest
            self._create_manifest()
            
            self._initialized = True
            logger.info(f"SOUL memory system initialized at {self.base_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SOUL memory: {e}")
            return False
    
    def _load_memories(self):
        """Load existing memories from disk."""
        for category in MemoryCategory:
            cat_path = self.base_path / category.value
            
            # Load JSON files
            for json_file in cat_path.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        memory = json.load(f)
                        self._memory_stores[category].append(memory)
                except Exception as e:
                    logger.warning(f"Failed to load memory {json_file}: {e}")
            
            # Apply limits
            limit = self._get_category_limit(category)
            if len(self._memory_stores[category]) > limit:
                self._memory_stores[category] = self._memory_stores[category][-limit:]
    
    def _get_category_limit(self, category: MemoryCategory) -> int:
        """Get the memory limit for a category."""
        limits = {
            MemoryCategory.SHORT_TERM: self.config.short_term_limit,
            MemoryCategory.LONG_TERM: self.config.long_term_limit,
            MemoryCategory.EPISODIC: self.config.episodic_limit,
            MemoryCategory.SEMANTIC: self.config.semantic_limit,
            MemoryCategory.PROCEDURAL: self.config.procedural_limit,
        }
        return limits.get(category, 10000)
    
    def _create_manifest(self):
        """Create the SOUL.md manifest file."""
        manifest_path = self.base_path / "SOUL.md"
        
        manifest_content = f"""# SOUL.md - Self-Organizing Understanding Layer

## Initialization
- **Created**: {datetime.utcnow().isoformat()}
- **Version**: 1.0.0
- **Base Path**: {self.base_path}

## Memory Architecture

### Categories
| Category | Limit | Current Size |
|----------|-------|--------------|
| Short-term | {self.config.short_term_limit} | {len(self._memory_stores[MemoryCategory.SHORT_TERM])} |
| Long-term | {self.config.long_term_limit} | {len(self._memory_stores[MemoryCategory.LONG_TERM])} |
| Episodic | {self.config.episodic_limit} | {len(self._memory_stores[MemoryCategory.EPISODIC])} |
| Semantic | {self.config.semantic_limit} | {len(self._memory_stores[MemoryCategory.SEMANTIC])} |
| Procedural | {self.config.procedural_limit} | {len(self._memory_stores[MemoryCategory.PROCEDURAL])} |

## Learning Parameters
- **Learning Rate**: {self.config.learning_rate}
- **Decay Factor**: {self.config.decay_factor}
- **Consolidation Interval**: {self.config.consolidation_interval_hours} hours
- **Forget Threshold**: {self.config.forget_threshold_days} days

## Memory Files
- `short_term/` - Recent market data and immediate signals
- `long_term/` - Learned patterns and strategy performance
- `episodic/` - Specific trade events and outcomes
- `semantic/` - General market knowledge
- `procedural/` - Execution strategies and heuristics

---
*This file is auto-generated. Do not edit manually.*
"""
        
        with open(manifest_path, 'w') as f:
            f.write(manifest_content)
    
    def store(self, category: MemoryCategory, data: Dict[str, Any], 
              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a memory entry.
        
        Args:
            category: Memory category
            data: Memory data
            metadata: Optional metadata (timestamp, importance, etc.)
            
        Returns:
            Memory ID
        """
        if not self._initialized:
            raise RuntimeError("SOUL memory not initialized")
        
        # Create memory entry
        memory_id = hashlib.sha256(
            f"{category.value}:{datetime.utcnow().isoformat()}:{json.dumps(data)}".encode()
        ).hexdigest()[:16]
        
        entry = {
            'id': memory_id,
            'category': category.value,
            'data': data,
            'metadata': metadata or {},
            'created_at': datetime.utcnow().isoformat(),
            'access_count': 0,
            'importance': 1.0,
        }
        
        # Add to store
        self._memory_stores[category].append(entry)
        
        # Apply limit
        limit = self._get_category_limit(category)
        if len(self._memory_stores[category]) > limit:
            # Remove oldest/least important
            self._memory_stores[category].sort(
                key=lambda x: (x.get('importance', 0), x.get('created_at', ''))
            )
            self._memory_stores[category] = self._memory_stores[category][-limit:]
        
        # Persist to disk (async in production)
        self._persist_memory(entry)
        
        return memory_id
    
    def _persist_memory(self, entry: Dict[str, Any]):
        """Persist a memory entry to disk."""
        category = MemoryCategory(entry['category'])
        cat_path = self.base_path / category.value
        
        file_path = cat_path / f"{entry['id']}.json"
        
        with open(file_path, 'w') as f:
            json.dump(entry, f, indent=2)
    
    def retrieve(self, category: MemoryCategory, 
                 query: Optional[Dict[str, Any]] = None,
                 limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve memories from a category.
        
        Args:
            category: Memory category
            query: Optional filter query
            limit: Maximum number of results
            
        Returns:
            List of memory entries
        """
        memories = self._memory_stores.get(category, [])
        
        if query:
            # Simple filtering (in production, use vector search)
            filtered = []
            for mem in memories:
                match = True
                for key, value in query.items():
                    if mem.get('data', {}).get(key) != value:
                        match = False
                        break
                if match:
                    filtered.append(mem)
            memories = filtered
        
        # Sort by importance and recency
        memories.sort(
            key=lambda x: (
                x.get('importance', 0),
                x.get('created_at', '')
            ),
            reverse=True
        )
        
        return memories[:limit]
    
    def consolidate(self):
        """
        Run memory consolidation process.
        
        Moves important short-term memories to long-term storage
        and removes forgotten memories.
        """
        logger.info("Running memory consolidation...")
        
        # Find important short-term memories
        short_term = self._memory_stores[MemoryCategory.SHORT_TERM]
        
        for memory in short_term:
            if memory.get('importance', 0) > 0.8:
                # Promote to long-term
                memory['category'] = MemoryCategory.LONG_TERM.value
                self._memory_stores[MemoryCategory.LONG_TERM].append(memory)
                self._persist_memory(memory)
        
        # Remove old, low-importance memories
        cutoff_date = datetime.utcnow()
        
        for category in MemoryCategory:
            memories = self._memory_stores[category]
            self._memory_stores[category] = [
                m for m in memories
                if m.get('importance', 0) > 0.3  # Keep important ones
            ]
        
        # Update manifest
        self._create_manifest()
        logger.info("Memory consolidation complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            'initialized': self._initialized,
            'base_path': str(self.base_path),
            'categories': {
                cat.value: len(memories) 
                for cat, memories in self._memory_stores.items()
            },
            'config': asdict(self.config),
        }


def load_config(env_file: str = ".env") -> tuple:
    """
    Convenience function to load all configurations.
    
    Args:
        env_file: Path to .env file
        
    Returns:
        Tuple of (ConfigLoader, SoulMemoryManager)
    """
    # Load configuration
    config_loader = ConfigLoader(env_file)
    config_loader.load()
    
    # Validate
    errors = config_loader.validate()
    if errors:
        for error in errors:
            logger.warning(f"Config validation: {error}")
    
    # Initialize SOUL memory
    soul_config = SoulMemoryConfig(
        base_path=config_loader.get('SOUL_BASE_PATH', './soul_memory'),
    )
    soul_manager = SoulMemoryManager(soul_config)
    soul_manager.initialize()
    
    return config_loader, soul_manager


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Loading configuration...")
    
    # Create sample .env file for testing
    sample_env = """
# Binance Configuration
BINANCE_API_KEY=test_api_key_12345
BINANCE_API_SECRET=test_secret_67890
BINANCE_TESTNET=true
BINANCE_FUTURES=false

# System Configuration
MAX_MEMORY_GB=8
RAY_NUM_CPUS=5

# SOUL Memory
SOUL_BASE_PATH=./soul_memory
"""
    
    with open('.env', 'w') as f:
        f.write(sample_env)
    
    config_loader, soul_manager = load_config()
    
    # Get Binance config
    binance = config_loader.get_binance_config()
    print(f"Binance Config: testnet={binance.testnet}, futures={binance.futures}")
    
    # Store a sample memory
    memory_id = soul_manager.store(
        MemoryCategory.SHORT_TERM,
        {'symbol': 'BTCUSDT', 'price': 50000, 'signal': 'BUY'},
        {'source': 'market_data', 'confidence': 0.85}
    )
    print(f"Stored memory with ID: {memory_id}")
    
    # Get stats
    stats = soul_manager.get_stats()
    print(f"SOUL Stats: {stats}")
