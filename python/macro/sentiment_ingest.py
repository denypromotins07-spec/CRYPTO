"""
Macro Economic and Sentiment Data Ingestion Pipeline
Ingests macroeconomic indicators (CPI, PPI, Fed rates, DXY, Bond yields)
and statistical sentiment proxies using keyword frequency and lexicon scoring
NO LLMs - purely statistical and rule-based methods
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any
from collections import deque, defaultdict
from dataclasses import dataclass, field
import numpy as np
import time
import re
from datetime import datetime, timedelta


@dataclass
class MacroIndicator:
    """Macroeconomic indicator data point."""
    name: str
    value: float
    previous_value: float
    forecast_value: Optional[float]
    timestamp: float
    release_date: datetime
    impact_score: float  # -1.0 to 1.0 (negative to positive for markets)
    surprise: float  # Actual vs Forecast difference


@dataclass
class SentimentScore:
    """Sentiment analysis result."""
    source: str
    timestamp: float
    bullish_score: float  # 0.0 to 1.0
    bearish_score: float  # 0.0 to 1.0
    neutral_score: float  # 0.0 to 1.0
    compound_score: float  # -1.0 to 1.0
    keyword_counts: Dict[str, int] = field(default_factory=dict)
    volume: int = 0


class LexiconBasedSentiment:
    """
    Lexicon-based sentiment analyzer without LLMs.
    Uses predefined word lists and statistical scoring.
    """
    
    # Bullish keywords with weights
    BULLISH_LEXICON = {
        'bull': 0.8, 'buy': 0.7, 'long': 0.6, 'moon': 0.9, 'rocket': 0.9,
        'surge': 0.7, 'rally': 0.8, 'breakout': 0.7, 'pump': 0.6, 'gain': 0.5,
        'profit': 0.6, 'green': 0.5, 'up': 0.4, 'higher': 0.5, 'rise': 0.5,
        'strong': 0.4, 'positive': 0.6, 'optimistic': 0.7, 'opportunity': 0.5,
        'accumulation': 0.6, 'support': 0.4, 'bounce': 0.5, 'recovery': 0.6,
        'ath': 0.8, 'alltimehigh': 0.8, 'fomo': 0.7, 'undervalued': 0.6,
        'oversold': 0.5, 'dip': 0.3, 'bottom': 0.4, 'reversal': 0.5
    }
    
    # Bearish keywords with weights
    BEARISH_LEXICON = {
        'bear': 0.8, 'sell': 0.7, 'short': 0.6, 'crash': 0.9, 'dump': 0.8,
        'plunge': 0.8, 'drop': 0.6, 'fall': 0.6, 'loss': 0.7, 'red': 0.5,
        'down': 0.4, 'lower': 0.5, 'decline': 0.6, 'weak': 0.5, 'negative': 0.6,
        'pessimistic': 0.7, 'risk': 0.4, 'warning': 0.5, 'danger': 0.7,
        'distribution': 0.5, 'resistance': 0.3, 'reject': 0.5, 'breakdown': 0.7,
        'overbought': 0.5, 'correction': 0.4, 'top': 0.4, 'bubble': 0.7,
        'panic': 0.8, 'capitulation': 0.8, 'liquidation': 0.7, 'rekt': 0.8
    }
    
    # Crypto-specific terms
    CRYPTO_TERMS = {
        'defi', 'nft', 'dao', 'layer2', 'scaling', 'staking', 'yield',
        'farm', 'mine', 'hashrate', 'whale', 'altcoin', 'btc', 'eth'
    }
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.text_buffer = deque(maxlen=window_size)
        
        # Compile regex patterns for efficiency
        self.word_pattern = re.compile(r'\b\w+\b')
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words."""
        return self.word_pattern.findall(text.lower())
    
    def analyze(self, text: str, source: str = 'unknown') -> SentimentScore:
        """
        Analyze sentiment of text using lexicon matching.
        Returns SentimentScore with detailed breakdown.
        """
        tokens = self._tokenize(text)
        self.text_buffer.append(tokens)
        
        bullish_total = 0.0
        bearish_total = 0.0
        keyword_counts = defaultdict(int)
        
        for token in tokens:
            if token in self.BULLISH_LEXICON:
                weight = self.BULLISH_LEXICON[token]
                bullish_total += weight
                keyword_counts[f"bull_{token}"] += 1
            
            if token in self.BEARISH_LEXICON:
                weight = self.BEARISH_LEXICON[token]
                bearish_total += weight
                keyword_counts[f"bear_{token}"] += 1
        
        total_words = len(tokens)
        total_keywords = bullish_total + bearish_total
        
        # Calculate scores
        if total_keywords == 0:
            bullish_score = 0.33
            bearish_score = 0.33
            neutral_score = 0.34
            compound_score = 0.0
        else:
            bullish_score = min(1.0, bullish_total / (total_words * 0.1))
            bearish_score = min(1.0, bearish_total / (total_words * 0.1))
            neutral_score = max(0.0, 1.0 - bullish_score - bearish_score)
            compound_score = (bullish_total - bearish_total) / total_keywords
        
        return SentimentScore(
            source=source,
            timestamp=time.time(),
            bullish_score=bullish_score,
            bearish_score=bearish_score,
            neutral_score=neutral_score,
            compound_score=np.clip(compound_score, -1.0, 1.0),
            keyword_counts=dict(keyword_counts),
            volume=total_words
        )
    
    def get_aggregate_sentiment(self, hours: int = 24) -> Dict[str, float]:
        """Get aggregate sentiment over recent period."""
        if not self.text_buffer:
            return {'bullish': 0.33, 'bearish': 0.33, 'neutral': 0.34, 'compound': 0.0}
        
        all_tokens = [token for tokens in self.text_buffer for token in tokens]
        
        bullish_count = sum(1 for t in all_tokens if t in self.BULLISH_LEXICON)
        bearish_count = sum(1 for t in all_tokens if t in self.BEARISH_LEXICON)
        total = len(all_tokens)
        
        if total == 0:
            return {'bullish': 0.33, 'bearish': 0.33, 'neutral': 0.34, 'compound': 0.0}
        
        bullish_ratio = bullish_count / total
        bearish_ratio = bearish_count / total
        neutral_ratio = 1.0 - bullish_ratio - bearish_ratio
        
        compound = (bullish_count - bearish_count) / (bullish_count + bearish_count + 1)
        
        return {
            'bullish': bullish_ratio,
            'bearish': bearish_ratio,
            'neutral': neutral_ratio,
            'compound': np.clip(compound, -1.0, 1.0),
            'total_words': total
        }


class MacroEconomicTracker:
    """
    Tracks macroeconomic indicators from various sources.
    Simulates data ingestion (in production would use actual APIs).
    """
    
    # Indicator metadata with typical ranges and impacts
    INDICATORS = {
        'cpi': {
            'name': 'Consumer Price Index',
            'frequency': 'monthly',
            'typical_range': (2.0, 4.0),
            'market_impact': 0.9  # High impact
        },
        'ppi': {
            'name': 'Producer Price Index',
            'frequency': 'monthly',
            'typical_range': (1.0, 5.0),
            'market_impact': 0.7
        },
        'fed_rate': {
            'name': 'Federal Funds Rate',
            'frequency': 'irregular',
            'typical_range': (0.0, 6.0),
            'market_impact': 1.0  # Maximum impact
        },
        'dxy': {
            'name': 'US Dollar Index',
            'frequency': 'daily',
            'typical_range': (95.0, 110.0),
            'market_impact': 0.6
        },
        'treasury_10y': {
            'name': '10-Year Treasury Yield',
            'frequency': 'daily',
            'typical_range': (2.0, 5.0),
            'market_impact': 0.8
        },
        'unemployment': {
            'name': 'Unemployment Rate',
            'frequency': 'monthly',
            'typical_range': (3.0, 6.0),
            'market_impact': 0.7
        },
        'gdp_growth': {
            'name': 'GDP Growth Rate',
            'frequency': 'quarterly',
            'typical_range': (0.0, 4.0),
            'market_impact': 0.8
        },
        'ism_manufacturing': {
            'name': 'ISM Manufacturing PMI',
            'frequency': 'monthly',
            'typical_range': (45.0, 60.0),
            'market_impact': 0.5
        }
    }
    
    def __init__(self):
        self.indicators: Dict[str, List[MacroIndicator]] = defaultdict(list)
        self.latest_values: Dict[str, MacroIndicator] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Initialize with some historical data
        self._initialize_historical_data()
    
    def _initialize_historical_data(self):
        """Initialize with simulated historical data."""
        base_time = time.time()
        
        # CPI data (monthly)
        for i in range(12):
            timestamp = base_time - (i * 30 * 24 * 3600)
            value = 3.0 + np.random.uniform(-0.5, 0.5)
            previous = value + np.random.uniform(-0.3, 0.3)
            
            indicator = MacroIndicator(
                name='cpi',
                value=value,
                previous_value=previous,
                forecast_value=previous,
                timestamp=timestamp,
                release_date=datetime.fromtimestamp(timestamp),
                impact_score=self._calculate_impact('cpi', value, previous),
                surprise=value - previous
            )
            self.indicators['cpi'].append(indicator)
        
        # DXY data (daily)
        for i in range(30):
            timestamp = base_time - (i * 24 * 3600)
            value = 102.0 + np.random.uniform(-3, 3)
            previous = value + np.random.uniform(-1, 1)
            
            indicator = MacroIndicator(
                name='dxy',
                value=value,
                previous_value=previous,
                forecast_value=None,
                timestamp=timestamp,
                release_date=datetime.fromtimestamp(timestamp),
                impact_score=self._calculate_impact('dxy', value, previous),
                surprise=value - previous
            )
            self.indicators['dxy'].append(indicator)
        
        # Update latest values
        for name, indicators in self.indicators.items():
            if indicators:
                self.latest_values[name] = indicators[0]
    
    def _calculate_impact(self, indicator_name: str, value: float, previous: float) -> float:
        """
        Calculate market impact score based on indicator surprise.
        Positive = good for risk assets, Negative = bad for risk assets
        """
        surprise = value - previous
        typical_range = self.INDICATORS.get(indicator_name, {}).get('typical_range', (0, 10))
        range_size = typical_range[1] - typical_range[0]
        
        # Normalize surprise
        normalized_surprise = surprise / range_size
        
        # Different indicators have different interpretations
        if indicator_name in ['cpi', 'ppi', 'fed_rate']:
            # Higher inflation/rates = negative for crypto
            impact = -normalized_surprise
        elif indicator_name in ['gdp_growth', 'ism_manufacturing']:
            # Higher growth = mixed (good economy but less stimulus)
            impact = normalized_surprise * 0.5
        elif indicator_name == 'dxy':
            # Stronger dollar = negative for crypto
            impact = -normalized_surprise
        elif indicator_name == 'unemployment':
            # Higher unemployment = mixed
            impact = -normalized_surprise * 0.3
        else:
            impact = normalized_surprise * 0.5
        
        return np.clip(impact, -1.0, 1.0)
    
    async def fetch_indicator(self, indicator_name: str) -> Optional[MacroIndicator]:
        """
        Fetch latest indicator value.
        In production, this would call economic data APIs (FRED, TradingEconomics).
        """
        try:
            if indicator_name not in self.INDICATORS:
                return None
            
            # Get previous value
            prev_indicator = self.latest_values.get(indicator_name)
            previous_value = prev_indicator.value if prev_indicator else 3.0
            
            # Simulate new data with some randomness
            typical_range = self.INDICATORS[indicator_name]['typical_range']
            mid_range = (typical_range[0] + typical_range[1]) / 2
            range_spread = (typical_range[1] - typical_range[0]) * 0.1
            
            new_value = mid_range + np.random.uniform(-range_spread, range_spread)
            forecast = previous_value + np.random.uniform(-0.2, 0.2)
            
            indicator = MacroIndicator(
                name=indicator_name,
                value=new_value,
                previous_value=previous_value,
                forecast_value=forecast,
                timestamp=time.time(),
                release_date=datetime.now(),
                impact_score=self._calculate_impact(indicator_name, new_value, previous_value),
                surprise=new_value - (forecast or previous_value)
            )
            
            self.indicators[indicator_name].insert(0, indicator)
            self.indicators[indicator_name] = self.indicators[indicator_name][:100]  # Keep last 100
            self.latest_values[indicator_name] = indicator
            
            return indicator
            
        except Exception as e:
            print(f"Error fetching {indicator_name}: {e}")
            return None
    
    def get_macro_feature_vector(self) -> np.ndarray:
        """
        Create feature vector from macro indicators for ML models.
        """
        features = []
        
        for indicator_name in self.INDICATORS.keys():
            if indicator_name in self.latest_values:
                ind = self.latest_values[indicator_name]
                features.extend([
                    ind.value,
                    ind.previous_value,
                    ind.surprise,
                    ind.impact_score,
                    (ind.value - ind.previous_value) / (ind.previous_value + 1e-6)  # Percent change
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def get_macro_summary(self) -> Dict[str, Any]:
        """Get summary of macro indicators."""
        return {
            name: {
                'value': ind.value,
                'previous': ind.previous_value,
                'surprise': ind.surprise,
                'impact': ind.impact_score
            }
            for name, ind in self.latest_values.items()
        }


class SentimentIngestionPipeline:
    """
    Main pipeline for ingesting and analyzing sentiment data.
    Combines lexicon-based analysis with keyword frequency tracking.
    """
    
    def __init__(
        self,
        buffer_size: int = 10000,
        update_interval: float = 60.0
    ):
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        
        # Sentiment analyzer
        self.sentiment_analyzer = LexiconBasedSentiment(window_size=buffer_size)
        
        # Macro tracker
        self.macro_tracker = MacroEconomicTracker()
        
        # Message buffers by source
        self.message_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Aggregated sentiment history
        self.sentiment_history = deque(maxlen=1000)
        
        # Running state
        self.running = False
        self.tasks: List[asyncio.Task] = []
        
        # Statistics
        self.stats = {
            'messages_processed': 0,
            'sentiment_updates': 0,
            'macro_updates': 0
        }
    
    def ingest_message(self, text: str, source: str = 'unknown'):
        """
        Ingest a single message/text for sentiment analysis.
        This is the main entry point for real-time data.
        """
        # Store in buffer
        self.message_buffers[source].append({
            'text': text,
            'timestamp': time.time()
        })
        
        # Analyze sentiment
        score = self.sentiment_analyzer.analyze(text, source)
        self.sentiment_history.append(score)
        
        self.stats['messages_processed'] += 1
        
        return score
    
    def ingest_batch(self, messages: List[Dict[str, str]]) -> List[SentimentScore]:
        """Ingest batch of messages."""
        scores = []
        for msg in messages:
            text = msg.get('text', '')
            source = msg.get('source', 'unknown')
            score = self.ingest_message(text, source)
            scores.append(score)
        
        return scores
    
    async def update_macro_indicators(self):
        """Update macroeconomic indicators."""
        tasks = []
        for indicator_name in self.macro_tracker.INDICATORS.keys():
            tasks.append(self.macro_tracker.fetch_indicator(indicator_name))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.stats['macro_updates'] += 1
    
    async def run_update_loop(self):
        """Continuous update loop for macro data."""
        while self.running:
            try:
                await self.update_macro_indicators()
                await asyncio.sleep(self.update_interval * 10)  # Less frequent updates
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Macro update error: {e}")
                await asyncio.sleep(self.update_interval * 10)
    
    async def start(self):
        """Start the ingestion pipeline."""
        self.running = True
        
        # Start macro update loop
        task = asyncio.create_task(self.run_update_loop())
        self.tasks.append(task)
        
        print("Macro/Sentiment ingestion pipeline started")
    
    async def stop(self):
        """Stop the pipeline."""
        self.running = False
        for task in self.tasks:
            task.cancel()
    
    def get_sentiment_features(self) -> np.ndarray:
        """
        Get sentiment feature vector for ML models.
        """
        aggregate = self.sentiment_analyzer.get_aggregate_sentiment()
        
        # Recent sentiment trend
        recent_scores = list(self.sentiment_history)[-100:]
        
        if recent_scores:
            recent_compound = [s.compound_score for s in recent_scores]
            sentiment_trend = np.polyfit(np.arange(len(recent_compound)), recent_compound, 1)[0]
            sentiment_volatility = np.std(recent_compound)
        else:
            sentiment_trend = 0.0
            sentiment_volatility = 0.0
        
        features = np.array([
            aggregate['bullish'],
            aggregate['bearish'],
            aggregate['neutral'],
            aggregate['compound'],
            aggregate.get('total_words', 0) / 10000,  # Normalized volume
            sentiment_trend,
            sentiment_volatility,
            self.stats['messages_processed'] / 100000  # Normalized total volume
        ], dtype=np.float32)
        
        return features
    
    def get_combined_features(self) -> np.ndarray:
        """
        Get combined macro + sentiment feature vector.
        This is the primary input for trading models.
        """
        macro_features = self.macro_tracker.get_macro_feature_vector()
        sentiment_features = self.get_sentiment_features()
        
        return np.concatenate([macro_features, sentiment_features])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline summary."""
        return {
            'stats': self.stats,
            'sentiment': self.sentiment_analyzer.get_aggregate_sentiment(),
            'macro': self.macro_tracker.get_macro_summary(),
            'running': self.running
        }


# Example usage and testing
if __name__ == '__main__':
    import asyncio
    
    async def test_pipeline():
        pipeline = SentimentIngestionPipeline()
        await pipeline.start()
        
        # Simulate ingesting social media posts
        sample_texts = [
            "Bitcoin is going to the moon! 🚀 Bull run incoming!",
            "Market crash imminent, selling everything",
            "Accumulating more ETH at these levels, great opportunity",
            "Fed rate decision tomorrow, expecting hawkish stance",
            "DXY breaking out, risk assets under pressure",
            "DeFi yields are amazing, staking rewards keep coming",
            "Whale alert! Large BTC transfer to exchange detected",
            "Oversold conditions, expecting bounce soon",
            "Resistance holding strong, need breakout confirmation",
            "Long-term bullish, short-term cautious"
        ]
        
        # Ingest sample texts
        for text in sample_texts:
            score = pipeline.ingest_message(text, source='twitter')
            print(f"Text: {text[:50]}...")
            print(f"  Compound Score: {score.compound_score:.3f}")
            print(f"  Bullish: {score.bullish_score:.3f}, Bearish: {score.bearish_score:.3f}")
        
        # Update macro indicators
        await pipeline.update_macro_indicators()
        
        # Get feature vectors
        sentiment_features = pipeline.get_sentiment_features()
        print(f"\nSentiment features shape: {sentiment_features.shape}")
        print(f"Sentiment features: {sentiment_features}")
        
        macro_features = pipeline.macro_tracker.get_macro_feature_vector()
        print(f"\nMacro features shape: {macro_features.shape}")
        
        combined_features = pipeline.get_combined_features()
        print(f"\nCombined features shape: {combined_features.shape}")
        
        # Get summary
        print("\nPipeline Summary:")
        summary = pipeline.get_summary()
        print(f"  Messages processed: {summary['stats']['messages_processed']}")
        print(f"  Aggregate sentiment: {summary['sentiment']}")
        print(f"  Latest CPI: {summary['macro'].get('cpi', {})}")
        print(f"  Latest DXY: {summary['macro'].get('dxy', {})}")
        
        await pipeline.stop()
    
    asyncio.run(test_pipeline())
