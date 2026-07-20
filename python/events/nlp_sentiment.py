"""
NLP Sentiment Analysis Module - High-Speed Deterministic Sentiment Scoring

High-speed, deterministic NLP sentiment analysis WITHOUT LLMs.
Uses custom financial lexicons, VADER, and lightweight FinBERT (quantized to INT8)
to score news events in <1ms on the AMD CPU.

Hardware Target: AMD Ryzen AI 5 with optimized vectorization
Memory Constraint: Pre-loaded models, bounded memory pools
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import time
import re


@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    compound: float  # Overall sentiment (-1 to 1)
    positive: float  # Positive ratio (0 to 1)
    neutral: float   # Neutral ratio (0 to 1)
    negative: float  # Negative ratio (0 to 1)
    urgency: float   # Urgency score (0 to 1)
    confidence: float  # Confidence in the score (0 to 1)
    processing_time_ns: int
    method: str  # Which method was used


class FinancialLexicon:
    """
    Custom financial sentiment lexicon optimized for crypto markets.
    Pre-computed for fast lookup without runtime parsing.
    """
    
    # Positive financial terms with weights
    POSITIVE_TERMS = {
        'bullish': 0.8, 'rally': 0.7, 'surge': 0.6, 'breakout': 0.7,
        'upgrade': 0.5, 'partnership': 0.6, 'adoption': 0.7, 'growth': 0.5,
        'profit': 0.6, 'gain': 0.5, 'rise': 0.4, 'jump': 0.5, 'soar': 0.7,
        'milestone': 0.5, 'success': 0.6, 'approval': 0.6, 'etf': 0.4,
        'institutional': 0.3, 'whale': 0.3, 'accumulation': 0.4, 'hodl': 0.5,
        'moon': 0.6, 'pump': 0.4, 'green': 0.3, 'long': 0.3,
    }
    
    # Negative financial terms with weights
    NEGATIVE_TERMS = {
        'bearish': -0.8, 'crash': -0.9, 'plunge': -0.7, 'breakdown': -0.7,
        'downgrade': -0.5, 'lawsuit': -0.6, 'hack': -0.9, 'exploit': -0.8,
        'loss': -0.6, 'decline': -0.5, 'fall': -0.4, 'drop': -0.5, 'dump': -0.6,
        'ban': -0.7, 'regulation': -0.4, 'sec': -0.3, 'warning': -0.5,
        'risk': -0.4, 'volatile': -0.3, 'liquidation': -0.7, 'margin': -0.3,
        'short': -0.3, 'red': -0.3, 'panic': -0.6, 'fud': -0.7,
    }
    
    # Intensifiers that modify sentiment strength
    INTENSIFIERS = {
        'very': 1.25, 'extremely': 1.5, 'highly': 1.3, 'significantly': 1.2,
        'massively': 1.5, 'slightly': 0.5, 'moderately': 0.75, 'somewhat': 0.6,
    }
    
    # Negators that flip sentiment
    NEGATORS = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere'}
    
    def __init__(self):
        # Pre-compile regex patterns
        self.word_pattern = re.compile(r'\b[a-z]+\b', re.IGNORECASE)
        
        # Create lookup dictionaries for O(1) access
        self._positive_lookup = {k.lower(): v for k, v in self.POSITIVE_TERMS.items()}
        self._negative_lookup = {k.lower(): v for k, v in self.NEGATIVE_TERMS.items()}
        self._intensifier_lookup = {k.lower(): v for k, v in self.INTENSIFIERS.items()}
        self._negator_set = set(k.lower() for k in self.NEGATORS)
    
    def analyze(self, text: str) -> Tuple[float, float, float]:
        """
        Analyze sentiment using lexicon approach.
        Returns (positive_score, negative_score, intensity_modifier)
        """
        words = self.word_pattern.findall(text.lower())
        
        pos_score = 0.0
        neg_score = 0.0
        intensity = 1.0
        
        prev_word_was_negator = False
        prev_word_was_intensifier = False
        intensifier_value = 1.0
        
        for i, word in enumerate(words):
            current_intensity = intensity
            
            # Check for intensifiers
            if word in self._intensifier_lookup:
                intensifier_value = self._intensifier_lookup[word]
                prev_word_was_intensifier = True
                continue
            
            # Check for negators
            if word in self._negator_set:
                prev_word_was_negator = True
                continue
            
            # Get base sentiment
            sentiment = 0.0
            if word in self._positive_lookup:
                sentiment = self._positive_lookup[word]
            elif word in self._negative_lookup:
                sentiment = self._negative_lookup[word]
            
            # Apply modifiers
            if sentiment != 0:
                if prev_word_was_negator:
                    sentiment = -sentiment * 0.5  # Negation weakens rather than flips
                
                if prev_word_was_intensifier:
                    sentiment *= intensifier_value
                
                if sentiment > 0:
                    pos_score += sentiment * current_intensity
                else:
                    neg_score += abs(sentiment) * current_intensity
            
            # Reset modifiers
            prev_word_was_negator = False
            prev_word_was_intensifier = False
            intensifier_value = 1.0
        
        return pos_score, neg_score, intensity


class VADERLite:
    """
    Lightweight VADER-inspired sentiment analyzer.
    Simplified version optimized for speed without external dependencies.
    """
    
    def __init__(self):
        self.lexicon = FinancialLexicon()
    
    def polarity_scores(self, text: str) -> Dict[str, float]:
        """Calculate sentiment polarity scores"""
        pos, neg, intensity = self.lexicon.analyze(text)
        
        total = pos + neg
        if total == 0:
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 1.0,
                'neg': 0.0,
            }
        
        # Normalize scores
        pos_norm = pos / (total + 1)
        neg_norm = neg / (total + 1)
        neu_norm = 1.0 - pos_norm - neg_norm
        
        # Compound score
        compound = pos_norm - neg_norm
        
        # Clamp to [-1, 1]
        compound = max(-1.0, min(1.0, compound))
        
        return {
            'compound': compound,
            'pos': max(0.0, pos_norm),
            'neu': max(0.0, neu_norm),
            'neg': max(0.0, neg_norm),
        }


class QuantizedFinBERT:
    """
    INT8 quantized FinBERT wrapper for fast inference.
    Uses ONNX Runtime with quantization for sub-millisecond inference.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.session = None
        self._initialized = False
        
        # Fallback to lexicon if model not available
        self.fallback = VADERLite()
        
        # Try to initialize ONNX runtime
        try:
            import onnxruntime as ort
            if model_path:
                self.session = ort.InferenceSession(
                    model_path,
                    providers=['CPUExecutionProvider']
                )
                self._initialized = True
        except ImportError:
            print("[Sentiment] ONNX Runtime not available, using fallback")
    
    def predict(self, text: str) -> Optional[Dict[str, float]]:
        """Run quantized FinBERT inference"""
        if not self._initialized or self.session is None:
            return None
        
        # Tokenize (simplified - in production use proper tokenizer)
        tokens = self._tokenize(text)
        
        # Run inference
        try:
            outputs = self.session.run(
                None,
                {'input_ids': np.array([tokens], dtype=np.int64)}
            )
            
            # Parse outputs (logits)
            logits = outputs[0][0]
            probs = self._softmax(logits)
            
            return {
                'positive': float(probs[2]) if len(probs) > 2 else 0.0,
                'neutral': float(probs[1]) if len(probs) > 1 else 0.0,
                'negative': float(probs[0]) if len(probs) > 0 else 0.0,
            }
        except Exception as e:
            print(f"[Sentiment] Inference error: {e}")
            return None
    
    def _tokenize(self, text: str, max_length: int = 128) -> List[int]:
        """Simple tokenization - replace with actual tokenizer in production"""
        # This is a placeholder - real implementation would use BERT tokenizer
        return [101] + [hash(w) % 1000 for w in text.split()[:max_length-2]] + [102]
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


class SentimentAnalyzer:
    """
    Main sentiment analysis engine combining multiple methods.
    Automatically selects fastest appropriate method based on content.
    """
    
    def __init__(self, use_finetuned: bool = False, model_path: Optional[str] = None):
        self.use_finetuned = use_finetuned
        self.vader = VADERLite()
        self.finboxrt = QuantizedFinBERT(model_path) if use_finetuned else None
        
        # Statistics
        self analyses_count = 0
        self.total_time_ns = 0
        self.method_counts: Dict[str, int] = defaultdict(int)
    
    def analyze(self, text: str, title: str = "") -> SentimentResult:
        """
        Analyze sentiment of text using best available method.
        Target: <1ms processing time on AMD Ryzen AI 5
        """
        start_ns = time.time_ns()
        
        combined_text = f"{title} {text}" if title else text
        
        # Try FinBERT first if enabled
        if self.use_finetuned and self.finboxrt:
            result = self.finboxrt.predict(combined_text)
            if result:
                processing_time = time.time_ns() - start_ns
                self._record_stats(processing_time, 'finbert')
                
                compound = result['positive'] - result['negative']
                return SentimentResult(
                    compound=compound,
                    positive=result['positive'],
                    neutral=result['neutral'],
                    negative=result['negative'],
                    urgency=self._calculate_urgency(combined_text),
                    confidence=0.85,  # Model confidence
                    processing_time_ns=processing_time,
                    method='finbert_int8',
                )
        
        # Fallback to VADER-lite (fast, deterministic)
        scores = self.vader.polarity_scores(combined_text)
        processing_time = time.time_ns() - start_ns
        self._record_stats(processing_time, 'vader')
        
        return SentimentResult(
            compound=scores['compound'],
            positive=scores['pos'],
            neutral=scores['neu'],
            negative=scores['neg'],
            urgency=self._calculate_urgency(combined_text),
            confidence=0.7,  # Lower confidence for lexicon-based
            processing_time_ns=processing_time,
            method='vader_lite',
        )
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts efficiently"""
        return [self.analyze(text) for text in texts]
    
    def _calculate_urgency(self, text: str) -> float:
        """Calculate urgency score based on text content"""
        urgency_words = {
            'breaking', 'urgent', 'alert', 'just', 'now', 'flash',
            'immediate', 'developing', 'live', 'watch',
        }
        
        text_lower = text.lower()
        urgency_count = sum(1 for word in urgency_words if word in text_lower)
        
        # Time-sensitive phrases
        time_phrases = ['minutes ago', 'hours ago', 'today', 'this week']
        time_count = sum(1 for phrase in time_phrases if phrase in text_lower)
        
        return min(1.0, (urgency_count * 0.3 + time_count * 0.2))
    
    def _record_stats(self, time_ns: int, method: str):
        """Record analysis statistics"""
        self.analyses_count += 1
        self.total_time_ns += time_ns
        self.method_counts[method] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        avg_time = self.total_time_ns / max(1, self.analyses_count)
        return {
            'total_analyses': self.analyses_count,
            'avg_time_ns': avg_time,
            'avg_time_ms': avg_time / 1_000_000,
            'method_distribution': dict(self.method_counts),
        }


# Example usage
if __name__ == '__main__':
    analyzer = SentimentAnalyzer(use_finetuned=False)
    
    test_texts = [
        "Bitcoin surges to new highs as institutional adoption grows",
        "Crypto market crashes amid regulatory concerns and hack reports",
        "Ethereum upgrade successfully deployed, network performance improves",
        "SEC warns investors about cryptocurrency risks",
    ]
    
    print("Sentiment Analysis Results:\n")
    for text in test_texts:
        result = analyzer.analyze(text)
        print(f"Text: {text[:50]}...")
        print(f"  Compound: {result.compound:.3f}")
        print(f"  Positive: {result.positive:.3f}, Negative: {result.negative:.3f}")
        print(f"  Urgency: {result.urgency:.3f}, Confidence: {result.confidence:.3f}")
        print(f"  Method: {result.method}, Time: {result.processing_time_ns/1_000_000:.2f}ms\n")
    
    stats = analyzer.get_stats()
    print(f"Stats: {stats}")
