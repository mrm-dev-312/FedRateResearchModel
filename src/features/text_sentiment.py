"""
Text sentiment analysis using Google Gemini API.
Processes text data and generates sentiment scores for market analysis.
"""

import asyncio
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Union, Tuple
import logging
from datetime import datetime
import google.generativeai as genai
from src.db.client import get_db

logger = logging.getLogger(__name__)

# Feature version constant for traceability
FEATURE_VERSION = "1.0.0"

class TextSentimentAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini sentiment analyzer.
        
        Args:
            api_key: Google Gemini API key. If None, reads from environment.
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or provided")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        self.feature_version = FEATURE_VERSION
    
    async def analyze_sentiment(
        self, 
        text: str, 
        context: str = "financial markets"
    ) -> Dict[str, float]:
        """
        Analyze sentiment of text using Gemini API.
        
        Args:
            text: Text to analyze
            context: Context for sentiment analysis
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            # Create a prompt for financial sentiment analysis
            prompt = f"""
            Analyze the sentiment of the following text in the context of {context}.
            
            Text: "{text}"
            
            Please provide:
            1. Overall sentiment score (-1.0 to 1.0, where -1.0 is very negative, 0 is neutral, 1.0 is very positive)
            2. Confidence score (0.0 to 1.0, where 1.0 is very confident)
            3. Key themes (bullish/bearish factors)
            
            Respond in JSON format:
            {{
                "sentiment_score": <float>,
                "confidence": <float>,
                "themes": ["theme1", "theme2"],
                "reasoning": "brief explanation"
            }}
            """
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Parse response (simplified - in production, use proper JSON parsing)
            response_text = response.text
            
            # Extract sentiment score (basic parsing - improve for production)
            import re
            import json
            
            try:
                # Try to parse as JSON
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return {
                        'sentiment_score': float(result.get('sentiment_score', 0.0)),
                        'confidence': float(result.get('confidence', 0.5)),
                        'themes': result.get('themes', []),
                        'reasoning': result.get('reasoning', ''),
                        'raw_response': response_text
                    }
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Fallback: basic keyword-based sentiment
            positive_words = ['positive', 'bullish', 'strong', 'growth', 'optimistic']
            negative_words = ['negative', 'bearish', 'weak', 'decline', 'pessimistic']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count + neg_count == 0:
                sentiment_score = 0.0
            else:
                sentiment_score = (pos_count - neg_count) / (pos_count + neg_count)
            
            return {
                'sentiment_score': sentiment_score,
                'confidence': 0.3,  # Low confidence for fallback
                'themes': [],
                'reasoning': 'Fallback keyword analysis',
                'raw_response': response_text
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'themes': [],
                'reasoning': f'Error: {str(e)}',
                'raw_response': ''
            }
    
    async def analyze_fed_minutes(self, text: str) -> Dict[str, float]:
        """
        Specialized analysis for Fed meeting minutes.
        
        Args:
            text: Fed minutes text
            
        Returns:
            Dictionary with Fed-specific sentiment scores
        """
        # Fed-specific context
        context = "Federal Reserve policy and economic outlook"
        
        # Base sentiment analysis
        base_result = await self.analyze_sentiment(text, context)
        
        # Add Fed-specific features
        hawkish_words = [
            'rate hike', 'inflation concerns', 'tightening', 'restrictive',
            'cooling economy', 'price pressures'
        ]
        
        dovish_words = [
            'rate cut', 'stimulus', 'accommodative', 'supportive',
            'employment concerns', 'economic softening'
        ]
        
        text_lower = text.lower()
        hawkish_score = sum(1 for phrase in hawkish_words if phrase in text_lower)
        dovish_score = sum(1 for phrase in dovish_words if phrase in text_lower)
        
        total_policy_signals = hawkish_score + dovish_score
        if total_policy_signals > 0:
            policy_bias = (hawkish_score - dovish_score) / total_policy_signals
        else:
            policy_bias = 0.0
        
        base_result.update({
            'policy_bias': policy_bias,  # -1 (dovish) to 1 (hawkish)
            'hawkish_signals': hawkish_score,
            'dovish_signals': dovish_score,
            'policy_certainty': min(total_policy_signals / 10, 1.0)  # Normalized
        })
        
        return base_result
    
    async def analyze_earnings_call(self, text: str, ticker: str) -> Dict[str, float]:
        """
        Specialized analysis for earnings call transcripts.
        
        Args:
            text: Earnings call transcript
            ticker: Stock ticker
            
        Returns:
            Dictionary with earnings-specific sentiment scores
        """
        context = f"earnings call for {ticker} stock"
        
        # Base sentiment analysis
        base_result = await self.analyze_sentiment(text, context)
        
        # Earnings-specific keywords
        positive_earnings = [
            'beat expectations', 'strong revenue', 'guidance raised',
            'margin expansion', 'market share gains'
        ]
        
        negative_earnings = [
            'missed expectations', 'revenue decline', 'guidance lowered',
            'margin compression', 'competitive pressures'
        ]
        
        text_lower = text.lower()
        pos_earnings = sum(1 for phrase in positive_earnings if phrase in text_lower)
        neg_earnings = sum(1 for phrase in negative_earnings if phrase in text_lower)
        
        total_earnings_signals = pos_earnings + neg_earnings
        if total_earnings_signals > 0:
            earnings_sentiment = (pos_earnings - neg_earnings) / total_earnings_signals
        else:
            earnings_sentiment = base_result['sentiment_score']
        
        base_result.update({
            'earnings_sentiment': earnings_sentiment,
            'positive_signals': pos_earnings,
            'negative_signals': neg_earnings,
            'earnings_certainty': min(total_earnings_signals / 5, 1.0)
        })
        
        return base_result
    
    async def store_sentiment_features(
        self,
        doc_id: str,
        timestamp: datetime,
        sentiment_result: Dict[str, float],
        source: str = "unknown"
    ) -> bool:
        """
        Store sentiment analysis results in TextFeature table.
        
        Args:
            doc_id: Document identifier
            timestamp: Document timestamp
            sentiment_result: Result from sentiment analysis
            source: Source of the text (e.g., "fed_minutes", "earnings_call")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            db = await get_db()
            
            # Create TextFeature record
            await db.textfeature.create(
                data={
                    'doc_id': doc_id,
                    'ts': timestamp,
                    'score': sentiment_result['sentiment_score'],
                    'source': source,
                    # Store additional data as JSON in embedding field (if needed)
                }
            )
            
            logger.info(f"Stored sentiment features for {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing sentiment features: {e}")
            return False

async def analyze_text_sentiment(
    texts: List[str],
    timestamps: List[datetime],
    doc_ids: List[str],
    source: str = "general",
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyze sentiment for multiple texts and return as DataFrame.
    
    Args:
        texts: List of text documents
        timestamps: List of timestamps for each text
        doc_ids: List of document IDs
        source: Source type (e.g., "fed_minutes", "earnings_call")
        api_key: Gemini API key
        
    Returns:
        DataFrame with sentiment features
    """
    analyzer = TextSentimentAnalyzer(api_key)
    
    results = []
    
    for i, (text, timestamp, doc_id) in enumerate(zip(texts, timestamps, doc_ids)):
        logger.info(f"Analyzing sentiment {i+1}/{len(texts)}: {doc_id}")
        
        try:
            if source == "fed_minutes":
                sentiment_result = await analyzer.analyze_fed_minutes(text)
            elif source == "earnings_call":
                # Extract ticker from doc_id if possible
                ticker = doc_id.split('_')[0] if '_' in doc_id else "UNKNOWN"
                sentiment_result = await analyzer.analyze_earnings_call(text, ticker)
            else:
                sentiment_result = await analyzer.analyze_sentiment(text)
            
            # Store in database
            await analyzer.store_sentiment_features(doc_id, timestamp, sentiment_result, source)
            
            # Add to results
            result_row = {
                'doc_id': doc_id,
                'ts': timestamp,
                'source': source,
                'sentiment_score': sentiment_result['sentiment_score'],
                'confidence': sentiment_result['confidence'],
                'feature_version': FEATURE_VERSION
            }
            
            # Add source-specific features
            if 'policy_bias' in sentiment_result:
                result_row.update({
                    'policy_bias': sentiment_result['policy_bias'],
                    'hawkish_signals': sentiment_result['hawkish_signals'],
                    'dovish_signals': sentiment_result['dovish_signals'],
                    'policy_certainty': sentiment_result['policy_certainty']
                })
            
            if 'earnings_sentiment' in sentiment_result:
                result_row.update({
                    'earnings_sentiment': sentiment_result['earnings_sentiment'],
                    'positive_signals': sentiment_result['positive_signals'],
                    'negative_signals': sentiment_result['negative_signals'],
                    'earnings_certainty': sentiment_result['earnings_certainty']
                })
            
            results.append(result_row)
            
            # Rate limiting - sleep briefly between requests
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error processing {doc_id}: {e}")
            # Add error row
            results.append({
                'doc_id': doc_id,
                'ts': timestamp,
                'source': source,
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'feature_version': FEATURE_VERSION,
                'error': str(e)
            })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    async def test_sentiment_analysis():
        """Test sentiment analysis functionality."""
        
        # Sample texts for testing
        sample_texts = [
            "The Federal Reserve maintains a dovish stance with continued accommodative policy to support economic recovery.",
            "Strong earnings beat expectations with robust revenue growth and positive guidance for next quarter.",
            "Market volatility increases amid concerns about inflation and potential rate hikes."
        ]
        
        sample_timestamps = [
            datetime(2024, 1, 1, 14, 0),
            datetime(2024, 1, 2, 16, 30),
            datetime(2024, 1, 3, 9, 0)
        ]
        
        sample_doc_ids = ["fed_minutes_2024_01", "AAPL_earnings_2024_q1", "market_analysis_2024_01"]
        
        print("Testing sentiment analysis...")
        
        # Test without API key (will use fallback)
        try:
            # This will likely fail gracefully and use fallback
            result_df = await analyze_text_sentiment(
                sample_texts, 
                sample_timestamps, 
                sample_doc_ids,
                source="general"
            )
            
            print(f"Results shape: {result_df.shape}")
            print("Sample results:")
            print(result_df[['doc_id', 'sentiment_score', 'confidence']].head())
            
        except Exception as e:
            print(f"Note: Sentiment analysis needs GEMINI_API_KEY: {e}")
            print("Creating mock results for demonstration...")
            
            # Create mock results
            mock_results = pd.DataFrame({
                'doc_id': sample_doc_ids,
                'ts': sample_timestamps,
                'source': 'general',
                'sentiment_score': [0.2, 0.7, -0.3],
                'confidence': [0.8, 0.9, 0.7],
                'feature_version': FEATURE_VERSION
            })
            
            print("Mock sentiment results:")
            print(mock_results)
        
        print("✓ Sentiment analysis test completed")
    
    asyncio.run(test_sentiment_analysis())
