```python
"""
AI Sentiment Analyzer
Analyzes text sentiment using VADER + TextBlob ensemble approach.
Author: Maheen Riaz
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd
import json
import re


class SentimentAnalyzer:
    """Multi-engine sentiment analyzer combining VADER and TextBlob."""

    def __init__(self):
        """Initialize sentiment analysis engines."""
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)

        self.vader = SentimentIntensityAnalyzer()

    def _clean_text(self, text):
        """Preprocess and clean input text."""
        text = str(text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        text = re.sub(r'#(\w+)', r'\1', text)  # Remove hashtag symbol
        text = re.sub(r'[^\w\s!?.,]', '', text)  # Keep basic punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Clean whitespace
        return text

    def _vader_analysis(self, text):
        """Analyze sentiment using VADER."""
        scores = self.vader.polarity_scores(text)
        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']
        }

    def _textblob_analysis(self, text):
        """Analyze sentiment using TextBlob."""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        return {
            'polarity': polarity,
            'subjectivity': subjectivity
        }

    def _classify(self, vader_scores, textblob_scores):
        """Classify sentiment based on ensemble scores."""
        # Weighted ensemble: VADER (60%) + TextBlob (40%)
        vader_compound = vader_scores['compound']
        textblob_polarity = textblob_scores['polarity']

        combined_score = (vader_compound * 0.6) + (textblob_polarity * 0.4)

        if combined_score >= 0.15:
            sentiment = 'positive'
        elif combined_score <= -0.15:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        confidence = min(abs(combined_score) + 0.5, 1.0)

        return sentiment, confidence, combined_score

    def analyze(self, text):
        """
        Analyze sentiment of a single text.

        Args:
            text (str): Text to analyze

        Returns:
            dict: Sentiment analysis results
        """
        cleaned = self._clean_text(text)

        if not cleaned:
            return {
                'text': text,
                'sentiment': 'neutral',
                'confidence': 0.0,
                'combined_score': 0.0,
                'scores': {'vader': {}, 'textblob': {}}
            }

        vader_scores = self._vader_analysis(cleaned)
        textblob_scores = self._textblob_analysis(cleaned)
        sentiment, confidence, combined_score = self._classify(vader_scores, textblob_scores)

        return {
            'text': text,
            'cleaned_text': cleaned,
            'sentiment': sentiment,
            'confidence': round(confidence, 4),
            'combined_score': round(combined_score, 4),
            'scores': {
                'vader': vader_scores,
                'textblob': textblob_scores
            }
        }

    def analyze_batch(self, texts):
        """
        Analyze sentiment of multiple texts.

        Args:
            texts (list): List of texts to analyze

        Returns:
            list: List of sentiment analysis results
        """
        return [self.analyze(text) for text in texts]

    def analyze_csv(self, filepath, text_column='text'):
        """
        Analyze sentiment from a CSV file.

        Args:
            filepath (str): Path to CSV file
            text_column (str): Name of the column containing text

        Returns:
            list: List of sentiment analysis results
        """
        df = pd.read_csv(filepath)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found. Available: {list(df.columns)}")

        texts = df[text_column].dropna().tolist()
        return self.analyze_batch(texts)

    def export_results(self, results, filepath, format='csv'):
        """
        Export analysis results to file.

        Args:
            results (list): Analysis results
            filepath (str): Output file path
            format (str): 'csv' or 'json'
        """
        if format == 'csv':
            rows = []
            for r in results:
                rows.append({
                    'text': r['text'],
                    'sentiment': r['sentiment'],
                    'confidence': r['confidence'],
                    'combined_score': r['combined_score'],
                    'vader_compound': r['scores']['vader'].get('compound', 0),
                    'textblob_polarity': r['scores']['textblob'].get('polarity', 0)
                })
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
            print(f"Results exported to {filepath}")

        elif format == 'json':
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results exported to {filepath}")

    def summary(self, results):
        """
        Generate summary statistics from results.

        Args:
            results (list): Analysis results

        Returns:
            dict: Summary statistics
        """
        total = len(results)
        sentiments = [r['sentiment'] for r in results]

        return {
            'total_analyzed': total,
            'positive': sentiments.count('positive'),
            'negative': sentiments.count('negative'),
            'neutral': sentiments.count('neutral'),
            'positive_pct': round(sentiments.count('positive') / total * 100, 1),
            'negative_pct': round(sentiments.count('negative') / total * 100, 1),
            'neutral_pct': round(sentiments.count('neutral') / total * 100, 1),
            'avg_confidence': round(sum(r['confidence'] for r in results) / total, 4)
        }
