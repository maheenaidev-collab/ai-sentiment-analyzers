"""
Visualization module for sentiment analysis results.
Author: Maheen Riaz
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_sentiment_distribution(results, save_path='output/sentiment_distribution.png'):
    """Generate a pie chart of sentiment distribution."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    sentiments = [r['sentiment'] for r in results]
    counts = {
        'Positive': sentiments.count('positive'),
        'Negative': sentiments.count('negative'),
        'Neutral': sentiments.count('neutral')
    }

    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    labels = [f"{k}\n({v})" for k, v in counts.items()]

    plt.figure(figsize=(8, 6))
    plt.pie(counts.values(), labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=140, textprops={'fontsize': 12})
    plt.title('Sentiment Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to {save_path}")


def plot_confidence_histogram(results, save_path='output/confidence_histogram.png'):
    """Generate a histogram of confidence scores."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    confidences = [r['confidence'] for r in results]

    plt.figure(figsize=(10, 6))
    sns.histplot(confidences, bins=20, kde=True, color='#3498db')
    plt.xlabel('Confidence Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Confidence Score Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to {save_path}")


def plot_score_comparison(results, save_path='output/score_comparison.png'):
    """Generate a bar chart comparing VADER and TextBlob scores."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    vader_scores = [r['scores']['vader']['compound'] for r in results]
    textblob_scores = [r['scores']['textblob']['polarity'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(vader_scores, bins=20, kde=True, color='#e74c3c', ax=axes[0])
    axes[0].set_title('VADER Compound Scores', fontsize=14)
    axes[0].set_xlabel('Score')

    sns.histplot(textblob_scores, bins=20, kde=True, color='#2ecc71', ax=axes[1])
    axes[1].set_title('TextBlob Polarity Scores', fontsize=14)
    axes[1].set_xlabel('Score')

    plt.suptitle('Engine Score Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to {save_path}")
