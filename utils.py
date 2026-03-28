"""
Utility functions for AI Sentiment Analyzer.
Author: Maheen Riaz
"""

import os
import csv


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def read_text_file(filepath):
    """Read texts from a plain text file (one per line)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def create_sample_csv(filepath='sample_data/sample_reviews.csv'):
    """Generate a sample CSV file for testing."""
    ensure_dir(os.path.dirname(filepath))

    reviews = [
        "This product is absolutely amazing! Best purchase I've ever made.",
        "Terrible quality. Broke after one day. Want my money back.",
        "It's okay, nothing special. Does what it says.",
        "I love this so much! Exceeded all my expectations!",
        "Worst customer service I've ever experienced. Never buying again.",
        "The delivery was on time. Package was intact.",
        "Fantastic! My kids love it. Will buy again for sure!",
        "Complete waste of money. Don't buy this garbage.",
        "Pretty good for the price. Would recommend.",
        "Not bad, but could be improved. Average product.",
        "Absolutely outstanding! Five stars all the way!",
        "Disappointing. The description was misleading.",
        "Received the item today. It matches the description.",
        "This changed my life! Can't imagine living without it.",
        "Horrible experience from start to finish. 0/10."
    ]

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'review', 'source'])
        for i, review in enumerate(reviews, 1):
            writer.writerow([i, review, 'sample'])

    print(f"Sample CSV created at {filepath}")
    return filepath
