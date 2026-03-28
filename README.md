# 🔍 AI Sentiment Analyzer

An intelligent sentiment analysis tool powered by Python and NLP that detects emotions and sentiment from text, reviews, tweets, and customer feedback.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NLP](https://img.shields.io/badge/NLP-TextBlob-green.svg)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🚀 Features

- **Multi-Source Analysis** — Analyze tweets, reviews, comments, and any text
- **Sentiment Detection** — Positive, Negative, Neutral classification
- **Confidence Score** — Get probability scores for each sentiment
- **Batch Processing** — Analyze multiple texts at once from CSV files
- **Visual Reports** — Generate charts and graphs of sentiment distribution
- **Real-time Analysis** — Interactive mode for instant sentiment checking
- **Export Results** — Save analysis results to CSV/JSON

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/maheenaidev-collab/ai-sentiment-analyzer.git
cd ai-sentiment-analyzer

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
