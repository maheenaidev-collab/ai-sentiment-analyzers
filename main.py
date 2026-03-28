"""
AI Sentiment Analyzer - CLI Entry Point
Author: Maheen Riaz
"""

import argparse
from sentiment_analyzer import SentimentAnalyzer
from visualizer import plot_sentiment_distribution, plot_confidence_histogram, plot_score_comparison
from utils import create_sample_csv, read_text_file


def interactive_mode(analyzer):
    """Run interactive sentiment analysis."""
    print("\n" + "=" * 50)
    print("🔍 AI Sentiment Analyzer - Interactive Mode")
    print("=" * 50)
    print("Type text to analyze. Type 'quit' to exit.\n")

    while True:
        text = input("📝 Enter text: ").strip()

        if text.lower() in ('quit', 'exit', 'q'):
            print("\nGoodbye! 👋")
            break

        if not text:
            print("⚠️  Please enter some text.\n")
            continue

        result = analyzer.analyze(text)

        emoji = {'positive': '✅', 'negative': '❌', 'neutral': '➖'}
        print(f"\n  Sentiment: {emoji[result['sentiment']]} {result['sentiment'].upper()}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Combined Score: {result['combined_score']:.4f}")
        print(f"  VADER Compound: {result['scores']['vader']['compound']:.4f}")
        print(f"  TextBlob Polarity: {result['scores']['textblob']['polarity']:.4f}")
        print()


def main():
    parser = argparse.ArgumentParser(description='AI Sentiment Analyzer')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--text', '-t', type=str, help='Analyze a single text')
    parser.add_argument('--file', '-f', type=str, help='Analyze texts from CSV file')
    parser.add_argument('--column', '-c', type=str, default='review', help='CSV column name')
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    parser.add_argument('--visualize', '-v', action='store_true', help='Generate visual reports')
    parser.add_argument('--sample', action='store_true', help='Run analysis on sample data')

    args = parser.parse_args()
    analyzer = SentimentAnalyzer()

    if args.interactive:
        interactive_mode(analyzer)

    elif args.text:
        result = analyzer.analyze(args.text)
        emoji = {'positive': '✅', 'negative': '❌', 'neutral': '➖'}
        print(f"\nText: {args.text}")
        print(f"Sentiment: {emoji[result['sentiment']]} {result['sentiment'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")

    elif args.file or args.sample:
        filepath = args.file
        if args.sample:
            filepath = create_sample_csv()

        results = analyzer.analyze_csv(filepath, text_column=args.column)
        summary = analyzer.summary(results)

        print(f"\n📊 Analysis Summary")
        print(f"{'=' * 40}")
        print(f"Total Analyzed: {summary['total_analyzed']}")
        print(f"✅ Positive: {summary['positive']} ({summary['positive_pct']}%)")
        print(f"❌ Negative: {summary['negative']} ({summary['negative_pct']}%)")
        print(f"➖ Neutral: {summary['neutral']} ({summary['neutral_pct']}%)")
        print(f"Avg Confidence: {summary['avg_confidence']:.2%}")

        if args.output:
            analyzer.export_results(results, args.output)

        if args.visualize:
            plot_sentiment_distribution(results)
            plot_confidence_histogram(results)
            plot_score_comparison(results)
            print("\n📈 Visual reports saved to output/ folder")

    else:
        print("🔍 AI Sentiment Analyzer")
        print("Run with --help to see options")
        print("Quick start: python main.py --interactive")
        print("Sample run: python main.py --sample --visualize")


if __name__ == '__main__':
    main()
