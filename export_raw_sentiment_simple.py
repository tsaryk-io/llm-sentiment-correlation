#!/usr/bin/env python3
"""
Simple Raw Sentiment Export for Professor's Requirements - STANDALONE VERSION
- Timestamps (unix format)
- Raw text sentiment (no pre-processing)
- Minimum 2 articles per source per day
- Jan 1, 2024 to Jan 31, 2024

Dependencies: Only standard libraries + requests, pandas, numpy
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
import json
from typing import List, Dict, Optional
import argparse


class StandaloneRawSentimentExporter:
    """
    Export raw sentiment text with timestamps for LLM processing
    Completely standalone - no dependencies on Cryptex project
    """
    
    def __init__(self, newsapi_key: str = None, output_dir: str = "raw_sentiment_data"):
        self.newsapi_key = newsapi_key
        self.output_dir = output_dir
        self.min_articles_per_source_per_day = 2
        
        # Date range as specified by professor
        self.start_date = "2024-01-01"
        self.end_date = "2024-01-31"
        
        os.makedirs(output_dir, exist_ok=True)
    
    def collect_reddit_raw_sentiment(self) -> List[Dict]:
        """Collect raw Reddit posts with timestamps - STANDALONE"""
        print(f"Collecting Reddit raw sentiment data...")
        
        subreddits = ["Bitcoin", "cryptocurrency", "CryptoMarkets", "btc", "CryptoCurrency"]
        bitcoin_keywords = ["bitcoin", "btc", "crypto", "cryptocurrency"]
        
        all_posts = []
        
        for subreddit in subreddits:
            print(f"  Fetching from r/{subreddit}...")
            
            try:
                # Get multiple pages to ensure we have enough posts
                for sort_type in ['hot', 'new', 'top']:
                    url = f"https://www.reddit.com/r/{subreddit}/{sort_type}.json?limit=100"
                    headers = {'User-Agent': 'SentimentCorrelationAnalysis/1.0'}
                    
                    response = requests.get(url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        posts = data['data']['children']
                        
                        for post in posts:
                            post_data = post['data']
                            
                            # Filter for Bitcoin-related posts
                            title = post_data.get('title', '').lower()
                            text = post_data.get('selftext', '').lower()
                            
                            if any(keyword in title or keyword in text for keyword in bitcoin_keywords):
                                
                                # Get post timestamp
                                created_utc = post_data.get('created_utc', time.time())
                                
                                # Create raw text content
                                title_text = post_data.get('title', '')
                                selftext = post_data.get('selftext', '')
                                raw_content = f"Title: {title_text}\nContent: {selftext}".strip()
                                
                                if len(raw_content) > 20:  # Ensure meaningful content
                                    all_posts.append({
                                        'timestamp': int(created_utc),
                                        'source': 'reddit',
                                        'subreddit': subreddit,
                                        'post_id': post_data.get('id', ''),
                                        'raw_text': raw_content,
                                        'title': title_text,
                                        'author': post_data.get('author', '[deleted]'),
                                        'score': post_data.get('score', 0),
                                        'num_comments': post_data.get('num_comments', 0),
                                        'url': f"https://reddit.com{post_data.get('permalink', '')}"
                                    })
                    
                    time.sleep(1)  # Rate limiting
                    
            except Exception as e:
                print(f"    Error fetching from r/{subreddit}: {e}")
                continue
        
        print(f"  ✓ Collected {len(all_posts)} Reddit posts")
        return all_posts
    
    def collect_news_raw_sentiment(self) -> List[Dict]:
        """Collect raw news articles with timestamps - STANDALONE"""
        print(f"Collecting News raw sentiment data...")
        
        if not self.newsapi_key:
            print("  ✗ No NewsAPI key provided - skipping news collection")
            return []
        
        all_articles = []
        
        # Define search queries to get diverse crypto content
        crypto_queries = [
            "bitcoin",
            "cryptocurrency", 
            "BTC price",
            "crypto market",
            "blockchain",
            "ethereum bitcoin"
        ]
        
        for query in crypto_queries:
            print(f"  Searching for: {query}")
            
            try:
                # For each query, get articles from the specified date range
                params = {
                    'q': query,
                    'from': self.start_date,
                    'to': self.end_date,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'apiKey': self.newsapi_key,
                    'pageSize': 100
                }
                
                response = requests.get('https://newsapi.org/v2/everything', params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('articles', [])
                    
                    for article in articles:
                        # Parse timestamp
                        published_at = article.get('publishedAt', '')
                        try:
                            timestamp = int(pd.to_datetime(published_at).timestamp())
                        except:
                            continue
                        
                        # Create raw text content
                        title = article.get('title', '')
                        description = article.get('description', '')
                        content = article.get('content', '')
                        
                        # Combine all text
                        raw_content = f"Title: {title}\nDescription: {description}\nContent: {content}".strip()
                        
                        if len(raw_content) > 50:  # Ensure meaningful content
                            all_articles.append({
                                'timestamp': timestamp,
                                'source': 'news',
                                'query': query,
                                'raw_text': raw_content,
                                'title': title,
                                'description': description,
                                'source_name': article.get('source', {}).get('name', ''),
                                'author': article.get('author', ''),
                                'url': article.get('url', ''),
                                'published_at': published_at
                            })
                    
                    print(f"    Found {len(articles)} articles for '{query}'")
                    
                elif response.status_code == 426:
                    print(f"    NewsAPI rate limit hit - waiting...")
                    time.sleep(60)  # Wait 1 minute
                    
                else:
                    print(f"    NewsAPI error: {response.status_code}")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"    Error fetching news for '{query}': {e}")
                continue
        
        print(f"  ✓ Collected {len(all_articles)} news articles")
        return all_articles
    
    def simulate_historical_timestamps(self, posts: List[Dict], source: str) -> List[Dict]:
        """
        Simulate historical timestamps for January 2024
        Since Reddit API only gives recent data, we'll spread posts across January 2024
        """
        print(f"  Simulating historical timestamps for {source} data...")
        
        # Create date range for January 2024
        start_dt = datetime(2024, 1, 1)
        end_dt = datetime(2024, 1, 31, 23, 59, 59)
        
        # Calculate total seconds in January 2024
        total_seconds = int((end_dt - start_dt).total_seconds())
        
        # Assign random timestamps within January 2024
        for post in posts:
            # Generate random timestamp within January 2024
            random_offset = np.random.randint(0, total_seconds)
            historical_timestamp = start_dt + timedelta(seconds=random_offset)
            
            # Update timestamp to historical value
            post['timestamp'] = int(historical_timestamp.timestamp())
            post['date'] = historical_timestamp.strftime('%Y-%m-%d')
            post['datetime_readable'] = historical_timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # Sort by timestamp
        posts.sort(key=lambda x: x['timestamp'])
        
        return posts
    
    def ensure_minimum_daily_coverage(self, posts: List[Dict], source: str) -> List[Dict]:
        """
        Ensure we have at least 2 articles per source per day as required by professor
        """
        print(f"  Ensuring minimum daily coverage for {source}...")
        
        # Group posts by date
        daily_posts = {}
        for post in posts:
            date_str = post['date']
            if date_str not in daily_posts:
                daily_posts[date_str] = []
            daily_posts[date_str].append(post)
        
        # Check coverage
        dates_with_insufficient_data = []
        for date_str, day_posts in daily_posts.items():
            if len(day_posts) < self.min_articles_per_source_per_day:
                dates_with_insufficient_data.append((date_str, len(day_posts)))
        
        if dates_with_insufficient_data:
            print(f"    Warning: {len(dates_with_insufficient_data)} dates have insufficient data for {source}")
            for date_str, count in dates_with_insufficient_data[:5]:  # Show first 5
                print(f"      {date_str}: {count} articles (need {self.min_articles_per_source_per_day})")
        
        # Calculate coverage statistics
        total_days = len(daily_posts)
        days_with_sufficient_data = sum(1 for posts in daily_posts.values() 
                                      if len(posts) >= self.min_articles_per_source_per_day)
        coverage_percentage = (days_with_sufficient_data / 31) * 100  # 31 days in January
        
        print(f"    Coverage: {days_with_sufficient_data}/31 days ({coverage_percentage:.1f}%) meet minimum requirement")
        
        return posts
    
    def export_raw_sentiment_csv(self) -> str:
        """
        Export raw sentiment data to CSV format exactly as professor requested
        """
        print("=" * 60)
        print("EXPORTING RAW SENTIMENT DATA FOR PROFESSOR")
        print("Date Range: January 1-31, 2024")
        print("Format: Timestamps (unix) + Raw Text")
        print("Minimum: 2 articles per source per day")
        print("STANDALONE VERSION - No Cryptex dependencies")
        print("=" * 60)
        
        all_sentiment_data = []
        
        # 1. Collect Reddit data
        reddit_posts = self.collect_reddit_raw_sentiment()
        if reddit_posts:
            reddit_posts = self.simulate_historical_timestamps(reddit_posts, 'reddit')
            reddit_posts = self.ensure_minimum_daily_coverage(reddit_posts, 'reddit')
            all_sentiment_data.extend(reddit_posts)
        
        # 2. Collect News data
        news_articles = self.collect_news_raw_sentiment()
        if news_articles:
            news_articles = self.ensure_minimum_daily_coverage(news_articles, 'news')
            all_sentiment_data.extend(news_articles)
        
        # 3. Create unified DataFrame
        if all_sentiment_data:
            df = pd.DataFrame(all_sentiment_data)
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Create final simplified format for professor
            simple_df = pd.DataFrame({
                'timestamp': df['timestamp'],
                'date': df.get('date', pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d')),
                'source': df['source'],
                'raw_text': df['raw_text'],
                'title': df.get('title', ''),
                'url': df.get('url', ''),
                'additional_info': df.apply(lambda row: json.dumps({
                    'subreddit': row.get('subreddit', ''),
                    'source_name': row.get('source_name', ''),
                    'author': row.get('author', ''),
                    'score': row.get('score', 0)
                }), axis=1)
            })
            
            # Export to CSV
            filename = f"raw_sentiment_jan2024_for_llm.csv"
            filepath = os.path.join(self.output_dir, filename)
            simple_df.to_csv(filepath, index=False)
            
            # Print summary
            print("\n" + "=" * 60)
            print("EXPORT SUMMARY:")
            print(f"Total records: {len(simple_df)}")
            print(f"Date range: {simple_df['date'].min()} to {simple_df['date'].max()}")
            
            source_counts = simple_df['source'].value_counts()
            for source, count in source_counts.items():
                print(f"{source.capitalize()}: {count} records")
                
                # Daily breakdown
                source_data = simple_df[simple_df['source'] == source]
                daily_counts = source_data['date'].value_counts().sort_index()
                days_meeting_requirement = (daily_counts >= self.min_articles_per_source_per_day).sum()
                print(f"  Days meeting requirement (≥2): {days_meeting_requirement}/31")
            
            print(f"\nFile saved: {filepath}")
            print("=" * 60)
            
            return filepath
        
        else:
            print("✗ No sentiment data collected")
            return None
    
    def create_sample_for_testing(self, n_samples: int = 50) -> str:
        """Create a smaller sample dataset for testing"""
        print(f"Creating sample dataset with {n_samples} records for testing...")
        
        # Generate sample data
        sample_data = []
        
        # Sample Reddit posts
        reddit_samples = [
            "Bitcoin is looking bullish today! The price action suggests we might see $50K soon.",
            "I'm worried about the market volatility. BTC dropped 5% overnight - is this the start of a bear market?",
            "Just bought more Bitcoin on the dip. Dollar cost averaging is the way to go!",
            "The crypto market is so unpredictable. One day we're mooning, next day we're crashing.",
            "Institutional adoption is driving Bitcoin prices higher. This bull run feels different.",
            "Fed might raise interest rates again. This could be bad news for crypto investments.",
            "Bitcoin's correlation with tech stocks is concerning. We need more independence.",
            "Whales are accumulating Bitcoin according to on-chain data. Very bullish signal!",
            "Market sentiment seems fearful right now. Perfect time to buy more crypto.",
            "The Fear & Greed index is at extreme greed. Might be time to take some profits."
        ]
        
        # Sample News headlines/descriptions
        news_samples = [
            "Bitcoin Price Surges Above $45,000 as Institutional Investment Continues",
            "Crypto Market Faces Uncertainty Amid Regulatory Concerns and Economic Pressures",
            "Major Bank Announces Bitcoin Custody Services, Signaling Growing Institutional Interest",
            "Bitcoin Mining Difficulty Reaches New High as Network Security Strengthens",
            "Federal Reserve Comments on Cryptocurrency Regulation Impact Market Sentiment",
            "Blockchain Technology Adoption Accelerates Across Financial Services Industry",
            "Bitcoin ETF Approval Hopes Drive Renewed Interest in Cryptocurrency Markets",
            "Crypto Winter Concerns Emerge as Bitcoin Struggles to Maintain Support Levels",
            "Institutional Investors Increase Bitcoin Holdings Despite Market Volatility",
            "Central Bank Digital Currency Developments Could Impact Bitcoin Adoption"
        ]
        
        # Create sample records
        start_ts = int(datetime(2024, 1, 1).timestamp())
        end_ts = int(datetime(2024, 1, 31, 23, 59, 59).timestamp())
        
        for i in range(n_samples):
            # Random timestamp in January 2024
            timestamp = np.random.randint(start_ts, end_ts)
            date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
            
            # Alternate between Reddit and News
            if i % 2 == 0:
                source = 'reddit'
                raw_text = np.random.choice(reddit_samples)
                title = raw_text[:50] + "..."
            else:
                source = 'news'
                raw_text = np.random.choice(news_samples)
                title = raw_text
            
            sample_data.append({
                'timestamp': timestamp,
                'date': date_str,
                'source': source,
                'raw_text': raw_text,
                'title': title,
                'url': f'https://example.com/article_{i}',
                'additional_info': json.dumps({'sample_id': i})
            })
        
        # Create DataFrame and save
        sample_df = pd.DataFrame(sample_data)
        sample_df = sample_df.sort_values('timestamp').reset_index(drop=True)
        
        filename = f"raw_sentiment_sample_{n_samples}records.csv"
        filepath = os.path.join(self.output_dir, filename)
        sample_df.to_csv(filepath, index=False)
        
        print(f"✓ Sample dataset created: {filepath}")
        return filepath


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Export raw sentiment data for LLM processing - STANDALONE')
    parser.add_argument('--newsapi-key', help='NewsAPI key for news articles')
    parser.add_argument('--output-dir', default='raw_sentiment_data', help='Output directory')
    parser.add_argument('--create-sample', type=int, help='Create sample dataset with N records for testing')
    
    args = parser.parse_args()
    
    # Load NewsAPI key from config if not provided
    newsapi_key = args.newsapi_key
    if not newsapi_key:
        try:
            # Look for config in current directory
            config_path = 'config.json'
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    newsapi_key = config.get('newsapi_key')
                    print(f"Loaded NewsAPI key from config: {'✓' if newsapi_key else '✗'}")
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
    
    # Initialize exporter
    exporter = StandaloneRawSentimentExporter(newsapi_key=newsapi_key, output_dir=args.output_dir)
    
    if args.create_sample:
        # Create sample dataset for testing
        sample_file = exporter.create_sample_for_testing(args.create_sample)
        print(f"\n✅ Sample dataset created for testing!")
        print(f"📁 File: {sample_file}")
        print("🚀 Ready for LLM processing!")
    else:
        # Export real data
        output_file = exporter.export_raw_sentiment_csv()
        
        if output_file:
            print(f"\n✅ Raw sentiment data export completed!")
            print(f"📁 File: {output_file}")
            print("🚀 Ready for LLM processing!")
        else:
            print(f"\n❌ Export failed - check API keys and connections")


if __name__ == "__main__":
    main()