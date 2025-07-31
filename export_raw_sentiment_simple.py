#!/usr/bin/env python3
"""
Raw Sentiment Data Export
Exports Reddit posts and news articles with unix timestamps and raw text
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


class RawSentimentExporter:
    """Export raw sentiment text with timestamps for LLM processing"""
    
    def __init__(self, newsapi_key: str = None, output_dir: str = "raw_sentiment_data"):
        self.newsapi_key = newsapi_key
        self.output_dir = output_dir
        self.start_date = "2025-01-01"
        self.end_date = "2025-01-31"
        os.makedirs(output_dir, exist_ok=True)
    
    def collect_reddit_data(self) -> List[Dict]:
        """Collect Reddit posts"""
        subreddits = ["Bitcoin", "cryptocurrency", "CryptoMarkets"]
        keywords = ["bitcoin", "btc", "crypto", "cryptocurrency"]
        all_posts = []
        
        for subreddit in subreddits:
            try:
                for sort_type in ['hot', 'new']:
                    url = f"https://www.reddit.com/r/{subreddit}/{sort_type}.json?limit=50"
                    headers = {'User-Agent': 'SentimentAnalysis/1.0'}
                    response = requests.get(url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        for post in data['data']['children']:
                            post_data = post['data']
                            title = post_data.get('title', '').lower()
                            text = post_data.get('selftext', '').lower()
                            
                            if any(keyword in title or keyword in text for keyword in keywords):
                                raw_content = f"{post_data.get('title', '')} {post_data.get('selftext', '')}".strip()
                                if len(raw_content) > 20:
                                    all_posts.append({
                                        'timestamp': int(post_data.get('created_utc', time.time())),
                                        'source': 'reddit',
                                        'raw_text': raw_content,
                                        'title': post_data.get('title', ''),
                                        'url': f"https://reddit.com{post_data.get('permalink', '')}"
                                    })
                    time.sleep(1)
            except:
                continue
        return all_posts
    
    def collect_news_data(self) -> List[Dict]:
        """Collect news articles"""
        if not self.newsapi_key:
            return []
        
        queries = ["bitcoin", "cryptocurrency", "BTC price"]
        all_articles = []
        
        for query in queries:
            try:
                params = {
                    'q': query,
                    'from': self.start_date,
                    'to': self.end_date,
                    'language': 'en',
                    'apiKey': self.newsapi_key,
                    'pageSize': 50
                }
                response = requests.get('https://newsapi.org/v2/everything', params=params, timeout=10)
                
                if response.status_code == 200:
                    articles = response.json().get('articles', [])
                    for article in articles:
                        try:
                            timestamp = int(pd.to_datetime(article.get('publishedAt')).timestamp())
                        except:
                            continue
                        
                        title = article.get('title', '')
                        description = article.get('description', '')
                        raw_content = f"{title} {description}".strip()
                        
                        if len(raw_content) > 20:
                            all_articles.append({
                                'timestamp': timestamp,
                                'source': 'news',
                                'raw_text': raw_content,
                                'title': title,
                                'url': article.get('url', '')
                            })
                time.sleep(1)
            except:
                continue
        return all_articles
    
    def add_date_info(self, posts: List[Dict]) -> List[Dict]:
        """Add readable date information"""
        for post in posts:
            dt = datetime.fromtimestamp(post['timestamp'])
            post['date'] = dt.strftime('%Y-%m-%d')
            post['datetime_readable'] = dt.strftime('%Y-%m-%d %H:%M:%S')
        return sorted(posts, key=lambda x: x['timestamp'])
    
    def export_csv(self) -> str:
        """Export sentiment data to CSV"""
        print("Collecting sentiment data...")
        
        reddit_posts = self.collect_reddit_data()
        news_articles = self.collect_news_data()
        
        all_data = reddit_posts + news_articles
        if not all_data:
            return None
        
        all_data = self.add_date_info(all_data)
        df = pd.DataFrame(all_data)
        
        filename = f"raw_sentiment_jan2025.csv"
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        
        print(f"Exported {len(df)} records to {filepath}")
        return filepath
    
    def create_sample(self, n_samples: int = 50) -> str:
        """Create sample data for testing"""
        reddit_samples = [
            "Bitcoin is looking bullish today! Price action suggests $50K soon.",
            "Market volatility has me worried. BTC dropped 5% overnight.",
            "Just bought more Bitcoin on the dip. DCA is the way!",
            "Crypto market is unpredictable. Mooning one day, crashing the next.",
            "Institutional adoption driving Bitcoin higher. This feels different."
        ]
        
        news_samples = [
            "Bitcoin Price Surges Above $45,000 as Institutional Investment Continues",
            "Crypto Market Faces Uncertainty Amid Regulatory Concerns",
            "Major Bank Announces Bitcoin Custody Services",
            "Bitcoin Mining Difficulty Reaches New High",
            "Federal Reserve Comments Impact Cryptocurrency Market Sentiment"
        ]
        
        start_ts = int(datetime(2025, 1, 1).timestamp())
        end_ts = int(datetime(2025, 1, 31, 23, 59, 59).timestamp())
        
        sample_data = []
        for i in range(n_samples):
            timestamp = np.random.randint(start_ts, end_ts)
            if i % 2 == 0:
                source, text = 'reddit', np.random.choice(reddit_samples)
            else:
                source, text = 'news', np.random.choice(news_samples)
            
            sample_data.append({
                'timestamp': timestamp,
                'date': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d'),
                'source': source,
                'raw_text': text,
                'title': text[:50] + "..." if len(text) > 50 else text,
                'url': f'https://example.com/{i}'
            })
        
        df = pd.DataFrame(sample_data).sort_values('timestamp')
        filename = f"raw_sentiment_sample_{n_samples}records.csv"
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False)
        
        print(f"Created sample: {filepath}")
        return filepath


def main():
    parser = argparse.ArgumentParser(description='Export raw sentiment data')
    parser.add_argument('--newsapi-key', help='NewsAPI key')
    parser.add_argument('--output-dir', default='raw_sentiment_data', help='Output directory')
    parser.add_argument('--create-sample', type=int, help='Create N sample records')
    args = parser.parse_args()
    
    # Load API key from config if available
    if not args.newsapi_key:
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
                args.newsapi_key = config.get('newsapi_key')
        except:
            pass
    
    exporter = RawSentimentExporter(newsapi_key=args.newsapi_key, output_dir=args.output_dir)
    
    if args.create_sample:
        exporter.create_sample(args.create_sample)
    else:
        exporter.export_csv()


if __name__ == "__main__":
    main()