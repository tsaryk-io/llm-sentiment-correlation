#!/usr/bin/env python3
"""
LLM-Based Sentiment Analysis using Actual LLMs - STANDALONE VERSION
Uses LLAMA3.1, LLAMA, QWEN, MISTRAL, DEEPSEEK, GEMMA for sentiment analysis

Dependencies: Only transformers, torch, pandas, numpy, tqdm
No dependencies on the Cryptex project
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    LlamaTokenizer, LlamaModel, 
    GPT2Tokenizer, BertTokenizer
)
from datetime import datetime
import argparse
from typing import Dict, List, Tuple, Optional, Union
import json
from tqdm import tqdm
import warnings
from dataclasses import dataclass
import re

warnings.filterwarnings('ignore')


class ActualLLMSentimentAnalyzer:
    """
    Analyze sentiment using actual LLMs - STANDALONE VERSION
    Compatible with LLAMA3.1, LLAMA, QWEN, MISTRAL, DEEPSEEK, GEMMA
    """
    
    def __init__(self, llm_model: str = 'DEEPSEEK', device: str = 'auto', batch_size: int = 4):
        self.llm_model = llm_model.upper()
        self.device = self._get_device(device)
        self.batch_size = batch_size
        
        # Model configurations - STANDALONE (no dependency on TimeLLM.py)
        self.model_configs = {
            'LLAMA': {
                'tokenizer_class': LlamaTokenizer,
                'model_class': AutoModelForCausalLM,
                'model_name': 'huggyllama/llama-7b'
            },
            'DEEPSEEK': {
                'tokenizer_class': AutoTokenizer,
                'model_class': AutoModelForCausalLM,
                'model_name': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
            },
            'QWEN': {
                'tokenizer_class': AutoTokenizer,
                'model_class': AutoModelForCausalLM,
                'model_name': 'Qwen/Qwen2.5-7B-Instruct'
            },
            'MISTRAL': {
                'tokenizer_class': AutoTokenizer,
                'model_class': AutoModelForCausalLM,
                'model_name': 'mistralai/Mistral-7B-Instruct-v0.3'
            },
            'GEMMA': {
                'tokenizer_class': AutoTokenizer,
                'model_class': AutoModelForCausalLM,
                'model_name': 'google/gemma-2-2b-it'
            },
            'LLAMA3.1': {
                'tokenizer_class': AutoTokenizer,
                'model_class': AutoModelForCausalLM,
                'model_name': 'meta-llama/Llama-3.1-8B-Instruct'
            }
        }
        
        print(f"Initializing {self.llm_model} for sentiment analysis...")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {batch_size}")
        print("  STANDALONE VERSION - No Cryptex dependencies")
        
        self._load_model()
    
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate PyTorch device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')  # Apple Silicon
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def _load_model(self):
        """Load the selected LLM model and tokenizer"""
        if self.llm_model not in self.model_configs:
            raise ValueError(f"Model {self.llm_model} not supported. Available: {list(self.model_configs.keys())}")
        
        model_config = self.model_configs[self.llm_model]
        model_name = model_config['model_name']
        
        try:
            print(f"  Loading tokenizer: {model_name}")
            self.tokenizer = model_config['tokenizer_class'].from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"  Loading model: {model_name}")
            self.model = model_config['model_class'].from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map="auto" if self.device.type == 'cuda' else None,
                trust_remote_code=True
            )
            
            if self.device.type != 'cuda':
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print("  ✓ Model loaded successfully")
            
        except Exception as e:
            print(f"  ✗ Error loading model: {e}")
            print("  Falling back to smaller model...")
            self._fallback_to_smaller_model()
    
    def _fallback_to_smaller_model(self):
        """Fallback to a smaller model if main model fails"""
        try:
            print("  Trying GPT-2 as fallback...")
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.model = self.model.to(self.device)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm_model = "GPT2_FALLBACK"
            print("  ✓ Fallback model loaded")
        except Exception as e:
            raise RuntimeError(f"Could not load any model: {e}")
    
    def create_sentiment_prompt(self, text: str) -> str:
        """
        Create a prompt for sentiment analysis
        """
        # Clean the text
        text = text.strip()[:500]  # Limit length
        
        # Create a clear sentiment analysis prompt
        prompt = f"""Analyze the sentiment of this cryptocurrency-related text. Respond with ONLY a number between 0 and 1, where:
- 0.0 = Very Negative
- 0.5 = Neutral  
- 1.0 = Very Positive

Text: "{text}"

Sentiment score:"""
        
        return prompt
    
    def extract_sentiment_score(self, response: str) -> float:
        """
        Extract numerical sentiment score from LLM response
        """
        # Try to find a number between 0 and 1
        import re
        
        # Look for decimal numbers between 0 and 1
        matches = re.findall(r'\b0?\.\d+\b|\b[01]\.?\d*\b', response)
        
        for match in matches:
            try:
                score = float(match)
                if 0 <= score <= 1:
                    return score
            except:
                continue
        
        # Look for just digits that could be interpreted as scores
        digit_matches = re.findall(r'\b\d+\b', response)
        for match in digit_matches:
            try:
                score = float(match)
                if score <= 1:
                    return score
                elif score <= 10:  # Scale from 1-10 to 0-1
                    return score / 10
                elif score <= 100:  # Scale from 1-100 to 0-1
                    return score / 100
            except:
                continue
        
        # Fallback: look for sentiment words
        response_lower = response.lower()
        if any(word in response_lower for word in ['very positive', 'bullish', 'great', 'excellent']):
            return 0.8
        elif any(word in response_lower for word in ['positive', 'good', 'optimistic']):
            return 0.7
        elif any(word in response_lower for word in ['neutral', 'mixed', 'uncertain']):
            return 0.5
        elif any(word in response_lower for word in ['negative', 'bad', 'pessimistic']):
            return 0.3
        elif any(word in response_lower for word in ['very negative', 'bearish', 'terrible']):
            return 0.2
        
        # Default neutral
        return 0.5
    
    def analyze_single_text(self, text: str) -> Dict:
        """Analyze sentiment for a single text using the LLM"""
        
        if len(text.strip()) < 10:
            return {
                'sentiment_score': 0.5,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'raw_response': '',
                'error': 'Text too short'
            }
        
        try:
            # Create prompt
            prompt = self.create_sentiment_prompt(text)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.3,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated part (after the prompt)
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            # Extract sentiment score
            sentiment_score = self.extract_sentiment_score(response)
            
            # Determine label and confidence
            if sentiment_score > 0.6:
                sentiment_label = 'positive'
                confidence = min((sentiment_score - 0.6) * 2.5, 1.0)
            elif sentiment_score < 0.4:
                sentiment_label = 'negative'
                confidence = min((0.4 - sentiment_score) * 2.5, 1.0)
            else:
                sentiment_label = 'neutral'
                confidence = 1.0 - abs(sentiment_score - 0.5) * 2
            
            return {
                'sentiment_score': float(sentiment_score),
                'sentiment_label': sentiment_label,
                'confidence': float(confidence),
                'raw_response': response,
                'llm_model': self.llm_model
            }
            
        except Exception as e:
            print(f"    Error analyzing text: {e}")
            return {
                'sentiment_score': 0.5,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'raw_response': '',
                'error': str(e),
                'llm_model': self.llm_model
            }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze sentiment for a batch of texts"""
        
        results = []
        
        # Process each text individually for better control
        for text in tqdm(texts, desc=f"Analyzing with {self.llm_model}"):
            result = self.analyze_single_text(text)
            results.append(result)
            
            # Small delay to prevent overwhelming the model
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return results
    
    def process_raw_sentiment_file(self, csv_file: str, output_file: str = None) -> pd.DataFrame:
        """
        Process raw sentiment CSV file using actual LLM
        """
        print(f"Processing raw sentiment file with {self.llm_model}: {csv_file}")
        
        # Load raw sentiment data
        df = pd.read_csv(csv_file)
        print(f"  Loaded {len(df)} records")
        
        # Validate required columns
        required_cols = ['timestamp', 'raw_text', 'source']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Process texts
        print(f"  Analyzing sentiment using {self.llm_model}...")
        texts = df['raw_text'].tolist()
        
        # Process in smaller batches to manage memory
        all_results = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            print(f"  Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
            
            batch_results = self.analyze_batch(batch_texts)
            all_results.extend(batch_results)
        
        # Add results to dataframe
        df['llm_sentiment_score'] = [r['sentiment_score'] for r in all_results]
        df['llm_sentiment_label'] = [r['sentiment_label'] for r in all_results]
        df['llm_confidence'] = [r['confidence'] for r in all_results]
        df['llm_raw_response'] = [r.get('raw_response', '') for r in all_results]
        df['llm_model_used'] = [r.get('llm_model', self.llm_model) for r in all_results]
        df['processing_timestamp'] = int(datetime.now().timestamp())
        
        # Add readable datetime
        df['datetime_readable'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate additional metrics
        df['sentiment_strength'] = abs(df['llm_sentiment_score'] - 0.5) * 2
        df['is_positive'] = (df['llm_sentiment_score'] > 0.6).astype(int)
        df['is_negative'] = (df['llm_sentiment_score'] < 0.4).astype(int)
        df['is_neutral'] = ((df['llm_sentiment_score'] >= 0.4) & (df['llm_sentiment_score'] <= 0.6)).astype(int)
        
        # Save results
        if output_file is None:
            base_name = os.path.splitext(csv_file)[0]
            output_file = f"{base_name}_{self.llm_model.lower()}_analyzed.csv"
        
        df.to_csv(output_file, index=False)
        
        # Print summary
        print(f"\n✅ {self.llm_model} Sentiment Analysis Complete!")
        print(f"  Output file: {output_file}")
        print(f"  Records processed: {len(df)}")
        
        print(f"\nSentiment Distribution:")
        sentiment_counts = df['llm_sentiment_label'].value_counts()
        for label, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {label.capitalize()}: {count} ({percentage:.1f}%)")
        
        print(f"\nAverage Sentiment Score: {df['llm_sentiment_score'].mean():.3f}")
        print(f"Average Confidence: {df['llm_confidence'].mean():.3f}")
        
        # Source breakdown
        print(f"\nBy Source:")
        for source in df['source'].unique():
            source_data = df[df['source'] == source]
            avg_sentiment = source_data['llm_sentiment_score'].mean()
            print(f"  {source.capitalize()}: {len(source_data)} records, avg sentiment: {avg_sentiment:.3f}")
        
        # Show some example responses
        print(f"\nExample {self.llm_model} Responses:")
        for i, (_, row) in enumerate(df.head(3).iterrows()):
            print(f"  {i+1}. Text: {row['raw_text'][:100]}...")
            print(f"     Score: {row['llm_sentiment_score']:.3f} ({row['llm_sentiment_label']})")
            print(f"     LLM Response: {row['llm_raw_response'][:100]}...")
            print()
        
        return df
    
    def create_correlation_ready_dataset(self, analyzed_df: pd.DataFrame) -> pd.DataFrame:
        """Create dataset ready for correlation analysis"""
        print("Creating correlation-ready dataset...")
        
        # Create hourly aggregations
        analyzed_df['hour'] = pd.to_datetime(analyzed_df['timestamp'], unit='s').dt.floor('H')
        
        # Aggregate by hour
        hourly_sentiment = analyzed_df.groupby('hour').agg({
            'llm_sentiment_score': ['mean', 'std', 'count'],
            'llm_confidence': 'mean',
            'sentiment_strength': 'mean',
            'is_positive': 'sum',
            'is_negative': 'sum',
            'is_neutral': 'sum',
            'timestamp': 'first'
        }).reset_index()
        
        # Flatten column names
        hourly_sentiment.columns = [
            'hour', 'avg_sentiment_score', 'sentiment_volatility', 'text_count',
            'avg_confidence', 'avg_sentiment_strength', 'positive_count',
            'negative_count', 'neutral_count', 'timestamp'
        ]
        
        # Calculate percentages
        hourly_sentiment['positive_ratio'] = hourly_sentiment['positive_count'] / hourly_sentiment['text_count']
        hourly_sentiment['negative_ratio'] = hourly_sentiment['negative_count'] / hourly_sentiment['text_count']
        hourly_sentiment['neutral_ratio'] = hourly_sentiment['neutral_count'] / hourly_sentiment['text_count']
        
        # Use hour timestamp for alignment
        hourly_sentiment['timestamp'] = hourly_sentiment['hour'].astype(int) // 10**9
        
        # Add readable datetime and model info
        hourly_sentiment['datetime_readable'] = hourly_sentiment['hour'].dt.strftime('%Y-%m-%d %H:%M:%S')
        hourly_sentiment['llm_model'] = self.llm_model
        
        # Fill NaN values
        hourly_sentiment = hourly_sentiment.fillna(0)
        
        print(f"  ✓ Created {len(hourly_sentiment)} hourly sentiment records")
        
        return hourly_sentiment


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Actual LLM-based sentiment analysis - STANDALONE')
    parser.add_argument('--input-file', required=True, help='Input CSV with raw sentiment data')
    parser.add_argument('--output-file', help='Output CSV file')
    parser.add_argument('--llm-model', default='DEEPSEEK', 
                       choices=['LLAMA', 'LLAMA3.1', 'QWEN', 'MISTRAL', 'DEEPSEEK', 'GEMMA'],
                       help='LLM model to use')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--device', default='auto', help='Device (auto, cpu, cuda, mps)')
    parser.add_argument('--create-correlation-dataset', action='store_true',
                       help='Create hourly aggregated dataset for correlation analysis')
    
    args = parser.parse_args()
    
    print("🚀 Starting Actual LLM Sentiment Analysis")
    print("Using LLMs: LLAMA3.1, QWEN, DEEPSEEK, MISTRAL, GEMMA")
    print("STANDALONE VERSION - No project dependencies")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = ActualLLMSentimentAnalyzer(
        llm_model=args.llm_model,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Process file
    analyzed_df = analyzer.process_raw_sentiment_file(args.input_file, args.output_file)
    
    # Create correlation dataset if requested
    if args.create_correlation_dataset:
        correlation_df = analyzer.create_correlation_ready_dataset(analyzed_df)
        
        base_name = os.path.splitext(args.output_file or args.input_file)[0]
        correlation_file = f"{base_name}_correlation_ready.csv"
        correlation_df.to_csv(correlation_file, index=False)
        
        print(f"\n📊 Correlation dataset created: {correlation_file}")
        print("Ready for correlation analysis with prediction data!")


if __name__ == "__main__":
    main()