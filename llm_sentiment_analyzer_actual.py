#!/usr/bin/env python3
"""
LLM Sentiment Analysis
Uses actual LLMs (DEEPSEEK, LLAMA3.1, QWEN, MISTRAL, GEMMA) for sentiment analysis
"""

import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import argparse
from typing import Dict, List
from tqdm import tqdm
import warnings
import re

warnings.filterwarnings('ignore')


class LLMSentimentAnalyzer:
    """Analyze sentiment using actual LLMs"""
    
    def __init__(self, llm_model: str = 'DEEPSEEK', device: str = 'auto'):
        self.llm_model = llm_model.upper()
        self.device = self._get_device(device)
        
        self.model_configs = {
            'DEEPSEEK': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
            'LLAMA3.1': 'meta-llama/Llama-3.1-8B-Instruct',
            'QWEN': 'Qwen/Qwen2.5-7B-Instruct',
            'MISTRAL': 'mistralai/Mistral-7B-Instruct-v0.3',
            'GEMMA': 'google/gemma-2-2b-it',
            'LLAMA': 'huggyllama/llama-7b'
        }
        
        print(f"Loading {self.llm_model}...")
        self._load_model()
    
    def _get_device(self, device: str) -> torch.device:
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def _load_model(self):
        if self.llm_model not in self.model_configs:
            raise ValueError(f"Model {self.llm_model} not supported")
        
        model_name = self.model_configs[self.llm_model]
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map="auto" if self.device.type == 'cuda' else None,
                trust_remote_code=True
            )
            
            if self.device.type != 'cuda':
                self.model = self.model.to(self.device)
            
            self.model.eval()
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def create_prompt(self, text: str) -> str:
        text = text.strip()[:500]
        return f"""Analyze the sentiment of this cryptocurrency text. Respond with ONLY a number between 0 and 1:
- 0.0 = Very Negative
- 0.5 = Neutral  
- 1.0 = Very Positive

Text: "{text}"

Sentiment score:"""
    
    def extract_score(self, response: str) -> float:
        # Find decimal between 0-1
        matches = re.findall(r'\b0?\.\d+\b|\b[01]\.?\d*\b', response)
        for match in matches:
            try:
                score = float(match)
                if 0 <= score <= 1:
                    return score
            except:
                continue
        
        # Find integers
        digit_matches = re.findall(r'\b\d+\b', response)
        for match in digit_matches:
            try:
                score = float(match)
                if score <= 1:
                    return score
                elif score <= 10:
                    return score / 10
                elif score <= 100:
                    return score / 100
            except:
                continue
        
        # Fallback to keyword analysis
        response_lower = response.lower()
        if any(word in response_lower for word in ['very positive', 'bullish', 'excellent']):
            return 0.8
        elif any(word in response_lower for word in ['positive', 'good', 'optimistic']):
            return 0.7
        elif any(word in response_lower for word in ['neutral', 'mixed']):
            return 0.5
        elif any(word in response_lower for word in ['negative', 'bad']):
            return 0.3
        elif any(word in response_lower for word in ['very negative', 'bearish']):
            return 0.2
        
        return 0.5
    
    def analyze_text(self, text: str) -> Dict:
        if len(text.strip()) < 10:
            return {'sentiment_score': 0.5, 'sentiment_label': 'neutral', 'confidence': 0.0}
        
        try:
            prompt = self.create_prompt(text)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                                  max_length=512, padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.3,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            sentiment_score = self.extract_score(response)
            
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
                'raw_response': response
            }
            
        except Exception as e:
            return {'sentiment_score': 0.5, 'sentiment_label': 'neutral', 
                   'confidence': 0.0, 'error': str(e)}
    
    def process_file(self, csv_file: str, output_file: str = None) -> pd.DataFrame:
        print(f"Processing {csv_file} with {self.llm_model}...")
        
        df = pd.read_csv(csv_file)
        required_cols = ['timestamp', 'raw_text', 'source']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        results = []
        for text in tqdm(df['raw_text'], desc=f"Analyzing with {self.llm_model}"):
            result = self.analyze_text(text)
            results.append(result)
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Add results to dataframe
        df['llm_sentiment_score'] = [r['sentiment_score'] for r in results]
        df['llm_sentiment_label'] = [r['sentiment_label'] for r in results]
        df['llm_confidence'] = [r['confidence'] for r in results]
        df['llm_model_used'] = self.llm_model
        df['processing_timestamp'] = int(datetime.now().timestamp())
        
        if 'datetime_readable' not in df.columns:
            df['datetime_readable'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save results
        if output_file is None:
            base_name = os.path.splitext(csv_file)[0]
            output_file = f"{base_name}_{self.llm_model.lower()}_analyzed.csv"
        
        df.to_csv(output_file, index=False)
        
        print(f"\nAnalysis complete: {output_file}")
        print(f"Records processed: {len(df)}")
        
        # Summary
        sentiment_counts = df['llm_sentiment_label'].value_counts()
        for label, count in sentiment_counts.items():
            print(f"{label.capitalize()}: {count} ({count/len(df)*100:.1f}%)")
        
        print(f"Average sentiment: {df['llm_sentiment_score'].mean():.3f}")
        print(f"Average confidence: {df['llm_confidence'].mean():.3f}")
        
        return df


def main():
    parser = argparse.ArgumentParser(description='LLM sentiment analysis')
    parser.add_argument('--input-file', required=True, help='Input CSV file')
    parser.add_argument('--output-file', help='Output CSV file')
    parser.add_argument('--llm-model', default='DEEPSEEK', 
                       choices=['LLAMA', 'LLAMA3.1', 'QWEN', 'MISTRAL', 'DEEPSEEK', 'GEMMA'],
                       help='LLM model to use')
    parser.add_argument('--device', default='auto', help='Device (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    analyzer = LLMSentimentAnalyzer(llm_model=args.llm_model, device=args.device)
    analyzer.process_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()