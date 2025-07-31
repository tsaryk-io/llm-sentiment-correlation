#!/usr/bin/env python3
"""
LLM Sentiment Correlation Analysis
Analyzes correlation between LLM-generated sentiment and prediction accuracy
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.filterwarnings('ignore')


class SentimentCorrelationAnalyzer:
    """Analyze correlations between LLM sentiment and predictions"""
    
    def __init__(self, output_dir: str = 'correlation_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_sentiment_data(self, sentiment_file: str) -> pd.DataFrame:
        """Load LLM sentiment analysis results"""
        df = pd.read_csv(sentiment_file)
        required_cols = ['timestamp', 'llm_sentiment_score', 'llm_sentiment_label', 'source']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        print(f"Loaded {len(df)} sentiment records")
        if 'llm_model_used' in df.columns:
            models = df['llm_model_used'].unique()
            print(f"LLM models: {models.tolist()}")
        
        return df
    
    
    def align_data(self, sentiment_df: pd.DataFrame, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """Align sentiment and prediction data by timestamp"""
        # Aggregate sentiment by hour
        sentiment_df['hour'] = pd.to_datetime(sentiment_df['timestamp'], unit='s').dt.floor('H')
        hourly_sentiment = sentiment_df.groupby('hour').agg({
            'llm_sentiment_score': ['mean', 'std', 'count'],
            'llm_confidence': 'mean',
            'timestamp': 'first'
        }).reset_index()
        
        hourly_sentiment.columns = ['hour', 'avg_sentiment', 'sentiment_volatility', 'text_count', 'avg_confidence', 'timestamp']
        hourly_sentiment['timestamp'] = hourly_sentiment['hour'].astype(int) // 10**9
        
        # Merge with predictions
        aligned_df = pd.merge(prediction_df, hourly_sentiment[['timestamp', 'avg_sentiment', 'sentiment_volatility', 'avg_confidence']], 
                             on='timestamp', how='inner')
        
        print(f"Aligned {len(aligned_df)} records")
        return aligned_df
    
    def compute_correlations(self, aligned_df: pd.DataFrame) -> dict:
        """Compute correlations between sentiment and prediction metrics"""
        sentiment_vars = ['avg_sentiment', 'sentiment_volatility', 'avg_confidence']
        prediction_vars = ['prediction_accuracy', 'direction_accuracy', 'prediction_error']
        
        correlations = {'pearson': {}, 'spearman': {}}
        
        for sent_var in sentiment_vars:
            if sent_var not in aligned_df.columns:
                continue
            
            correlations['pearson'][sent_var] = {}
            correlations['spearman'][sent_var] = {}
            
            for pred_var in prediction_vars:
                if pred_var not in aligned_df.columns:
                    continue
                
                # Remove NaN values
                mask = ~(aligned_df[sent_var].isna() | aligned_df[pred_var].isna())
                if mask.sum() < 10:
                    continue
                
                x = aligned_df.loc[mask, sent_var]
                y = aligned_df.loc[mask, pred_var]
                
                try:
                    # Pearson correlation
                    pearson_r, pearson_p = pearsonr(x, y)
                    correlations['pearson'][sent_var][pred_var] = {
                        'correlation': pearson_r,
                        'p_value': pearson_p,
                        'n_samples': len(x),
                        'significant': pearson_p < 0.05
                    }
                    
                    # Spearman correlation
                    spearman_r, spearman_p = spearmanr(x, y)
                    correlations['spearman'][sent_var][pred_var] = {
                        'correlation': spearman_r,
                        'p_value': spearman_p,
                        'n_samples': len(x),
                        'significant': spearman_p < 0.05
                    }
                except:
                    pass
        
        # Print key results
        if ('avg_sentiment' in correlations['pearson'] and 
            'prediction_accuracy' in correlations['pearson']['avg_sentiment']):
            main_corr = correlations['pearson']['avg_sentiment']['prediction_accuracy']
            print(f"\nKey Result:")
            print(f"LLM Sentiment vs Prediction Accuracy: r = {main_corr['correlation']:.3f}")
            print(f"Statistical significance: {'YES' if main_corr['significant'] else 'NO'}")
            print(f"P-value: {main_corr['p_value']:.4f}")
        
        return correlations
    
    def create_visualizations(self, aligned_df: pd.DataFrame, correlations: dict):
        """Create correlation visualizations"""
        plt.style.use('default')
        
        # Main scatter plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('LLM Sentiment vs Prediction Metrics', fontsize=14)
        
        # Sentiment vs Prediction Accuracy
        if 'avg_sentiment' in aligned_df.columns and 'prediction_accuracy' in aligned_df.columns:
            axes[0,0].scatter(aligned_df['avg_sentiment'], aligned_df['prediction_accuracy'], alpha=0.6)
            axes[0,0].set_xlabel('LLM Sentiment Score')
            axes[0,0].set_ylabel('Prediction Accuracy')
            axes[0,0].set_title('Sentiment vs Accuracy')
            
            # Trend line
            z = np.polyfit(aligned_df['avg_sentiment'], aligned_df['prediction_accuracy'], 1)
            p = np.poly1d(z)
            axes[0,0].plot(aligned_df['avg_sentiment'], p(aligned_df['avg_sentiment']), "r--", alpha=0.8)
        
        # Sentiment vs Direction Accuracy
        if 'avg_sentiment' in aligned_df.columns and 'direction_accuracy' in aligned_df.columns:
            axes[0,1].scatter(aligned_df['avg_sentiment'], aligned_df['direction_accuracy'], alpha=0.6, color='orange')
            axes[0,1].set_xlabel('LLM Sentiment Score')
            axes[0,1].set_ylabel('Direction Accuracy')
            axes[0,1].set_title('Sentiment vs Direction')
        
        # Sentiment Volatility vs Error
        if 'sentiment_volatility' in aligned_df.columns and 'prediction_error' in aligned_df.columns:
            axes[1,0].scatter(aligned_df['sentiment_volatility'], aligned_df['prediction_error'], alpha=0.6, color='green')
            axes[1,0].set_xlabel('Sentiment Volatility')
            axes[1,0].set_ylabel('Prediction Error (%)')
            axes[1,0].set_title('Volatility vs Error')
        
        # Time series
        if 'timestamp' in aligned_df.columns:
            aligned_df['datetime'] = pd.to_datetime(aligned_df['timestamp'], unit='s')
            aligned_df_sorted = aligned_df.sort_values('datetime')
            
            ax2 = axes[1,1]
            ax3 = ax2.twinx()
            
            ax2.plot(aligned_df_sorted['datetime'], aligned_df_sorted['avg_sentiment'], 'b-', alpha=0.7, label='Sentiment')
            ax3.plot(aligned_df_sorted['datetime'], aligned_df_sorted['prediction_accuracy'], 'r-', alpha=0.7, label='Accuracy')
            
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Sentiment', color='b')
            ax3.set_ylabel('Accuracy', color='r')
            ax2.set_title('Time Series')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/sentiment_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Correlation heatmap
        correlation_data = []
        for method in ['pearson']:
            for sent_var, pred_vars in correlations[method].items():
                for pred_var, stats in pred_vars.items():
                    correlation_data.append({
                        'Sentiment': sent_var,
                        'Prediction': pred_var,
                        'Correlation': stats['correlation']
                    })
        
        if correlation_data:
            corr_df = pd.DataFrame(correlation_data)
            pivot_df = corr_df.pivot(index='Sentiment', columns='Prediction', values='Correlation')
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot_df, annot=True, cmap='RdBu_r', center=0, fmt='.3f')
            plt.title('Sentiment-Prediction Correlations')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {self.output_dir}/")
    
    def generate_report(self, aligned_df: pd.DataFrame, correlations: dict) -> str:
        """Generate summary report"""
        report_lines = [
            "=" * 60,
            "LLM SENTIMENT CORRELATION ANALYSIS REPORT",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total records: {len(aligned_df)}",
            ""
        ]
        
        # Key findings
        report_lines.append("KEY FINDINGS:")
        report_lines.append("-" * 30)
        
        if ('avg_sentiment' in correlations['pearson'] and 
            'prediction_accuracy' in correlations['pearson']['avg_sentiment']):
            main_corr = correlations['pearson']['avg_sentiment']['prediction_accuracy']
            significance = "SIGNIFICANT" if main_corr['significant'] else "NOT SIGNIFICANT"
            
            report_lines.extend([
                f"LLM Sentiment vs Prediction Accuracy:",
                f"  Correlation: r = {main_corr['correlation']:.3f}",
                f"  P-value: {main_corr['p_value']:.4f}",
                f"  Statistical significance: {significance}",
                f"  Sample size: {main_corr['n_samples']}",
                ""
            ])
        
        # Other correlations
        report_lines.append("ALL CORRELATIONS:")
        report_lines.append("-" * 30)
        for sent_var, pred_vars in correlations['pearson'].items():
            for pred_var, stats in pred_vars.items():
                sig = "*" if stats['significant'] else " "
                report_lines.append(f"{sig} {sent_var} vs {pred_var}: r = {stats['correlation']:.3f}")
        
        report_text = "\n".join(report_lines)
        
        report_file = os.path.join(self.output_dir, 'correlation_report.txt')
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"Report saved: {report_file}")
        return report_text
    
    def run_analysis(self, sentiment_file: str, prediction_file: str):
        """Run complete correlation analysis"""
        print("Starting LLM Sentiment Correlation Analysis")
        print("=" * 50)
        
        # Load sentiment data
        sentiment_df = self.load_sentiment_data(sentiment_file)
        
        # Load prediction data
        if not os.path.exists(prediction_file):
            raise FileNotFoundError(f"Prediction file not found: {prediction_file}")
        prediction_df = pd.read_csv(prediction_file)
        
        # Align data
        aligned_df = self.align_data(sentiment_df, prediction_df)
        
        # Compute correlations
        correlations = self.compute_correlations(aligned_df)
        
        # Create visualizations
        self.create_visualizations(aligned_df, correlations)
        
        # Generate report
        self.generate_report(aligned_df, correlations)
        
        # Save detailed results
        aligned_df.to_csv(os.path.join(self.output_dir, 'detailed_results.csv'), index=False)
        
        print(f"\nAnalysis complete! Results in: {self.output_dir}")
        
        return {'aligned_data': aligned_df, 'correlations': correlations}


def main():
    parser = argparse.ArgumentParser(description='LLM Sentiment Correlation Analysis')
    parser.add_argument('--sentiment-file', required=True, help='LLM sentiment analysis CSV file')
    parser.add_argument('--prediction-file', required=True, help='Prediction data CSV file')
    parser.add_argument('--output-dir', default='correlation_results', help='Output directory')
    
    args = parser.parse_args()
    
    analyzer = SentimentCorrelationAnalyzer(output_dir=args.output_dir)
    results = analyzer.run_analysis(
        sentiment_file=args.sentiment_file,
        prediction_file=args.prediction_file
    )
    
    # Print final result
    correlations = results['correlations']
    if ('avg_sentiment' in correlations['pearson'] and 
        'prediction_accuracy' in correlations['pearson']['avg_sentiment']):
        main_corr = correlations['pearson']['avg_sentiment']['prediction_accuracy']
        print(f"\nFINAL RESULT:")
        print(f"LLM Sentiment vs Prediction Accuracy: r = {main_corr['correlation']:.3f}")
        print(f"Statistical significance: {'YES' if main_corr['significant'] else 'NO'}")


if __name__ == "__main__":
    main()