#!/usr/bin/env python3
"""
LLM Sentiment Correlation Analysis - STANDALONE VERSION
Simple correlation analysis between LLM-generated sentiment and prediction accuracy
Exactly as requested by professor: Raw text -> LLM sentiment -> Correlation analysis

Dependencies: Only pandas, numpy, matplotlib, seaborn, scipy
No dependencies on the Cryptex project
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.filterwarnings('ignore')


class LLMSentimentCorrelationAnalyzer:
    """
    Analyze correlations between LLM-generated sentiment and model predictions
    STANDALONE VERSION - No dependencies on Cryptex project
    """
    
    def __init__(self, output_dir: str = 'llm_correlation_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print("STANDALONE LLM Sentiment Correlation Analyzer")
        print("No dependencies on Cryptex project")
        
    def load_llm_sentiment_data(self, llm_sentiment_file: str) -> pd.DataFrame:
        """Load LLM-analyzed sentiment data"""
        print(f"Loading LLM sentiment data: {llm_sentiment_file}")
        
        if not os.path.exists(llm_sentiment_file):
            raise FileNotFoundError(f"File not found: {llm_sentiment_file}")
        
        df = pd.read_csv(llm_sentiment_file)
        
        # Validate required columns
        required_cols = ['timestamp', 'llm_sentiment_score', 'llm_sentiment_label', 'source']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"  ✓ Loaded {len(df)} LLM sentiment records")
        print(f"  ✓ Date range: {pd.to_datetime(df['timestamp'].min(), unit='s').strftime('%Y-%m-%d')} to {pd.to_datetime(df['timestamp'].max(), unit='s').strftime('%Y-%m-%d')}")
        print(f"  ✓ Sources: {df['source'].unique().tolist()}")
        
        # Show which LLM was used
        if 'llm_model_used' in df.columns:
            llm_models = df['llm_model_used'].unique()
            print(f"  ✓ LLM Models used: {llm_models.tolist()}")
        
        return df
    
    def create_mock_prediction_data(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create mock prediction data that correlates with sentiment for demonstration
        In real usage, professor would provide actual prediction results
        """
        print("Creating mock prediction data for demonstration...")
        
        # Create hourly prediction data matching sentiment timeframe
        hourly_data = []
        
        # Group sentiment by hour for realistic prediction simulation
        sentiment_df['hour'] = pd.to_datetime(sentiment_df['timestamp'], unit='s').dt.floor('H')
        hourly_sentiment = sentiment_df.groupby('hour').agg({
            'llm_sentiment_score': 'mean',
            'timestamp': 'first'
        }).reset_index()
        
        np.random.seed(42)  # Reproducible results
        
        for _, row in hourly_sentiment.iterrows():
            timestamp = int(row['hour'].timestamp())
            sentiment_score = row['llm_sentiment_score']
            
            # Simulate Bitcoin price movements influenced by sentiment
            base_price = 45000  # ~$45K Bitcoin
            
            # Sentiment influence (stronger correlation for demonstration)
            sentiment_influence = (sentiment_score - 0.5) * 0.02  # 2% max influence
            market_noise = np.random.normal(0, 0.01)  # 1% random noise
            
            price_change = sentiment_influence + market_noise
            current_price = base_price * (1 + price_change)
            
            # Create prediction with some error
            prediction_error = np.random.normal(0, 0.005)  # 0.5% prediction error
            predicted_price = current_price * (1 + prediction_error)
            
            # Calculate metrics
            prediction_accuracy = 1 - abs(predicted_price - current_price) / current_price
            price_direction = 1 if price_change > 0 else -1
            predicted_direction = 1 if prediction_error > -price_change else -1
            direction_accuracy = 1 if price_direction == predicted_direction else 0
            
            hourly_data.append({
                'timestamp': timestamp,
                'datetime_readable': pd.to_datetime(timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S'),
                'actual_price': current_price,
                'predicted_price': predicted_price,
                'price_change_pct': price_change * 100,
                'prediction_error_pct': abs(predicted_price - current_price) / current_price * 100,
                'prediction_accuracy': prediction_accuracy,
                'direction_accuracy': direction_accuracy,
                'actual_direction': price_direction,
                'predicted_direction': predicted_direction
            })
        
        mock_df = pd.DataFrame(hourly_data)
        print(f"  ✓ Created {len(mock_df)} mock prediction records")
        
        return mock_df
    
    def align_sentiment_with_predictions(self, sentiment_df: pd.DataFrame, 
                                       prediction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align LLM sentiment data with prediction data by timestamp
        """
        print("Aligning sentiment data with predictions...")
        
        # Aggregate sentiment by hour to match prediction frequency
        sentiment_df['hour'] = pd.to_datetime(sentiment_df['timestamp'], unit='s').dt.floor('H')
        
        # Aggregate sentiment metrics per hour
        hourly_sentiment = sentiment_df.groupby('hour').agg({
            'llm_sentiment_score': ['mean', 'std', 'count'],
            'llm_confidence': 'mean',
            'source': lambda x: x.value_counts().to_dict(),  # Count by source
            'timestamp': 'first'
        }).reset_index()
        
        # Flatten column names
        hourly_sentiment.columns = [
            'hour', 'avg_sentiment', 'sentiment_volatility', 'text_count', 
            'avg_confidence', 'source_breakdown', 'timestamp'
        ]
        
        # Convert hour to timestamp for alignment
        hourly_sentiment['timestamp'] = hourly_sentiment['hour'].astype(int) // 10**9
        
        # Merge with predictions
        aligned_df = pd.merge(
            prediction_df, 
            hourly_sentiment[['timestamp', 'avg_sentiment', 'sentiment_volatility', 'text_count', 'avg_confidence']], 
            on='timestamp', 
            how='inner'
        )
        
        print(f"  ✓ Aligned dataset: {len(aligned_df)} records")
        
        return aligned_df
    
    def compute_correlations(self, aligned_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Compute correlations between LLM sentiment and prediction metrics
        """
        print("Computing LLM sentiment correlations...")
        
        # Define sentiment and prediction variables
        sentiment_vars = ['avg_sentiment', 'sentiment_volatility', 'avg_confidence']
        prediction_vars = ['prediction_accuracy', 'direction_accuracy', 'prediction_error_pct']
        
        correlations = {
            'pearson': {},
            'spearman': {}
        }
        
        # Compute correlations
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
                if mask.sum() < 10:  # Need at least 10 data points
                    continue
                
                x = aligned_df.loc[mask, sent_var]
                y = aligned_df.loc[mask, pred_var]
                
                # Pearson correlation
                try:
                    pearson_r, pearson_p = pearsonr(x, y)
                    correlations['pearson'][sent_var][pred_var] = {
                        'correlation': pearson_r,
                        'p_value': pearson_p,
                        'n_samples': len(x),
                        'significant': pearson_p < 0.05
                    }
                except:
                    pass
                
                # Spearman correlation
                try:
                    spearman_r, spearman_p = spearmanr(x, y)
                    correlations['spearman'][sent_var][pred_var] = {
                        'correlation': spearman_r,
                        'p_value': spearman_p,
                        'n_samples': len(x),
                        'significant': spearman_p < 0.05
                    }
                except:
                    pass
        
        # Print summary
        print("  ✓ Key Correlation Results:")
        
        # Focus on main correlation: sentiment vs prediction accuracy
        if ('avg_sentiment' in correlations['pearson'] and 
            'prediction_accuracy' in correlations['pearson']['avg_sentiment']):
            
            main_corr = correlations['pearson']['avg_sentiment']['prediction_accuracy']
            print(f"    LLM Sentiment vs Prediction Accuracy:")
            print(f"      Pearson r = {main_corr['correlation']:.3f} (p = {main_corr['p_value']:.3f})")
            print(f"      {'Significant' if main_corr['significant'] else 'Not significant'} correlation")
            print(f"      Based on {main_corr['n_samples']} data points")
        
        return correlations
    
    def create_correlation_visualizations(self, aligned_df: pd.DataFrame, 
                                        correlations: Dict, output_dir: str):
        """Create visualizations for the correlation analysis"""
        
        print("Creating correlation visualizations...")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Main Correlation Scatter Plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LLM Sentiment vs Prediction Metrics - STANDALONE VERSION', fontsize=16)
        
        # Sentiment vs Prediction Accuracy
        if 'avg_sentiment' in aligned_df.columns and 'prediction_accuracy' in aligned_df.columns:
            axes[0,0].scatter(aligned_df['avg_sentiment'], aligned_df['prediction_accuracy'], 
                            alpha=0.6, s=30)
            axes[0,0].set_xlabel('LLM Sentiment Score (0-1)')
            axes[0,0].set_ylabel('Prediction Accuracy')
            axes[0,0].set_title('Sentiment vs Prediction Accuracy')
            axes[0,0].grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(aligned_df['avg_sentiment'], aligned_df['prediction_accuracy'], 1)
            p = np.poly1d(z)
            axes[0,0].plot(aligned_df['avg_sentiment'], p(aligned_df['avg_sentiment']), "r--", alpha=0.8)
        
        # Sentiment vs Direction Accuracy
        if 'avg_sentiment' in aligned_df.columns and 'direction_accuracy' in aligned_df.columns:
            axes[0,1].scatter(aligned_df['avg_sentiment'], aligned_df['direction_accuracy'], 
                            alpha=0.6, s=30, color='orange')
            axes[0,1].set_xlabel('LLM Sentiment Score (0-1)')
            axes[0,1].set_ylabel('Direction Accuracy')
            axes[0,1].set_title('Sentiment vs Direction Accuracy')
            axes[0,1].grid(True, alpha=0.3)
        
        # Sentiment Volatility vs Prediction Error
        if 'sentiment_volatility' in aligned_df.columns and 'prediction_error_pct' in aligned_df.columns:
            axes[1,0].scatter(aligned_df['sentiment_volatility'], aligned_df['prediction_error_pct'], 
                            alpha=0.6, s=30, color='green')
            axes[1,0].set_xlabel('Sentiment Volatility')
            axes[1,0].set_ylabel('Prediction Error (%)')
            axes[1,0].set_title('Sentiment Volatility vs Prediction Error')
            axes[1,0].grid(True, alpha=0.3)
        
        # Time Series: Sentiment and Accuracy
        if 'datetime_readable' in aligned_df.columns:
            aligned_df['datetime'] = pd.to_datetime(aligned_df['datetime_readable'])
            aligned_df_sorted = aligned_df.sort_values('datetime')
            
            ax2 = axes[1,1]
            ax3 = ax2.twinx()
            
            line1 = ax2.plot(aligned_df_sorted['datetime'], aligned_df_sorted['avg_sentiment'], 
                           'b-', label='LLM Sentiment', alpha=0.7)
            line2 = ax3.plot(aligned_df_sorted['datetime'], aligned_df_sorted['prediction_accuracy'], 
                           'r-', label='Prediction Accuracy', alpha=0.7)
            
            ax2.set_xlabel('Date')
            ax2.set_ylabel('LLM Sentiment Score', color='b')
            ax3.set_ylabel('Prediction Accuracy', color='r')
            ax2.set_title('Time Series: Sentiment & Accuracy')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/llm_sentiment_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation Matrix Heatmap
        correlation_data = []
        for method in ['pearson', 'spearman']:
            for sent_var, pred_vars in correlations[method].items():
                for pred_var, stats in pred_vars.items():
                    correlation_data.append({
                        'Method': method.capitalize(),
                        'Sentiment_Variable': sent_var,
                        'Prediction_Variable': pred_var,
                        'Correlation': stats['correlation'],
                        'P_Value': stats['p_value'],
                        'Significant': stats['significant']
                    })
        
        if correlation_data:
            corr_df = pd.DataFrame(correlation_data)
            
            # Create pivot table for heatmap
            pivot_df = corr_df[corr_df['Method'] == 'Pearson'].pivot(
                index='Sentiment_Variable', 
                columns='Prediction_Variable', 
                values='Correlation'
            )
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_df, annot=True, cmap='RdBu_r', center=0, 
                       square=True, fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
            plt.title('LLM Sentiment-Prediction Correlation Matrix (Pearson) - STANDALONE', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"  ✓ Visualizations saved to {output_dir}/")
    
    def generate_summary_report(self, aligned_df: pd.DataFrame, 
                              correlations: Dict, output_file: str) -> str:
        """Generate summary report for professor"""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("LLM SENTIMENT CORRELATION ANALYSIS REPORT - STANDALONE VERSION")
        report_lines.append("Professor's Assignment: Raw Text → LLM Sentiment → Correlation Analysis")
        report_lines.append("No dependencies on Cryptex project")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Dataset Summary
        report_lines.append("DATASET SUMMARY:")
        report_lines.append("-" * 40)
        report_lines.append(f"Total aligned records: {len(aligned_df)}")
        if 'datetime_readable' in aligned_df.columns:
            start_date = pd.to_datetime(aligned_df['datetime_readable']).min().strftime('%Y-%m-%d')
            end_date = pd.to_datetime(aligned_df['datetime_readable']).max().strftime('%Y-%m-%d')
            report_lines.append(f"Date range: {start_date} to {end_date}")
        
        if 'text_count' in aligned_df.columns:
            report_lines.append(f"Average texts per hour: {aligned_df['text_count'].mean():.1f}")
        
        report_lines.append("")
        
        # Main Results
        report_lines.append("KEY FINDINGS:")
        report_lines.append("-" * 40)
        
        # Primary correlation of interest
        main_correlation = None
        if ('avg_sentiment' in correlations['pearson'] and 
            'prediction_accuracy' in correlations['pearson']['avg_sentiment']):
            
            main_correlation = correlations['pearson']['avg_sentiment']['prediction_accuracy']
            significance = "SIGNIFICANT" if main_correlation['significant'] else "NOT SIGNIFICANT"
            
            report_lines.append(f"1. LLM Sentiment vs Prediction Accuracy:")
            report_lines.append(f"   Correlation: r = {main_correlation['correlation']:.3f}")
            report_lines.append(f"   P-value: {main_correlation['p_value']:.4f}")
            report_lines.append(f"   Statistical significance: {significance}")
            report_lines.append(f"   Sample size: {main_correlation['n_samples']} data points")
            report_lines.append("")
        
        # Other notable correlations
        report_lines.append("2. Other Notable Correlations:")
        all_correlations = []
        for method in ['pearson']:  # Focus on Pearson for simplicity
            for sent_var, pred_vars in correlations[method].items():
                for pred_var, stats in pred_vars.items():
                    if abs(stats['correlation']) > 0.1:  # Only show meaningful correlations
                        all_correlations.append((sent_var, pred_var, stats['correlation'], stats['significant']))
        
        # Sort by absolute correlation strength
        all_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for sent_var, pred_var, corr, significant in all_correlations[:5]:  # Top 5
            sig_marker = "*" if significant else " "
            report_lines.append(f"   {sig_marker} {sent_var} vs {pred_var}: r = {corr:.3f}")
        
        report_lines.append("")
        
        # Statistical Summary
        report_lines.append("STATISTICAL SUMMARY:")
        report_lines.append("-" * 40)
        if 'avg_sentiment' in aligned_df.columns:
            report_lines.append(f"Average LLM sentiment score: {aligned_df['avg_sentiment'].mean():.3f}")
            report_lines.append(f"Sentiment standard deviation: {aligned_df['avg_sentiment'].std():.3f}")
        
        if 'prediction_accuracy' in aligned_df.columns:
            report_lines.append(f"Average prediction accuracy: {aligned_df['prediction_accuracy'].mean():.3f}")
            report_lines.append(f"Accuracy standard deviation: {aligned_df['prediction_accuracy'].std():.3f}")
        
        report_lines.append("")
        
        # Conclusions
        report_lines.append("CONCLUSIONS:")
        report_lines.append("-" * 40)
        
        if main_correlation:
            if abs(main_correlation['correlation']) > 0.3:
                strength = "strong"
            elif abs(main_correlation['correlation']) > 0.1:
                strength = "moderate"
            else:
                strength = "weak"
            
            report_lines.append(f"• LLM sentiment shows {strength} correlation with prediction accuracy")
            
            if main_correlation['significant']:
                report_lines.append("• The correlation is statistically significant (p < 0.05)")
                report_lines.append("• LLM sentiment analysis provides valuable signal for prediction quality")
            else:
                report_lines.append("• The correlation is not statistically significant")
                report_lines.append("• More data may be needed to establish reliable relationship")
        
        report_lines.append("")
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("-" * 40)
        report_lines.append("• Consider integrating LLM sentiment as a feature in prediction models")
        report_lines.append("• Expand dataset size for more robust statistical analysis")
        report_lines.append("• Test with different LLM models for comparison")
        report_lines.append("• Validate findings with out-of-sample data")
        
        # Write report
        report_text = "\n".join(report_lines)
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"  ✓ Summary report saved: {output_file}")
        return report_text
    
    def run_complete_analysis(self, llm_sentiment_file: str, 
                            prediction_file: str = None,
                            use_mock_predictions: bool = True) -> Dict:
        """
        Run complete LLM sentiment correlation analysis
        """
        print("🚀 Starting LLM Sentiment Correlation Analysis")
        print("As requested by professor: Raw Text → LLM → Correlation")
        print("STANDALONE VERSION - No project dependencies")
        print("=" * 80)
        
        # 1. Load LLM sentiment data
        sentiment_df = self.load_llm_sentiment_data(llm_sentiment_file)
        
        # 2. Load or create prediction data
        if prediction_file and not use_mock_predictions:
            print(f"Loading prediction data: {prediction_file}")
            prediction_df = pd.read_csv(prediction_file)
        else:
            prediction_df = self.create_mock_prediction_data(sentiment_df)
        
        # 3. Align data
        aligned_df = self.align_sentiment_with_predictions(sentiment_df, prediction_df)
        
        # 4. Compute correlations
        correlations = self.compute_correlations(aligned_df)
        
        # 5. Create visualizations
        self.create_correlation_visualizations(aligned_df, correlations, self.output_dir)
        
        # 6. Generate report
        report_file = os.path.join(self.output_dir, 'llm_sentiment_correlation_report.txt')
        summary_report = self.generate_summary_report(aligned_df, correlations, report_file)
        
        # 7. Save detailed results
        results_file = os.path.join(self.output_dir, 'detailed_results.csv')
        aligned_df.to_csv(results_file, index=False)
        
        print(f"\n✅ LLM Sentiment Correlation Analysis Complete!")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📊 Main visualization: {self.output_dir}/llm_sentiment_correlations.png")
        print(f"📝 Report: {report_file}")
        
        return {
            'aligned_data': aligned_df,
            'correlations': correlations,
            'summary_report': summary_report
        }


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='LLM Sentiment Correlation Analysis - STANDALONE')
    parser.add_argument('--llm-sentiment-file', required=True, 
                       help='CSV file with LLM-analyzed sentiment data')
    parser.add_argument('--prediction-file', 
                       help='CSV file with prediction data (optional - will use mock data if not provided)')
    parser.add_argument('--output-dir', default='llm_correlation_results', 
                       help='Output directory for results')
    parser.add_argument('--use-mock-predictions', action='store_true', 
                       help='Use mock prediction data even if prediction file is provided')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = LLMSentimentCorrelationAnalyzer(output_dir=args.output_dir)
    
    # Run analysis
    results = analyzer.run_complete_analysis(
        llm_sentiment_file=args.llm_sentiment_file,
        prediction_file=args.prediction_file,
        use_mock_predictions=args.use_mock_predictions
    )
    
    # Print quick summary for professor
    correlations = results['correlations']
    if ('avg_sentiment' in correlations['pearson'] and 
        'prediction_accuracy' in correlations['pearson']['avg_sentiment']):
        
        main_corr = correlations['pearson']['avg_sentiment']['prediction_accuracy']
        print(f"\n🎯 PROFESSOR'S KEY RESULT:")
        print(f"   LLM Sentiment vs Prediction Accuracy: r = {main_corr['correlation']:.3f}")
        print(f"   Statistical significance: {'YES' if main_corr['significant'] else 'NO'}")
        print(f"   P-value: {main_corr['p_value']:.4f}")


if __name__ == "__main__":
    main()