# LLM Sentiment Correlation Analysis - STANDALONE

This is a **completely standalone** sentiment analysis and correlation system. It has **NO dependencies** on the main Cryptex project.

## Purpose

Analyze the correlation between LLM-generated sentiment scores and model prediction accuracy:
1. **Raw Text** → 2. **LLM Analysis** → 3. **Correlation Study**

## Files

### Core Scripts
- `export_raw_sentiment_simple.py` - Export raw sentiment data (Jan 1-31, 2025)
- `llm_sentiment_analyzer_actual.py` - Use actual LLMs for sentiment analysis
- `llm_sentiment_correlation.py` - Correlation analysis and visualization

### Dependencies
**Only standard libraries needed:**
- `pandas`, `numpy` - Data processing
- `transformers`, `torch` - LLM integration
- `matplotlib`, `seaborn` - Visualization
- `scipy` - Statistical analysis
- `requests` - API calls

**NO dependencies on:**
- Cryptex project files
- TimeLLM models
- Project-specific utilities

## Quick Start

### Step 1: Export Raw Sentiment Data
```bash
cd sentiment_correlation/

# Create sample dataset for testing
python export_raw_sentiment_simple.py --create-sample 50

# Or export real data (requires NewsAPI key)
python export_raw_sentiment_simple.py --newsapi-key YOUR_KEY
```

### Step 2: Analyze with LLM
```bash
# Use DEEPSEEK
python llm_sentiment_analyzer_actual.py \
    --input-file raw_sentiment_data/raw_sentiment_sample_50records.csv \
    --llm-model DEEPSEEK \
    --create-correlation-dataset

# Try different LLMs
python llm_sentiment_analyzer_actual.py \
    --input-file raw_sentiment_data/raw_sentiment_sample_50records.csv \
    --llm-model LLAMA3.1 \
    --create-correlation-dataset
```

### Step 3: Run Correlation Analysis
```bash
python llm_sentiment_correlation.py \
    --sentiment-file raw_sentiment_data/raw_sentiment_sample_50records_deepseek_analyzed.csv \
    --prediction-file your_prediction_data.csv
```

## Available LLMs

- **DEEPSEEK** - `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- **LLAMA3.1** - `meta-llama/Llama-3.1-8B-Instruct`
- **QWEN** - `Qwen/Qwen2.5-7B-Instruct`
- **MISTRAL** - `mistralai/Mistral-7B-Instruct-v0.3`
- **GEMMA** - `google/gemma-2-2b-it`
- **LLAMA** - `huggyllama/llama-7b`

## Output

The system generates:
1. **Raw sentiment CSV** - Timestamps + raw text
2. **LLM-analyzed CSV** - Sentiment scores from actual LLMs
3. **Correlation visualizations** - Scatter plots, heatmaps
4. **Statistical report** - Correlation coefficients, significance tests
5. **Professor's key result** - Main correlation value

## Key Features

### Raw Sentiment Export
- **Unix timestamps** (matching project format)
- **Raw text** (no pre-processing)
- **2+ articles per source per day** minimum
- **Jan 1-31, 2025** date range
- Sources: Reddit + News (configurable)

### LLM Analysis
- Uses **your actual project LLMs**, not specialized sentiment models
- Prompts LLMs: "Analyze sentiment, respond with 0-1 score"
- Extracts numerical scores from LLM responses
- Batch processing for efficiency

### Correlation Analysis
- **Pearson & Spearman** correlations
- **Statistical significance** testing
- **Time-series alignment** between sentiment and predictions

## Configuration

### Optional NewsAPI Key
Create `config.json`:
```json
{
  "newsapi_key": "your_newsapi_key_here"
}
```

### Command Line Options
```bash
# Export options
--create-sample N          # Create N sample records
--newsapi-key KEY          # NewsAPI key for news articles
--output-dir DIR           # Output directory

# LLM options
--llm-model MODEL          # DEEPSEEK, LLAMA3.1, QWEN, etc.
--batch-size N             # Batch size for processing
--device DEVICE            # auto, cpu, cuda, mps

# Correlation options
--prediction-file FILE     # Prediction CSV file (required)
```

## Expected Results

**Key Metric:**
```
LLM Sentiment vs Prediction Accuracy: r = 0.XXX
Statistical significance: YES/NO
P-value: 0.XXX
```

The system will show whether LLM-generated sentiment scores correlate with model prediction accuracy for academic analysis.

## Academic Usage

This standalone system is designed for:
- **Research papers** - Cite correlation results
- **Academic presentations** - Use generated visualizations
- **Statistical analysis** - Export detailed correlation data
- **Model comparison** - Test different LLMs (DEEPSEEK vs LLAMA3.1, etc.)

Perfect for academic research with zero dependencies on the main project!