# News Recommendation Inference with Co_NAML_LSTUR

This guide explains how to use the inference system for the Co_NAML_LSTUR model on Mac with MPS support.

## Features

- ✅ **MPS Support**: Optimized for Apple Silicon Macs
- ✅ **Efficient Inference**: Precomputed news vectors for fast predictions
- ✅ **Flexible Interface**: Single predictions or batch processing
- ✅ **Memory Optimized**: Configurable batch sizes for different hardware
- ✅ **Easy to Use**: Simple command-line interface and Python API

## Quick Start

### 1. Single Prediction

```bash
# Activate your virtual environment
source venv/bin/activate

# Run inference with default parameters
python inference.py --model_name "Co_NAML_LSTUR"

# Custom prediction
python inference.py \
    --model_name "Co_NAML_LSTUR" \
    --user_id 12345 \
    --clicked_news "N37378" "N14827" "N50398" \
    --candidate_news "N37378" "N14827" "N50398" "N48265" "N42793" \
    --top_k 5
```

### 2. Batch Prediction

```bash
# Run batch inference using JSON file
python inference.py \
    --model_name "Co_NAML_LSTUR" \
    --batch_file sample_batch_inference.json
```

### 3. Programmatic Usage

```python
from inference import NewsRecommendationInference

# Initialize inference system
inferencer = NewsRecommendationInference(model_name="Co_NAML_LSTUR")

# Load data and precompute vectors
inferencer.load_news_data("./data/test/news_parsed.tsv")
inferencer.precompute_news_vectors()

# Make recommendations
recommendations = inferencer.recommend_top_k(
    user_id=12345,
    clicked_news_ids=["N37378", "N14827", "N50398"],
    candidate_news_ids=["N37378", "N14827", "N50398", "N48265", "N42793"],
    k=5
)

for news_id, prob in recommendations:
    print(f"News {news_id}: {prob:.4f}")
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model_name` | Model name to use | `"Co_NAML_LSTUR"` |
| `--checkpoint_path` | Path to model checkpoint | Auto-detect latest |
| `--news_data` | Path to news data file | `"./data/test/news_parsed.tsv"` |
| `--user_id` | User ID for prediction | `1` |
| `--clicked_news` | List of clicked news IDs | `["N37378", "N14827", "N50398"]` |
| `--candidate_news` | List of candidate news IDs | `["N37378", "N14827", "N50398", "N48265", "N42793"]` |
| `--top_k` | Number of recommendations | `5` |
| `--batch_file` | JSON file for batch prediction | None |

## Batch Inference Format

Create a JSON file with the following structure:

```json
{
  "users": [
    {
      "user_id": 12345,
      "clicked_news_ids": ["N37378", "N14827", "N50398"]
    },
    {
      "user_id": 12346,
      "clicked_news_ids": ["N48265", "N42793", "N20404"]
    }
  ],
  "candidate_news": [
    "N37378", "N14827", "N50398", "N48265", "N42793"
  ]
}
```

## Example Outputs

### Single Prediction
```
Using device: mps
Loading checkpoint: ./checkpoint/Co_NAML_LSTUR/ckpt-1010.pth
Model loaded successfully!
Loading news data...
Loaded 101527 news articles
Precomputing news vectors...
Computing news vectors: 100%|████████████| 101527/101527 [02:15<00:00, 749.42it/s]
Precomputed vectors for 101527 news articles

Making prediction for user 12345
Clicked news: ['N37378', 'N14827', 'N50398']
Candidate news: ['N37378', 'N14827', 'N50398', 'N48265', 'N42793']

Top 5 recommendations:
1. News N48265: 0.8234
2. News N42793: 0.7891
3. News N37378: 0.7234
4. News N14827: 0.6891
5. News N50398: 0.6234
```

### Batch Prediction
Results are saved to `sample_batch_inference_results.json`:

```json
{
  "12345": [
    ["N48265", 0.8234],
    ["N42793", 0.7891],
    ["N37378", 0.7234]
  ],
  "12346": [
    ["N37378", 0.8934],
    ["N14827", 0.8234],
    ["N50398", 0.7891]
  ]
}
```

## Performance Tips

### For Mac with MPS:
- The system automatically uses optimized settings for MPS
- Batch size is set to 1 for memory efficiency
- Pin memory is disabled for MPS compatibility
- News vectors are stored on CPU to save GPU memory

### Memory Optimization:
- News vectors are precomputed once and reused
- Vectors are moved to CPU after computation to save GPU memory
- Batch processing reduces memory peaks

## Error Handling

The system handles common errors gracefully:

- **Missing news IDs**: Replaced with padding vectors
- **Invalid user IDs**: Processed with default embeddings  
- **Model loading errors**: Clear error messages with suggestions
- **Memory issues**: Automatic fallback to smaller batch sizes

## Integration Examples

### Real-time API
```python
from flask import Flask, request, jsonify
from inference import NewsRecommendationInference

app = Flask(__name__)
inferencer = NewsRecommendationInference()
inferencer.load_news_data()
inferencer.precompute_news_vectors()

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    recommendations = inferencer.recommend_top_k(
        user_id=data['user_id'],
        clicked_news_ids=data['clicked_news'],
        candidate_news_ids=data['candidates'],
        k=data.get('k', 10)
    )
    return jsonify(recommendations)
```

### Recommendation Pipeline
```python
# Daily batch processing
users = load_daily_users()
candidates = load_trending_news()

results = inferencer.batch_predict(users, candidates)
save_recommendations(results)
```

## Troubleshooting

### Common Issues:

1. **MPS not detected**: Update PyTorch to latest version
2. **Memory errors**: Reduce batch size or use CPU
3. **Model loading errors**: Check checkpoint path
4. **Slow inference**: Ensure news vectors are precomputed

### Performance Monitoring:
- Monitor GPU memory usage with `top -l 1 | grep "PhysMem"`
- Check inference speed with different batch sizes
- Profile memory usage for large datasets

## Files Overview

- `inference.py`: Main inference script
- `example_inference.py`: Usage examples
- `sample_batch_inference.json`: Sample batch data
- `INFERENCE_README.md`: This documentation

## Next Steps

1. Test with your specific news dataset
2. Tune batch sizes for your hardware
3. Integrate with your recommendation pipeline
4. Monitor performance in production 