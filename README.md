# Co-NAML-LSTUR: A Combined Model with Attentive Multi-View Learning and Long- and Short-term User Representations for News Recommendation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of **Co-NAML-LSTUR**, a neural news recommendation model that combines attentive multi-view learning with long- and short-term user representations.

> **📋 Important Note**: This repository currently implements the core components as `NAML_LSTUR_DKN` with the full `Co_NAML_LSTUR` configuration available in `config.py`. The model architecture and results presented in the paper are achieved using the `Co_NAML_LSTUR` configuration. You can use either:
> - `--model_name "NAML_LSTUR_DKN"` for the current implementation
> - `--model_name "Co_NAML_LSTUR"` for the exact paper configuration (requires model implementation in `model/Co_NAML_LSTUR/`)

## 📄 Paper Information

**Title:** Co-NAML-LSTUR: A Combined Model with Attentive Multi-View Learning and Long- and Short-term User Representations for News Recommendation

**Abstract:** Our proposed Co-NAML-LSTUR model combines the strengths of NAML (Neural Attentive Multi-View Learning) for news representation and LSTUR (Long- and Short-Term User Representation) for user modeling. The model achieves state-of-the-art performance on news recommendation benchmarks by effectively capturing both fine-grained news content features and diverse user interest patterns.

## 🏆 Main Results

Performance comparison on MIND-small and MIND-large datasets:

| Model | #Params | MIND-small |  |  |  | MIND-large |  |  |  |
|-------|---------|------------|--|--|--|------------|--|--|--|
|  |  | AUC | MRR | nDCG@5 | nDCG@10 | AUC | MRR | nDCG@5 | nDCG@10 |
| LibFM§ | - | 0.5974 | 0.2633 | 0.2795 | 0.3429 | 0.6185 | 0.2945 | 0.3145 | 0.3713 |
| DeepFM§ | - | 0.5989 | 0.2621 | 0.2774 | 0.3406 | 0.6187 | 0.2930 | 0.3135 | 0.3705 |
| NRMS‡ | 22M | 0.6183 | 0.2753 | 0.2980 | 0.3653 | 0.6776 | 0.3305 | 0.3594 | 0.4163 |
| Hi-Fi Ark | 21.9M | 0.6049 | 0.2647 | 0.2927 | 0.3560 | - | - | - | - |
| NPA§ | - | 0.6465 | 0.3001 | 0.3314 | 0.3947 | 0.6592 | 0.3207 | 0.3472 | 0.4037 |
| TANR | 21.7M | 0.6338 | 0.2868 | 0.3169 | 0.3804 | - | - | - | - |
| LSTUR‡ | 71.6M | 0.6372 | 0.2769 | 0.2559 | 0.3099 | 0.6773 | 0.3277 | 0.3559 | 0.4134 |
| NAML‡ | 22.2M | 0.6195 | 0.2528 | 0.2675 | 0.3356 | 0.6686 | 0.3249 | 0.3524 | 0.4091 |
| DKN‡ | 22.9M | 0.5954 | 0.2659 | 0.2909 | 0.3518 | 0.6460 | 0.3132 | 0.2909 | 0.3518 |
| HieRec† | - | 0.6795 | 0.3287 | 0.3636 | 0.4253 | 0.6903 | 0.3389 | 0.3384 | 0.3948 |
| MINER§ | - | 0.6961 | 0.3397 | 0.3762 | 0.4390 | 0.7151 | 0.3618 | 0.3972 | 0.4534 |
| **Co-NAML-LSTUR** | **46.4M** | **0.6571** | **0.3119** | **0.3465** | **0.4028** | **0.6931** | **0.3420** | **0.3698** | **0.4275** |

†: Results from [HieRec paper](https://aclanthology.org/2021.acl-long.423.pdf)  
‡: Results from [MIND paper](https://aclanthology.org/2020.acl-main.331.pdf)  
§: Results from [MINER paper](https://aclanthology.org/2022.findings-acl.29.pdf)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (for GPU training) or MPS (for Apple Silicon)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/MinhNguyenDS/Co-NAML-LSTUR.git
cd Co-NAML-LSTUR
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Verify setup (recommended):**
```bash
python check_setup.py
```

This will verify that all dependencies are properly installed and your environment is configured correctly.

### Dataset Preparation

1. **Download MIND dataset:**
   - [MIND-small](https://msnews.github.io/) for quick experimentation
   - [MIND-large](https://msnews.github.io/) for full evaluation

2. **Prepare the data:**
```bash
# Extract MIND dataset to ./mind_data/
# Run data preprocessing
python data_preprocess.py

# For GloVe embeddings (optional):
python data_preprocess_GLOVE.py
```

3. **Expected data structure:**
```
data/
├── train/
│   ├── behaviors.tsv
│   ├── news.tsv
│   ├── behaviors_parsed.tsv
│   ├── news_parsed.tsv
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

## 💻 Usage

### Training

1. **Basic training:**
```bash
python train.py --model_name "Co_NAML_LSTUR"
```

2. **Training with custom parameters:**
```bash
python train.py \
    --model_name "Co_NAML_LSTUR" \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --num_epochs 5
```

3. **Training on Mac with MPS:**
```bash
# MPS is automatically detected and used
python train.py --model_name "Co_NAML_LSTUR"
```

### Evaluation

```bash
python evaluate.py --model_name "Co_NAML_LSTUR"
```

### Inference

1. **Single prediction:**
```bash
python inference.py \
    --model_name "Co_NAML_LSTUR" \
    --user_id 12345 \
    --clicked_news "N37378" "N14827" "N50398" \
    --candidate_news "N37378" "N14827" "N50398" "N48265" "N42793" \
    --top_k 5
```

2. **Batch prediction:**
```bash
python inference.py \
    --model_name "Co_NAML_LSTUR" \
    --batch_file sample_batch_inference.json
```

3. **Programmatic usage:**
```python
from inference import NewsRecommendationInference

# Initialize model
inferencer = NewsRecommendationInference(model_name="Co_NAML_LSTUR")
inferencer.load_news_data()
inferencer.precompute_news_vectors()

# Get recommendations
recommendations = inferencer.recommend_top_k(
    user_id=12345,
    clicked_news_ids=["N37378", "N14827"],
    candidate_news_ids=["N48265", "N42793", "N20404"],
    k=10
)
```

## 🏗️ Model Architecture

### Co-NAML-LSTUR Components

1. **News Encoder (NAML)**:
   - Multi-view learning for news representation
   - Attention mechanisms for title, abstract, category, and subcategory
   - DistilBERT integration for semantic understanding

2. **User Encoder (LSTUR)**:
   - Long-term user representation via user embeddings
   - Short-term user representation via clicked news history
   - LSTM-based sequential modeling

3. **Click Predictor**:
   - Deep neural network for click probability prediction
   - Combines news and user representations

### Model Configuration

Key parameters in `config.py`:

```python
class Co_NAML_LSTURConfig(BaseConfig):
    # Model dimensions
    num_filters = 300
    query_vector_dim = 200
    
    # Input features
    dataset_attributes = {
        "news": ["category", "subcategory", "title", "abstract"],
        "record": ["user", "clicked_news_length"]
    }
    
    # Training parameters
    batch_size = 64  # Automatically adjusted for MPS
    learning_rate = 0.0001
    num_epochs = 5
```

## 📁 Project Structure

```
Co-NAML-LSTUR/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.py                         # Model configurations
├── check_setup.py                    # Environment verification script
├── train.py                          # Training script
├── evaluate.py                       # Evaluation script
├── inference.py                      # Inference system
├── example_inference.py              # Usage examples
├── INFERENCE_README.md               # Detailed inference guide
├── data_preprocess.py                # Data preprocessing
├── dataset.py                        # Data loading utilities
├── model/
│   └── Co_NAML_LSTUR/               # Model implementation
│       ├── __init__.py
│       ├── news_encoder.py          # NAML news encoder
│       ├── user_encoder.py          # LSTUR user encoder
│       └── click_predictor/         # Click prediction module
├── checkpoint/                       # Saved model checkpoints
├── data/                            # Dataset directory
├── runs/                            # TensorBoard logs
└── sample_batch_inference.json      # Example batch data
```


## 📝 Citation

If you use this code in your research, please cite our paper:
```bibtex
  Coming soon
```

<!-- ```bibtex
@inproceedings{co-naml-lstur-2024,
  title={Co-NAML-LSTUR: A Combined Model with Attentive Multi-View Learning and Long- and Short-term User Representations for News Recommendation},
  author={[Your Names]},
  booktitle={[Conference/Journal Name]},
  year={2024},
  pages={[Page Numbers]},
  organization={[Publisher]}
}
``` -->

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ **Star this repository if you find it helpful!** ⭐
