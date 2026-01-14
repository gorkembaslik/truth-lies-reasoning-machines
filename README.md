# Truth, Lies, and Reasoning Machines

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License:  MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **NLP Final Project**: Investigating how Large Language Models reason when exposed to false, incomplete, or contradictory information. 

## 📋 Overview

This project explores the fragility and resilience of LLM reasoning under "truth distortion" scenarios.  We investigate whether models can: 
- Detect inconsistencies in provided information
- Resist misinformation and maintain factual accuracy
- Preserve logical coherence during multi-step reasoning

### Research Questions

1. How does logical coherence deteriorate under misinformation stress?
2. Can explicit self-verification prompts help preserve reasoning integrity?
3. What types of reasoning failures are most common (belief persistence, hallucination, circular logic)?

## 🏗️ Project Structure

```
truth-lies-reasoning-machines/
├── src/
│   ├── data/                    # Dataset loaders and perturbations
│   │   ├── base_dataset.py      # Abstract base class for datasets
│   │   ├── truthfulqa_loader.py # TruthfulQA dataset handler
│   │   ├── hotpotqa_loader.py   # HotpotQA dataset handler
│   │   └── perturbations.py     # Truth distortion strategies
│   │
│   ├── models/                  # LLM interfaces
│   │   ├── base_llm.py          # Abstract LLM interface
│   │   ├── gemini_client.py     # Google Gemini client
│   │   ├── github_models_client.py  # GitHub Models client
│   │   └── prompts.py           # Prompt templates
│   │
│   ├── evaluation/              # Metrics and analysis
│   │   ├── metrics. py           # Truthfulness, F1, consistency metrics
│   │   └── failure_analysis.py  # Reasoning failure detection
│   │
│   └── visualization/           # Plotting utilities
│       └── plots.py             # Result visualization
│
├── notebooks/                   # Jupyter notebooks for experiments
├── configs/                     # Configuration files
├── data/                        # Datasets (not tracked in git)
│   ├── raw/                     # Original datasets
│   ├── processed/               # Preprocessed data
│   └── results/                 # Experiment outputs
├── paper/                       # Paper figures and drafts
└── tests/                       # Unit tests
```

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/gorkembaslik/truth-lies-reasoning-machines.git
cd truth-lies-reasoning-machines

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Set Up API Keys

Copy the example environment file and add your API keys: 

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# For Google Gemini
GOOGLE_API_KEY=your-google-api-key

# For GitHub Models (backup)
GITHUB_TOKEN=your-github-token
```

### 3. Download Datasets

```bash
# TruthfulQA
wget -O data/raw/TruthfulQA.csv https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv

# HotpotQA (dev set - distractor setting)
wget -O data/raw/hotpot_dev_distractor_v1.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
```

### 4. Run Your First Experiment

```python
from src.data import TruthfulQADataset
from src.models import GeminiClient
from src.evaluation import MetricsCalculator

# Load dataset
dataset = TruthfulQADataset("data/raw/TruthfulQA.csv")

# Initialize model
llm = GeminiClient(model_name="gemini-1.5-flash")

# Run evaluation
calculator = MetricsCalculator()
for example in dataset.sample(10, seed=42):
    response = llm.generate(f"Question: {example. question}\nAnswer:")
    calculator.add_result(
        example_id=example.id,
        prediction=response.text,
        ground_truth=example.correct_answer,
        incorrect_answers=example.incorrect_answers
    )

print(calculator.get_aggregate_metrics())
```

## 📊 Datasets

### TruthfulQA
- **Purpose**: Measures if models mimic human falsehoods
- **Size**: 817 questions across 38 categories
- **Use Case**: Testing susceptibility to misconceptions

### HotpotQA
- **Purpose**: Multi-hop question answering with explainable reasoning
- **Size**: 113k questions (we use dev set:  7,405 questions)
- **Use Case**: Testing reasoning under context perturbations

## 🔬 Perturbation Strategies

| Strategy | Description | Example |
|----------|-------------|---------|
| **Counterfactual** | Hypothetical premises contradicting reality | "If Paris were in Germany..." |
| **Falsehood Injection** | Insert false facts into context | Adding "vaccines cause autism" to medical context |
| **Contradictory Context** | Add conflicting information | Two paragraphs with opposite claims |
| **Answer Leak** | Suggest wrong answer upfront | "The answer is X" (where X is wrong) |

## 📈 Evaluation Metrics

- **Exact Match (EM)**: Binary correctness
- **F1 Score**: Token-level overlap
- **Truthfulness Score**: Correct similarity - Incorrect similarity
- **Robustness Score**: Performance retention under perturbation
- **Consistency Score**: Agreement across multiple runs

## 🐛 Failure Types Analyzed

1. **Belief Persistence**:  Sticking with suggested wrong answers
2. **Hallucination**:  Generating unsupported claims
3. **Circular Logic**: Reasoning that references itself
4. **Contradiction Blindness**: Failing to notice conflicting information
5. **Anchoring Bias**: Over-relying on first-seen information

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | Explore TruthfulQA and HotpotQA datasets |
| `02_api_setup_test.ipynb` | Test Gemini and GitHub Models APIs |
| `03_baseline_experiments.ipynb` | Run baseline experiments |
| `04_perturbation_experiments.ipynb` | Truth distortion experiments |
| `05_self_verification.ipynb` | Self-correction prompt experiments |
| `06_analysis_visualization.ipynb` | Final analysis and visualizations |

## 🤝 Contributing

This is an academic project. Contributions are welcome for:
- Bug fixes
- Documentation improvements
- Additional perturbation strategies
- New evaluation metrics

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 📚 References

- Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods.  ACL 2022.
- Yang, Z., et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.  EMNLP 2018.

## 👤 Author

**Gorkem Baslik** - NLP Final Project 2026