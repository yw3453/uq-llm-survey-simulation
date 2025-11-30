# UQ for LLM Survey Simulation

This repository contains the code and data for our paper:

**How Many Human Survey Respondents is a Large Language Model Worth? An Uncertainty Quantification Perspective**\
Chengpiao Huang, Yuhang Wu, Kaizheng Wang\
Paper link: [arXiv:2502.17773, 2025](https://arxiv.org/abs/2502.17773)\
Short version "Uncertainty Quantification for LLM-Based Survey Simulations" appeared at ICML 2025.


## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd uq-llm-survey-simulation
```

### 2. Extract Data Folder

The data folder is provided as a zip file. After cloning the repository, extract the data folder and place it in the project root directory. The data folder should be located at the root of the project (same level as `src/` and `notebooks/` directories).

### 3. Environment Setup

This project supports two methods for dependency management:

#### Option A: Using `uv` (Recommended)

Install `uv` following the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/). Then, install the dependencies:

```bash
# Install dependencies using uv
uv sync
```

This will create a virtual environment and install all required packages specified in `pyproject.toml`. Activate the virtual environment:

```bash
# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

#### Option B: Using `pip` (Classic Method)

Alternatively, you can use `pip` with the provided `requirements.txt` file:

```bash
# Create a virtual environment (if not already created)
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

#### Set Up Jupyter Kernel (Optional)

If you plan to use the Jupyter notebooks, register the virtual environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name=uq-llm-survey-simulation
```

## Project Structure

```
uq-llm-survey-simulation/
├── .venv/                  # Virtual environment
├── data/                   # Dataset files (extracted from data.zip)
│   ├── EEDI/               # EEDI educational assessment dataset
│   └── OpinionQA/          # OpinionQA opinion polling dataset
├── src/                    # Python source code
│   ├── simulations.py      # Synthetic response generation
│   └── evaluations.py      # Evaluation and analysis
├── notebooks/              # Jupyter notebooks
│   ├── simulations.ipynb   # Example usage of simulations.py
│   └── evaluations.ipynb   # Example usage of evaluations.py
├── .gitignore              # Git ignore file
├── .python-version         # Python version
├── LICENSE                 # License
├── pyproject.toml          # Project dependencies (for uv)
├── requirements.txt        # Project dependencies (for pip)
├── README.md               # This file
└── uv.lock                 # uv lock file
```

## Source Code Files

### `src/simulations.py`

**Purpose**: Generate synthetic survey responses using LLM APIs.

**Main Function**: `generate_responses()`
This module provides functionality to:
- Generate synthetic answers to survey questions using various LLM APIs (OpenAI, Anthropic, TogetherAI)
- Process questions concurrently for efficiency
- Handle retries, error recovery, and concurrent API calls
- Post-process raw responses to extract structured answers
- Support two datasets: **EEDI** (educational assessment) and **OpinionQA** (opinion polling)

**Key Features**:
- Asynchronous API calls for concurrent processing
- Automatic retry with exponential backoff
- Incremental saving (allows resuming interrupted runs)
- Dataset-specific post-processing (extracts answers from LLM responses)

**Usage Example**:
```python
from src.simulations import generate_responses

await generate_responses(
    api_platform='togetherai',       # 'openai', 'anthropic', or 'togetherai'
    api_key=api_key,                 # Your API key
    llm='deepseek-v3',               # Model identifier
    dataset_name='OpinionQA',        # 'EEDI' or 'OpinionQA'
    first_synthetic_profile_id=0,    # Starting profile index
    num_of_synthetic_answers=50,     # Number of responses per question
    folder_name='synthetic_answers', # Output folder name
    max_concurrent_requests=10,      # Number of concurrent API calls
    max_retries=3                    # Retry attempts
)
```

**Supported Models**:
- OpenAI: `gpt-3.5-turbo`, `gpt-4o-mini`, `gpt-4o`, `gpt-5-mini`
- Anthropic: `claude-3.5-haiku`
- TogetherAI: `deepseek-v3`, `llama-3.3-70B-instruct-turbo`, `mistral-7B-instruct-v0.3`

**Output**:
- Raw responses: `data/{dataset_name}/{folder_name}/raw/{llm}.json`
- Clean responses: `data/{dataset_name}/{folder_name}/clean/{llm}.json`
- Random baseline: `data/{dataset_name}/{folder_name}/clean/random.json`

The random baseline contains randomly generated responses for comparison:
- **EEDI**: Random binary correctness scores from {0, 1}
- **OpinionQA**: Random opinion scores from {-1, -1/3, 0, 1/3, 1}
Each question has 2 * num_of_synthetic_answers random responses.

### `src/evaluations.py`

**Purpose**: Evaluate the quality of synthetic survey responses using statistical methods.

**Main Functions**:
- `evaluations()`: Run comprehensive evaluation pipeline for multiple models
- `plot_from_saved_evaluations()`: Generate plots and tables from saved evaluation results
- `sharpness_analysis()`: Perform sharpness analysis from saved evaluation results

This module provides functionality to:
- Calculate confidence intervals for real survey data using the Central Limit Theorem (CLT)
- Compute synthetic confidence intervals from LLM-generated responses
- Evaluate miscoverage rates to assess how well synthetic CIs capture real CIs
- Find optimal sample sizes (k_hat) for synthetic responses
- Perform train-test splits and cross-validation to assess model performance
- Generate reports and visualizations comparing different LLM models

**Key Features**:
- Multiple confidence interval types: CLT, Hoeffding's inequality, Bernstein's inequality
- Two coverage test types: 'general' (confidence set inclusion) and 'simple' (empirical mean inclusion)
- Train-test split evaluation with multiple random seeds
- Comprehensive reporting with plots and tables

**Usage Example**:
```python
from src.evaluations import evaluations, plot_from_saved_evaluations, sharpness_analysis

# Run evaluation for all models
evaluations(
    dataset_name='OpinionQA',                             # 'EEDI' or 'OpinionQA'
    models=['gpt-4o', 'claude-3.5-haiku', 'deepseek-v3'], # List of models to evaluate
    synthetic_answer_folder_name='synthetic_answers',     # Name of folder containing synthetic answers
    evaluation_results_folder_name='evaluation_results',  # Name of folder to save evaluation results
    alphas=[0.05, 0.10, 0.15, 0.20],                      # List of significance levels to evaluate
    gamma=0.5,                                            # Coverage probability for real CI
    k_max=200,                                            # Maximum number of synthetic answers to evaluate (200 for OpinionQA, 60 for EEDI)
    C=3,                                                  # Scaling constant for synthetic CI
    train_proportion=0.6,                                 # Proportion of questions to use for training
    k_min=2,                                              # Minimum k value required for valid synthetic CI (must be at least 2)
    CI_type='clt',                                        # 'clt', 'hoeffding', or 'bernstein'
    num_splits=100                                        # Number of train-test splits to perform
)

# Generate plots and tables from saved results
plot_from_saved_evaluations(
    dataset_name='OpinionQA',
    evaluation_results_folder_name='evaluation_results',
    num_splits=100,
    alphas=[0.05, 0.10, 0.15, 0.20],
    gamma=0.5,
    C=3
)

# Perform sharpness analysis from saved evaluation results
sharpness_analysis(
    dataset_name='OpinionQA',
    evaluation_results_folder_name='evaluation_results',
    type='general',
    gamma=0.5,
    histogram_model='gpt-4o'
)
```

**Output**:
- Evaluation reports (JSON):
  - `data/{dataset_name}/{evaluation_results_folder_name}/general/reports_all.json`
  - `data/{dataset_name}/{evaluation_results_folder_name}/simple/reports_all.json`
  - `data/{dataset_name}/{evaluation_results_folder_name}/general/sharpness_analysis_all.json`
  - `data/{dataset_name}/{evaluation_results_folder_name}/simple/sharpness_analysis_all.json`
- Plots (PDF): `data/{dataset_name}/{evaluation_results_folder_name}/{type}/{metric}.pdf`
  - Where `{type}` is either 'general' or 'simple'
  - Where `{metric}` is 'kappa_hat', 'test_miscov_rate', or 'synth_CI_width' (synthetic CI half-width)
- Tables (CSV): `data/{dataset_name}/{evaluation_results_folder_name}/{type}/{metric}.csv`
  - Same structure as plots
- Sharpness analysis table (CSV): `data/{dataset_name}/{evaluation_results_folder_name}/{type}/sharpness_analysis_table.csv`
- Sharpness analysis histogram (PDF): `data/{dataset_name}/{evaluation_results_folder_name}/{type}/sharpness_histogram_{model}.pdf` (if `histogram_model` is provided)

## Jupyter Notebooks

### `notebooks/simulations.ipynb`

**Purpose**: Demonstrate how to use `generate_responses()` to generate synthetic survey responses.

**Contents**:
- Import instructions and setup
- Example code for generating synthetic responses
- Parameter explanations
- Notes on API keys, processing time, and error handling

**Usage**: 
1. Open the notebook in Jupyter
2. Set your API key (either as an environment variable or in the notebook)
3. Configure parameters (API platform, LLM model, dataset, etc.)
4. Run the cells to generate synthetic responses

**Note**: Processing time depends on the number of questions, synthetic answers per question, LLM model, and API platform. Expect several hours for large-scale generation.

### `notebooks/evaluations.ipynb`

**Purpose**: Demonstrate how to use the evaluation functions to assess synthetic response quality.

**Contents**:
- Import instructions and setup
- Example code for running evaluations
- Example code for plotting evaluation results
- Example code for plotting sharpness analysis

**Usage**:
1. Open the notebook in Jupyter
2. Configure parameters (dataset name, models to evaluate, synthetic answer folder name, evaluation results folder name, significance levels, coverage probability, scaling constant, train-test split proportion, confidence interval type, number of train-test splits)
3. Run the cells to run evaluations
4. View results and plots

**Note**: This notebook does not require an API key. It only needs synthetic answer files that have already been generated (see `simulations.ipynb`).

## Data Structure

See [data/data_info.md](data/data_info.md) for details.

## License

See [LICENSE](LICENSE) file for details.
