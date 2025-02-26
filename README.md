# Paper
This repository contains the code and data for our paper:

**Uncertainty Quantification for LLM-Based Survey Simulations**\
_Chengpiao Huang, Yuhang Wu, Kaizheng Wang_\
Paper: (https://arxiv.org/abs/2502.17773)

## Datasets
To start, download `Data.zip` and extract the files into the same directory as the code. We used two datasets publicly available on GitHub: [Eedi](https://github.com/joyheyueya/psychometric-alignment) and [OpinionQA](https://github.com/tatsu-lab/opinions_qa). In the `synthetic answers` folder, we keep a folder of `raw` answers from the LLMs and a folder of cleaned answers ready for evaluation. For example, for the Eedi dataset, a `raw` answer may contain an LLM's reasoning and a final answer letter (e.g., 'A'), while a cleaned answer is binary, indicating whether the answer is correct.

## Running the Code
1. `evaluations.py` contains all function implementations.
2. See `EEDI llm calls 4o.ipynb` and `OpinionQA llm calls 4o.ipynb` for two examples of calling an LLM API for simulated answers. The codes can be easily modified to accomodate other APIs. More specifically, we used [OpenAI](https://openai.com/) API for GPT, [Anthropic](https://www.anthropic.com/) API for Claude, and [Together AI](https://www.together.ai/) API for Llama, Mistral, and DeepSeek.
3. See `EEDI evaluations 4o.ipynb` and `OpinionQA evaluations 4o.ipynb` for example uses of functions to evaluate synthetic answers from **one** model.
4. See `EEDI evaluations all.ipynb` and `OpinionQA evaluations all.ipynb` for comparisons of synthetic answers generated by multiple models. These are the codes used to generate the graphs and tables in the paper.
