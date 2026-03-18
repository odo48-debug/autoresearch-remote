# AutoResearch — Generic Text Modeling 🤖

This project is a remote implementation designed to test and experiment with Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) repository without running it locally. 

It provides an autonomous agent loop for ANY sequential text dataset, using Gemini 2.5 Flash as an agent and Modal.com for GPU infrastructure to optimize a nano GPT model through iterative experimentation.

---

## Requirements

- Python 3.10+
- A [modal.com](https://modal.com) account
- A [Google AI Studio](https://aistudio.google.com) API key (free for Gemini 2.5 Flash)

---

## Setup (5 minutes)

### 1. Install Modal
```bash
pip install modal
modal setup   # Authenticate in your browser
```

### 2. Configure Gemini API Key
```bash
modal secret create gemini-api-key GEMINI_API_KEY=your_api_key_here
```
Get your free key at: https://aistudio.google.com/apikey

---

## Usage

The system will:
1. Download or load the specified dataset.
2. Launch the research loop on a cloud GPU (T4 by default).
3. Let Gemini propose code changes, train the model, and measure performance.
4. Keep improvements and discard failures automatically.

### A. Using a HuggingFace Dataset
```bash
# Example: Irish music (ABC notation)
modal run app.py --repo-id "sander-wood/irishman" --column "abc notation"

# Example: Python code instructions
modal run app.py --repo-id "iamtarun/python_code_instructions_18k_alpaca" --column "output"
```

### B. Using a Local File
1. Upload your file to the Modal volume:
```bash
modal volume put autoresearch-text-vol my_data.txt /vol/uploads/my_data.txt
```
2. Run the research loop:
```bash
# For .txt files
modal run app.py --local-file "my_data.txt"

# For .csv or .json files, specify the column/key
modal run app.py --local-file "data.csv" --column "text_content"
```

### C. Customizing Iterations
```bash
modal run app.py --iters 50
```

---

## Monitoring and Results

### View Logs
```bash
# In another terminal
modal logs autoresearch-text
```

### Download Best Model and Logs
```bash
# List files in the volume
modal volume ls autoresearch-text-vol

# Download the best train.py found
modal volume get autoresearch-text-vol best_train.py ./best_train.py

# Download the experiment log (JSONL)
modal volume get autoresearch-text-vol autoresearch.jsonl ./autoresearch.jsonl
```

---

## Estimated Costs (on Modal)

| Session Type | GPU | Duration | Approx. Cost |
|--------------|-----|----------|--------------|
| Quick Test   | T4  | 1h (~12 iters) | ~$0.30 |
| Overnight    | T4  | 8h (~100 iters) | ~$2.40 |
| Fast Run     | A10G| 8h (~200 iters) | ~$8.00 |

*Note: Modal provides $30 free monthly credit.*

---

## Advanced Options

### Change GPU Type
Edit the `@app.function` decorator for `research_loop` in `app.py`:
```python
gpu="A10G"   # Faster than T4 (~3x more iterations per hour)
```

### Volume Structure
```
/vol/
  data/
    dataset.txt         ← Processed training data
  train.py              ← Current evolving version
  best_train.py         ← Best version discovered so far
  autoresearch.jsonl    ← Full experiment history
  results/
    train_iter_N_...py  ← Each improved version saved
  uploads/              ← Your uploaded local files
```