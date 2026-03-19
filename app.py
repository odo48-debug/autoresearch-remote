"""
autoresearch_text.py
Autonomous agent loop for any sequential text dataset.
Agent: Gemini 2.5 Flash | Compute: Modal.com

Usage:
  modal run autoresearch_text.py --repo-id "sander-wood/irishman" --column "abc notation"
  modal run autoresearch_text.py --repo-id "iamtarun/python_code_instructions_18k_alpaca" --column "output"
"""

import modal
import os
import subprocess
import json
import re
from datetime import datetime

# ---------------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch", "numpy",
        "google-genai", "requests",
        "datasets", "pandas", "pyarrow"
    ])
)

app = modal.App("autoresearch-text", image=image)
volume = modal.Volume.from_name("autoresearch-text-vol", create_if_missing=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAIN_PATH   = "/vol/train.py"
LOG_PATH     = "/vol/autoresearch.jsonl"
DATASET_PATH = "/vol/data/dataset.txt"
RESULTS_PATH = "/vol/results"
MODEL_PATH   = "/vol/best_model.pt"
VOCAB_PATH   = "/vol/vocab.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

def _write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def _parse_metric(output: str) -> float | None:
    """Try val_bpb first (vocab-independent), fall back to val_loss."""
    for pattern in [r"val_bpb:\s*([\d.]+)", r"val_loss:\s*([\d.]+)"]:
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))
    return None

def _detect_metric(train_py: str) -> str:
    """Detect which metric the current train.py reports."""
    return "val_bpb" if "val_bpb" in train_py else "val_loss"

def _validate_code(code: str) -> tuple[bool, str]:
    """Compile without executing to catch syntax errors."""
    try:
        compile(code, "<string>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError line {e.lineno}: {e.msg}"

def _gemini_fix(broken_code: str, error: str) -> str:
    """Ask Gemini to fix its own broken code."""
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    prompt = f"""Fix this Python syntax error:

ERROR: {error}

CODE:
{broken_code}

Return ONLY the corrected Python code. No markdown, no explanations, no comments added."""
    code = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    ).text.strip()
    code = re.sub(r"^```python\n?", "", code)
    code = re.sub(r"\n?```$", "", code)
    return code

# ---------------------------------------------------------------------------
# Dataset download — supports HuggingFace, TXT, CSV, JSON
# ---------------------------------------------------------------------------
@app.function(volumes={"/vol": volume}, timeout=600)
def download_dataset(
    repo_id: str = "",
    column: str = "text",
    local_file: str = "",   # filename already uploaded to /vol/uploads/
):
    import pandas as pd

    os.makedirs("/vol/data", exist_ok=True)

    # ── Option A: file already in the volume ──────────────────────────────
    if local_file:
        src = f"/vol/uploads/{local_file}"
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"File not found at {src}. "
                f"Upload it first with:\n"
                f"  modal volume put autoresearch-text-vol {local_file} /vol/uploads/{local_file}"
            )
        ext = local_file.rsplit(".", 1)[-1].lower()
        print(f"Loading local file: {src} (format: {ext})")

        if ext == "txt":
            content = open(src, encoding="utf-8").read()

        elif ext == "csv":
            df = pd.read_csv(src)
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found. Available: {', '.join(df.columns)}")
            content = "\n\n".join(df[column].astype(str).tolist())

        elif ext == "json":
            import json as _json
            with open(src, encoding="utf-8") as f:
                data = _json.load(f)
            # Support both list of strings and list of dicts
            if isinstance(data, list):
                if isinstance(data[0], str):
                    content = "\n\n".join(data)
                elif isinstance(data[0], dict):
                    if column not in data[0]:
                        raise ValueError(f"Key '{column}' not found. Available: {', '.join(data[0].keys())}")
                    content = "\n\n".join(str(d[column]) for d in data)
                else:
                    raise ValueError("JSON must be a list of strings or list of dicts")
            else:
                raise ValueError("JSON must be a list at the top level")

        elif ext == "jsonl":
            import json as _json
            lines = open(src, encoding="utf-8").readlines()
            rows = [_json.loads(l) for l in lines if l.strip()]
            if isinstance(rows[0], str):
                content = "\n\n".join(rows)
            else:
                if column not in rows[0]:
                    raise ValueError(f"Key '{column}' not found. Available: {', '.join(rows[0].keys())}")
                content = "\n\n".join(str(r[column]) for r in rows)

        else:
            raise ValueError(f"Unsupported format: .{ext}. Supported: txt, csv, json, jsonl")

    # ── Option B: HuggingFace dataset ─────────────────────────────────────
    elif repo_id:
        from datasets import load_dataset
        print(f"Downloading from HuggingFace: {repo_id} (column: {column})")
        ds = load_dataset(repo_id, split="train")
        df = ds.to_pandas()
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found. Available: {', '.join(df.columns)}")
        content = "\n\n".join(df[column].astype(str).tolist())

    else:
        raise ValueError("Provide either --repo-id (HuggingFace) or --local-file (local file)")

    _write(DATASET_PATH, content)
    print(f"Dataset ready: {len(content):,} chars at {DATASET_PATH}")
    volume.commit()

# ---------------------------------------------------------------------------
# Base train.py — generic nano GPT for any text
# ---------------------------------------------------------------------------
BASE_TRAIN_PY = '''"""
train.py — Nano GPT for generic text sequence modeling.
This file is automatically modified by the agent.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time, os, math, json

# ── Hyperparameters (agent-editable zone) ─────────────────────────────────
BLOCK_SIZE    = 384
BATCH_SIZE    = 16 # Changed from 32 to 16
N_EMBD        = 384
N_HEAD        = 1
N_LAYER       = 7
DROPOUT       = 0.05 # Changed from 0.1 to 0.05
LEARNING_RATE = 3e-4
TIME_BUDGET   = 300
EVAL_ITERS    = 40
WEIGHT_DECAY  = 0.01 # Changed from 0.015 to 0.01
GRAD_CLIP     = 1.0
# ──────────────────────────────────────────────────────────────────────────

DATASET_PATH = "/vol/data/dataset.txt"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data
with open(DATASET_PATH, encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: "".join(itos.get(i, "") for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        ls = [model(*get_batch(split))[1].item() for _ in range(EVAL_ITERS)]
        losses[split] = np.mean(ls)
    model.train()
    return losses

# ── Fixed evaluation metric (DO NOT MODIFY) ───────────────────────────────
@torch.no_grad()
def evaluate_val_bpb(model):
    """Bits per byte — vocab-size independent. Never modify this function."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for _ in range(EVAL_ITERS * 2):
        x, y = get_batch("val")
        _, loss = model(x, y)
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()
    model.train()
    return total_loss / (math.log(2) * max(1, total_tokens))
# ──────────────────────────────────────────────────────────────────────────

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.k = nn.Linear(N_EMBD, head_size, bias=False)
        self.q = nn.Linear(N_EMBD, head_size, bias=False)
        self.v = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.drop = nn.Dropout(DROPOUT)
    def forward(self, x):
        B, T, C = x.shape
        w = self.q(x) @ self.k(x).transpose(-2,-1) * C**-0.5
        w = w.masked_fill(self.tril[:T,:T]==0, float("-inf"))
        w = self.drop(F.softmax(w, dim=-1))
        return w @ self.v(x)

class MultiHead(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        # If n_heads is 1, this effectively becomes a single attention head with N_EMBD capacity
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj  = nn.Linear(N_EMBD, N_EMBD)
        self.drop  = nn.Dropout(DROPOUT)
    def forward(self, x):
        # The concatenation still works correctly for n_heads=1, resulting in a list of one tensor
        return self.drop(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        # SwiGLU (Swish GLU) activation block, replacing GELU.
        # The intermediate dimension is scaled by 2/3 to approximately match parameter count with the original GELU FFN.
        HIDDEN_DIM = int(4 * N_EMBD * 2 / 3) # e.g., for N_EMBD=256, HIDDEN_DIM=682
        self.w1 = nn.Linear(N_EMBD, HIDDEN_DIM) # Corresponds to W_gate
        self.w2 = nn.Linear(N_EMBD, HIDDEN_DIM) # Corresponds to W_value
        self.w3 = nn.Linear(HIDDEN_DIM, N_EMBD) # Corresponds to W_output
        self.drop = nn.Dropout(DROPOUT)
    def forward(self, x):
        # SwiGLU computation: W_output(SiLU(W_gate(x)) * W_value(x))
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa  = MultiHead(N_HEAD, N_EMBD // N_HEAD)
        self.ff  = FeedForward()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class NanoGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks  = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln      = nn.LayerNorm(N_EMBD)
        self.head    = nn.Linear(N_EMBD, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=device))
        x = self.ln(self.blocks(x))
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -BLOCK_SIZE:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

# Training
model = NanoGPT().to(device)
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,} | Device: {device} | Vocab: {vocab_size}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

t0 = time.time()
step = 0
while True:
    elapsed = time.time() - t0
    if elapsed >= TIME_BUDGET:
        break

    x, y = get_batch("train")
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    if GRAD_CLIP > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    step += 1

    if step % 200 == 0:
        losses = estimate_loss(model)
        print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed:.0f}s")

# Final fixed metric
val_bpb = evaluate_val_bpb(model)
print(f"val_bpb: {val_bpb:.4f}")
print(f"steps: {step}")

# Save weights and vocab
torch.save(model.state_dict(), "/vol/best_model.pt")
vocab = {"stoi": stoi, "itos": itos, "vocab_size": vocab_size}
with open("/vol/vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f)
'''

# ---------------------------------------------------------------------------
# Gemini agent
# ---------------------------------------------------------------------------
PROGRAM_MD = """
You are an expert ML researcher optimizing a nano GPT for generic text sequence modeling.

# Experimentation

Each experiment runs on a single GPU. The training script runs for a fixed time budget of 5 minutes (wall clock training time, excluding startup/compilation).

## What you CAN do
- Modify train.py — this is the only file you edit.
- Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

## What you CANNOT do
- Modify evaluate_val_bpb() — it is the fixed evaluation metric and ground truth.
- Change DATASET_PATH or any file paths.
- Install new packages or add dependencies.
- Modify the final print statement: it must always be exactly: val_bpb: X.XXXX

## Goal
Get the lowest val_bpb. Since the time budget is fixed at 5 minutes, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

## VRAM
VRAM is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

## Simplicity criterion
All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.
- A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it.
- A 0.001 val_bpb improvement from deleting code? Definitely keep.
- An improvement of ~0 but much simpler code? Keep.

## The first run
Your very first run should always be to establish the baseline — run the training script as is.

## Do not repeat failed experiments
Before proposing a change, check the experiment history. If a similar change was already tried and failed, explore a different direction.

## Hard constraints (NEVER modify)
- evaluate_val_bpb() function — this is the fixed metric
- DATASET_PATH = "/vol/data/dataset.txt"
- Final print must be exactly: val_bpb: X.XXXX
- Do not add external dependencies
- TIME_BUDGET = 300 (5-minute fixed budget)
"""

def gemini_propose(train_py: str, history: list[dict], baseline: float) -> str:
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Include best results and last 10 iterations for context
    best = sorted([h for h in history if h["improved"]], key=lambda x: x["val_bpb"])[:5]
    recent = history[-10:]
    combined = {h["iter"]: h for h in best + recent}
    history_str = "\n".join([
        f"- Iter {h['iter']:3d}: {h['change'][:60]} → val_bpb {h['val_bpb']:.4f} ({'✓' if h['improved'] else '✗'})"
        for h in sorted(combined.values(), key=lambda x: x["iter"])
    ])

    prompt = f"""{PROGRAM_MD}

## Baseline val_bpb (unmodified model): {baseline:.4f}
## Current best val_bpb: {min((h['val_bpb'] for h in history if h['improved']), default=baseline):.4f}

## Experiment history
{history_str if history_str else "No experiments yet — this is the first iteration."}

## Current train.py
```python
{train_py}
```
"""

    code = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    ).text.strip()
    code = re.sub(r"^```python\n?", "", code)
    code = re.sub(r"\n?```$", "", code)
    return code

# ---------------------------------------------------------------------------
# Main research loop
# ---------------------------------------------------------------------------
@app.function(
    volumes={"/vol": volume},
    gpu="T4",
    timeout=3600 * 8,
    secrets=[modal.Secret.from_name("gemini-api-key")],
)
def research_loop(n_iters: int = 50):
    import shutil

    print("=" * 60)
    print("AutoResearch — Generic Text Modeling")
    print(f"Starting loop: {n_iters} iterations")
    print("=" * 60)

    # Initialize base train.py if not present
    if not os.path.exists(TRAIN_PATH):
        _write(TRAIN_PATH, BASE_TRAIN_PY)
        print("Base train.py created")

    os.makedirs(RESULTS_PATH, exist_ok=True)

    # ── Iteration 0: baseline ─────────────────────────────────────────────
    print("\n--- Iteration 0 / baseline ---")
    baseline_score = float("inf")
    try:
        result = subprocess.run(
            ["python", TRAIN_PATH],
            capture_output=True, text=True, timeout=360
        )
        baseline_score = _parse_metric(result.stdout + result.stderr)
        if baseline_score is not None:
            print(f"Baseline val_bpb: {baseline_score:.4f}")
        else:
            print("Warning: could not parse baseline metric, continuing without reference")
            baseline_score = float("inf")
    except Exception as e:
        print(f"Warning: baseline run failed ({e}), continuing without reference")
    # ──────────────────────────────────────────────────────────────────────

    history = []
    best_score = baseline_score
    best_train_py = _read(TRAIN_PATH)

    for i in range(n_iters):
        print(f"\n{'─'*50}")
        print(f"Iteration {i+1}/{n_iters}")

        current = _read(TRAIN_PATH)

        # 1. Gemini proposes a change
        print("Gemini generating hypothesis...")
        try:
            proposed = gemini_propose(current, history, baseline_score)
        except Exception as e:
            print(f"Gemini error: {e}, skipping")
            continue

        change_desc = proposed.split("\n")[0].replace("# CHANGE:", "").strip() \
            if proposed.startswith("# CHANGE:") else "no description"
        print(f"Hypothesis: {change_desc}")

        # 2. Validate syntax — attempt auto-fix once
        valid, error = _validate_code(proposed)
        if not valid:
            print(f"Syntax error: {error} — asking Gemini to fix...")
            try:
                proposed = _gemini_fix(proposed, error)
                valid, error = _validate_code(proposed)
                if not valid:
                    print(f"Fix failed: {error}, skipping")
                    continue
            except Exception as e:
                print(f"Fix error: {e}, skipping")
                continue

        # 3. Backup and write proposal
        shutil.copy(TRAIN_PATH, TRAIN_PATH + ".bak")
        _write(TRAIN_PATH, proposed)

        # 4. Run training
        print("Training...")
        t0 = datetime.now()
        try:
            result = subprocess.run(
                ["python", TRAIN_PATH],
                capture_output=True, text=True, timeout=360
            )
            output = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            print("Timeout — discarding")
            shutil.copy(TRAIN_PATH + ".bak", TRAIN_PATH)
            continue
        except Exception as e:
            print(f"Run error: {e} — discarding")
            shutil.copy(TRAIN_PATH + ".bak", TRAIN_PATH)
            continue

        # 5. Parse metric — if failed, try to fix and re-run once
        score = _parse_metric(output)
        if score is None:
            # Extract error from output and ask Gemini to fix
            error_match = re.search(r"(Traceback[\s\S]+)", output)
            if error_match:
                error_text = error_match.group(1)[:800]
                print(f"Runtime error detected:\n{error_text}")
                print("Asking Gemini to fix runtime error...")
                try:
                    fixed = _gemini_fix(_read(TRAIN_PATH), error_text)
                    valid, syntax_err = _validate_code(fixed)
                    if valid:
                        _write(TRAIN_PATH, fixed)
                        result2 = subprocess.run(
                            ["python", TRAIN_PATH],
                            capture_output=True, text=True, timeout=360
                        )
                        score = _parse_metric(result2.stdout + result2.stderr)
                        if score is not None:
                            print(f"Fix successful — metric parsed after retry")
                        else:
                            print(f"Fix failed — discarding\n{result2.stderr[-300:]}")
                            shutil.copy(TRAIN_PATH + ".bak", TRAIN_PATH)
                            continue
                    else:
                        print(f"Fix introduced syntax error: {syntax_err} — discarding")
                        shutil.copy(TRAIN_PATH + ".bak", TRAIN_PATH)
                        continue
                except Exception as e:
                    print(f"Fix attempt failed: {e} — discarding")
                    shutil.copy(TRAIN_PATH + ".bak", TRAIN_PATH)
                    continue
            else:
                print(f"Could not parse metric, no traceback found\n{output[-400:]}")
                shutil.copy(TRAIN_PATH + ".bak", TRAIN_PATH)
                continue

        elapsed = (datetime.now() - t0).seconds
        improved = score < best_score
        metric_name = _detect_metric(proposed)
        print(f"{metric_name}: {score:.4f} (best: {best_score:.4f}) "
              f"{'✅ IMPROVED' if improved else '❌ discarded'} [{elapsed}s]")

        # 6. Keep or discard
        if improved:
            best_score = score
            best_train_py = proposed
            _write(f"{RESULTS_PATH}/train_iter_{i+1}_{metric_name}_{score:.4f}.py", proposed)
            _write("/vol/best_train.py", proposed)
        else:
            shutil.copy(TRAIN_PATH + ".bak", TRAIN_PATH)

        # 7. Log
        entry = {
            "iter": i + 1,
            "change": change_desc,
            "val_bpb": score,
            "improved": improved,
            "best_val_bpb": best_score,
            "baseline": baseline_score,
            "elapsed_s": elapsed,
            "timestamp": datetime.now().isoformat()
        }
        history.append(entry)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        volume.commit()

    # Save final best
    _write("/vol/best_train.py", best_train_py)
    volume.commit()

    improvement = ((baseline_score - best_score) / baseline_score * 100) if baseline_score != float("inf") else 0
    print("\n" + "=" * 60)
    print(f"Loop completed")
    print(f"  Iterations:    {n_iters}")
    print(f"  Baseline:      {baseline_score:.4f}")
    print(f"  Best val_bpb:  {best_score:.4f}")
    print(f"  Improvement:   {improvement:.1f}%")
    print(f"  Best model:    /vol/best_train.py")
    print("=" * 60)

    return {"baseline": baseline_score, "best_val_bpb": best_score, "improvement_pct": improvement}

# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    repo_id: str = "sander-wood/irishman",
    column: str = "abc notation",
    local_file: str = "",
    iters: int = 50
):
    if not repo_id and not local_file:
        print("ERROR: provide --repo-id or --local-file")
        print("\nExamples:")
        print("  # HuggingFace dataset")
        print('  modal run autoresearch_text.py --repo-id "sander-wood/irishman" --column "abc notation"')
        print("\n  # Local TXT file")
        print('  modal volume put autoresearch-text-vol my_data.txt /vol/uploads/my_data.txt')
        print('  modal run autoresearch_text.py --local-file "my_data.txt"')
        print("\n  # Local CSV file")
        print('  modal volume put autoresearch-text-vol data.csv /vol/uploads/data.csv')
        print('  modal run autoresearch_text.py --local-file "data.csv" --column "text_column"')
        return

    print(f"Step 1: Preparing dataset...")
    download_dataset.remote(repo_id=repo_id, column=column, local_file=local_file)

    print(f"Step 2: Launching research loop ({iters} iterations)...")
    result = research_loop.remote(n_iters=iters)
    print(f"Final result: {result}")
