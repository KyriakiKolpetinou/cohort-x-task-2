# CohortX Task 2 — Reproduction (Gemma-2-9B, LB 0.80)

Challenge: https://www.kaggle.com/competitions/cohort-x-task-2/overview

Transform free-text clinical-trial eligibility criteria into structured
subject–predicate–object triples.

This repository reproduces our leaderboard submission
[`submission/submission_task2_gemma9b.csv`](submission/submission_task2_gemma9b.csv),
**public score 0.80**. The method is **retrieval-augmented few-shot prompting with a
frozen, pre-trained LLM — no fine-tuning, no external data** — and it runs **offline on
CPU within 16 GB RAM**, satisfying the challenge's hardware/reproducibility requirements.

- **Model:** Gemma-2-9B-it, Q4_K_M GGUF (~5.4 GB), run via `llama.cpp`.
- **Retriever:** PubMedBERT (frozen), for few-shot example selection.
- **Verified on CPU:** peak resident memory **9.35 GB** (< 16 GB), fully offline. See §6.

---

## 1. How it works

```
eligibility_criteria (free text)
   │
   ▼
PubMedBERT embedding ──► cosine retrieval over the 100 Train rows
   │                         │
   │                         ▼
   │                  k=5 nearest (criteria, structured) pairs,
   │                  reranked to prefer a similar criterion COUNT   ── few-shot examples
   ▼                         │
Gemma-2-9B-it (Q4_K_M GGUF, llama.cpp)  ◄── prompt: instruction + 5 example turns + query
   │   greedy decode, temperature 0
   ▼
deterministic post-processing (root-label convention, dedup, guarantee root triples)
   │
   ▼
structured triples  →  submission_task2_gemma9b.csv / .txt
```

**Design notes** (all in [`gemma9b_pipeline.py`](gemma9b_pipeline.py)):

- **Retrieval** — PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`)
  mean-pooled embeddings, cosine similarity over the 100 Train rows. Precomputed
  embeddings are cached in `train_index_task2.json` (rebuild with `--rebuild-index`).
- **Structural shot reranking** — among the nearest neighbours, prefer examples whose
  number of criteria is close to the query's (estimated from input bullets), so the
  few-shot examples teach the right output granularity.
- **Interleaved example formatting** (`_regroup_structured`) — each gold example is
  reordered so a criterion's reference triple is immediately followed by its expansion
  triples. Score-neutral (the metric is order-independent) but it teaches the model the
  interleaved pattern and prevents "reference starvation" (listing criteria with no
  expansions).
- **Gemma-2 chat prompt** — Gemma-2 has no system role, so the instruction is folded into
  the first user turn, using `<start_of_turn>`/`<end_of_turn>` markers.
- **Greedy decoding** (`temperature=0`) — deterministic given the model and prompt.
- **Post-processing** — copy the `"…Criteria"` vs `"…Criteria Set"` root-label convention
  from the top retrieved example (71/100 Train trials use "…Set"), near-duplicate dedup,
  guarantee the two root triples, cap at 120 triples.

### Data & model policy (compliance)

- The **only** task data used is `Task_2.xlsx` — the `Train` sheet is the few-shot
  retrieval pool, the `Test` sheet is the input. **No external datasets.**
- **Gemma-2-9B-it** and **PubMedBERT** are pre-trained HuggingFace models used **frozen**
  (no fine-tuning) — explicitly permitted by the rules.
- **WordNet** (`nltk_data/`) is used **only** by `eval_metric.py` for *local* scoring; it
  never touches the predictions or the submission file.

---

## 2. Repository contents

| Path | Purpose |
|---|---|
| `gemma9b_pipeline.py` | The full pipeline (retrieval + LLM + post-processing). |
| `eval_metric.py` | Local scorer — replicates the competition's Hungarian-matching algorithm with a WordNet proxy for FM3S (for local checks only). |
| `Task_2.xlsx` | Challenge data (`Train` = 100 rows, `Test` = 50 rows). |
| `train_index_task2.json` | Cached PubMedBERT embeddings of the Train sheet. |
| `run_cpu.sh` | Run on CPU, offline, with peak-RAM measurement. |
| `run_gpu.sh` | Optional GPU run (identical output, faster). |
| `requirements.txt` | Python dependencies. |
| `submission/submission_task2_gemma9b.csv` | The submitted predictions (LB 0.80). |
| `submission/submission_task2_gemma9b.txt` | Same predictions, triple-blocks joined by `\n\n`. |

The model, PubMedBERT, and WordNet are **not** in Git (too large) — download them once
with the commands in §4. They are git-ignored.

---

## 3. Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
# torch + llama-cpp-python CPU builds (if not already pulled by requirements.txt):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install llama-cpp-python
```

Tested with Python 3.12.

---

## 4. Download the offline assets (once, then no network needed)

```bash
# 1) Gemma-2-9B-it Q4_K_M GGUF (~5.4 GB)
mkdir -p models
curl -L -C - --retry 10 -o models/gemma-2-9b-it-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf?download=true"

# 2) PubMedBERT (into ./hf_cache via HF_HOME)
HF_HOME="$PWD/hf_cache" python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')"

# 3) WordNet (only needed for local scoring with eval_metric.py)
python -c "import nltk; nltk.download('wordnet', download_dir='nltk_data'); \
  nltk.download('omw-1.4', download_dir='nltk_data')"
```

Resulting layout:

```
models/gemma-2-9b-it-Q4_K_M.gguf     ~5.4 GB   (LLM)
hf_cache/                            ~440 MB   (PubMedBERT)
nltk_data/                           ~30 MB    (WordNet, scoring only)
```

---

## 5. Reproduce the submission

```bash
# Full 50-row Test submission on CPU (offline), with peak-RAM measurement:
./run_cpu.sh                 # writes submission_task2_gemma9b.csv + .txt

# Quick 3-row smoke test:
./run_cpu.sh --limit 3

# Optional — faster on an NVIDIA GPU (identical output at temperature 0):
./run_gpu.sh
```

Output files: `submission_task2_gemma9b.csv` (Test sheet with the `structured` column
filled) and `submission_task2_gemma9b.txt` (triples concatenated, blocks separated by a
blank line — the plain-text submission format).

Tunable env vars: `TASK2_GPU_LAYERS` (0 = CPU), `TASK2_THREADS`, `TASK2_NCTX`
(default 12288), `TASK2_KSHOTS` (default 5), `TASK2_GGUF` (model path override).

---

## 6. Reproducibility notes

**Verified CPU run** (this pipeline, `TASK2_GPU_LAYERS=0`, fully offline):

| Metric | Value |
|---|---|
| Peak resident memory | **9.35 GB** (within the 16 GB requirement) |
| Network | none (fully offline) |
| Output | well-formed triples, matches the submitted format |

**Determinism.** Decoding is greedy (`temperature=0`), so the pipeline is deterministic
given a fixed model file and `llama.cpp` backend. Note that CPU and GPU (and different
`llama.cpp` builds) can differ by a token here and there due to floating-point rounding —
the **triple structure and counts are preserved and the semantic-similarity score
reproduces**, but the output is not guaranteed byte-identical across different hardware.
The committed `submission/submission_task2_gemma9b.csv` is the exact file scored at 0.80.

**No time limit.** On CPU, a 9B model decodes at a few tokens/sec, so the full 50-row set
takes on the order of hours on a mid-range CPU. The competition requires CPU + 16 GB RAM
but sets no time limit, so a slow-but-complete CPU run satisfies the requirement. Use
`run_gpu.sh` for a fast run with identical output.
