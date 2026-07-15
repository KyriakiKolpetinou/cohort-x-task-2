"""
CohortX Task 2 — submission pipeline (Gemma-2-9B-it, LB 0.80).

Retrieval-augmented few-shot prompting with a frozen, pre-trained LLM — no fine-tuning,
no external data. Runs offline on CPU within 16 GB RAM.

Steps per trial:
  1. PubMedBERT-embed the free-text criteria and cosine-retrieve the k=5 nearest
     (criteria, structured) pairs from the 100 Train rows, reranked to prefer a similar
     criterion count (TASK2_STRUCT_SHOTS=0 disables the rerank).
  2. Build a Gemma-2 chat prompt (instruction folded into the first user turn — Gemma-2
     has no system role — with <start_of_turn>/<end_of_turn> markers) and greedily decode
     with llama.cpp (temperature 0).
  3. Deterministic post-processing: copy the "…Criteria"/"…Criteria Set" root-label
     convention from the top shot, near-duplicate dedup, guarantee the two root triples.

The exact same code runs on GPU by setting TASK2_GPU_LAYERS>0 (identical output at
temperature 0, for speed only). Env vars: TASK2_GPU_LAYERS (0=CPU), TASK2_THREADS,
TASK2_NCTX (default 12288), TASK2_KSHOTS (default 5), TASK2_GGUF (model path),
TASK2_CHAT_FORMAT (auto/gemma/chatml).

Steps per trial:
  1. PubMedBERT-embed the free-text criteria and cosine-retrieve the k=5 nearest
     (criteria, structured) pairs from the 100 Train rows.
  2. STRUCTURAL SHOT SELECTION — among the semantically nearest training rows,
     prefer those whose criterion COUNT is close to the query's (estimated from
     input bullets), so the few-shot examples teach the right granularity.
     Toggle with TASK2_STRUCT_SHOTS=0.
  3. Build a Qwen ChatML prompt (system instruction + few-shot user/assistant
     turns + the query) and greedily decode with llama.cpp (temperature 0).
     n_ctx = 12288 so long trials with k=5 shots don't overflow the window.
  4. Deterministic post-processing: copy the "…Criteria" vs "…Criteria Set" root
     label convention from the top shot, near-duplicate dedup (case/whitespace),
     guarantee the two root triples, cap at MAX_TRIPLES.

Runs on CPU (TASK2_GPU_LAYERS=0) and fully offline. The exact same code runs on
GPU by setting TASK2_GPU_LAYERS>0 (that is how the 0.80 file was generated, for
speed only — the output is model-deterministic at temperature 0).

Env vars: TASK2_GPU_LAYERS (0=CPU), TASK2_THREADS, TASK2_NCTX (default 12288),
  TASK2_KSHOTS (default 5), TASK2_GGUF (model path override),
  TASK2_REPEAT_PENALTY (default 1.0), TASK2_STRUCT_SHOTS (default 1).
"""
import os, re, json, time, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

os.environ.setdefault('HF_HOME', os.path.join(HERE, 'hf_cache'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

MODEL_PATH    = os.environ.get(
    'TASK2_GGUF',
    os.path.join(HERE, 'models', 'gemma-2-9b-it-Q4_K_M.gguf'))
DATA_PATH     = os.path.join(HERE, 'Task_2.xlsx')
INDEX_PATH    = os.path.join(HERE, 'train_index_task2.json')
BIOBERT_MODEL = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'

GPU_LAYERS  = int(os.environ.get('TASK2_GPU_LAYERS', '0'))
N_THREADS   = int(os.environ.get('TASK2_THREADS', str(os.cpu_count() or 8)))
N_CTX       = int(os.environ.get('TASK2_NCTX', '12288'))          # was 8192 in v4
K_SHOTS     = int(os.environ.get('TASK2_KSHOTS', '5'))
MAX_TRIPLES = 120
MAX_GEN_TOKENS = int(os.environ.get('TASK2_MAXGEN', '2048'))
# NOTE: keep at 1.0 (no penalty) — same as MedGemma/v4. A penalty (tried 1.15)
# distorts the wording of object values that legitimately reuse tokens and LOWERED
# the held-out score (0.8328 → 0.7967). It does NOT meaningfully reduce repetition
# here; dedup + MAX_TRIPLES handle that.
REPEAT_PENALTY = float(os.environ.get('TASK2_REPEAT_PENALTY', '1.0'))
STRUCT_SHOTS   = os.environ.get('TASK2_STRUCT_SHOTS', '1') != '0'

# Chat wrapper: 'chatml' (Qwen), 'gemma' (Gemma-2), or 'auto' (from model filename).
# Gemma-2 has NO system role, so the instruction is folded into the first user turn.
def _detect_format():
    fmt = os.environ.get('TASK2_CHAT_FORMAT', 'auto')
    if fmt != 'auto':
        return fmt
    return 'gemma' if 'gemma' in os.path.basename(MODEL_PATH).lower() else 'chatml'
CHAT_FORMAT = _detect_format()

# ── Singletons ────────────────────────────────────────────────────────────────
_llm = _tokenizer = _bert_model = None
_index = _embeddings = None

def _get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        _llm = Llama(model_path=MODEL_PATH, n_ctx=N_CTX, n_threads=N_THREADS,
                     n_gpu_layers=GPU_LAYERS, verbose=False)
    return _llm

# ── PubMedBERT embeddings ─────────────────────────────────────────────────────
def _get_bert():
    global _tokenizer, _bert_model
    if _tokenizer is None:
        from transformers import AutoTokenizer, AutoModel
        _tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
        _bert_model = AutoModel.from_pretrained(BIOBERT_MODEL)
        _bert_model.eval()
    return _tokenizer, _bert_model

def _embed(text: str) -> np.ndarray:
    import torch
    tok, model = _get_bert()
    enc = tok(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    with torch.no_grad():
        out = model(**enc)
    mask = enc['attention_mask'].unsqueeze(-1).float()
    emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
    emb = emb.cpu().numpy()[0]
    return emb / max(np.linalg.norm(emb), 1e-9)

# ── Criterion-count helpers (for structural shot selection) ────────────────────
def _input_criterion_estimate(text: str) -> int:
    text = str(text); low = text.lower()
    m = re.search(r'exclusion criteria', low)
    inc = text[:m.start()] if m else text
    exc = text[m.start():] if m else ''
    def bullets(s):
        return len(re.findall(r'(?m)^\s*(?:[-*•·]|\d+[.)]|[a-z][.)])\s+\S', s))
    return bullets(inc) + bullets(exc)

def _shot_criterion_count(structured: str) -> int:
    inc = len(set(re.findall(r'—\s*"(Inclusion Criterion \d+)"', structured)))
    exc = len(set(re.findall(r'—\s*"(Exclusion Criterion \d+)"', structured)))
    return inc + exc

# ── Training index ────────────────────────────────────────────────────────────
def _build_index(train_df: pd.DataFrame):
    print("Building PubMedBERT embedding index…")
    records, embeddings = [], []
    for i, row in train_df.iterrows():
        crit = str(row['eligibility_criteria'])
        records.append({'idx': int(i), 'eligibility_criteria': crit,
                        'structured': str(row['structured'])})
        embeddings.append(_embed(crit).tolist())
        if len(records) % 10 == 0:
            print(f"  embedded {len(records)}/{len(train_df)}", flush=True)
    json.dump({'records': records, 'embeddings': embeddings}, open(INDEX_PATH, 'w'))
    print(f"Index saved → {INDEX_PATH}")
    return records, np.array(embeddings, dtype=np.float32)

def _get_index(train_df, rebuild=False):
    global _index, _embeddings
    if _index is not None and not rebuild:
        return _index, _embeddings
    if os.path.exists(INDEX_PATH) and not rebuild:
        data = json.load(open(INDEX_PATH))
        _index = data['records']
        _embeddings = np.array(data['embeddings'], dtype=np.float32)
        norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
        _embeddings = _embeddings / np.maximum(norms, 1e-9)
    else:
        _index, _embeddings = _build_index(train_df)
    return _index, _embeddings

def _retrieve(query, train_df, k=K_SHOTS, exclude_idx=-1):
    idx, embs = _get_index(train_df)
    qemb = _embed(query)
    scores = embs @ qemb
    ranked = [i for i in np.argsort(scores)[::-1] if idx[i]['idx'] != exclude_idx]

    if not STRUCT_SHOTS:
        return [idx[i] for i in ranked[:k]]

    # Structural rerank: take a semantic shortlist (3k), then prefer shots whose
    # criterion count is close to the query's estimated count. Blend the two ranks
    # so we never stray far from semantic relevance.
    shortlist = ranked[:max(k * 3, k)]
    q_cnt = _input_criterion_estimate(query)
    def blended(rank_pos, cand_i):
        sem = rank_pos                                  # 0 = most similar
        struct = abs(_shot_criterion_count(idx[cand_i]['structured']) - q_cnt)
        return 0.7 * sem + 0.3 * struct
    order = sorted(range(len(shortlist)),
                   key=lambda p: blended(p, shortlist[p]))
    chosen = [shortlist[p] for p in order[:k]]
    # keep most-similar-first ordering among the chosen for prompt stability
    chosen.sort(key=lambda i: ranked.index(i))
    return [idx[i] for i in chosen]

# ── Prompt (Qwen2.5 ChatML format) ─────────────────────────────────────────────
INSTRUCTION = """\
You are a clinical NLP assistant. Transform the free-text eligibility criteria of a clinical trial into a structured list of triples.

Rules:
1. Each triple has the form: "Subject" — "predicate" — "object"  (delimiter: space, em-dash —, space).
2. Separate every triple from the next with a blank line.
3. Always include these two root triples:
   "Study" — "has inclusion criteria" — "{inc_label}"
   "Study" — "has exclusion criteria" — "{exc_label}"
4. EVERY criterion in the text is either an inclusion or an exclusion criterion. Sections such as
   "Disease Characteristics", "Patient Characteristics", "Prior/Concurrent Therapy", "Age", etc. are
   INCLUSION requirements (unless they state who is excluded). Map everything onto {inc_label} /
   {exc_label}. NEVER invent other top-level sets.
5. Process the criteria ONE AT A TIME. For each criterion, immediately write its reference triple
   AND THEN its expansion triples, before moving to the next criterion:
   "{inc_label}" — "includes criterion" — "Inclusion Criterion 1"
   "Inclusion Criterion 1" — "<predicate>" — "<value>"
   "Inclusion Criterion 1" — "<predicate>" — "<value>"
   "{inc_label}" — "includes criterion" — "Inclusion Criterion 2"
   "Inclusion Criterion 2" — "<predicate>" — "<value>"
   ... (same for Exclusion Criterion N).
6. EVERY criterion MUST have at least one expansion triple describing its content. Do NOT output a
   bare list of "includes criterion" references with no expansions.
7. Choose predicates that precisely describe the clinical concept (e.g. "age", "condition",
   "disease", "procedure", "BMI", "consent", "timing", "diagnosis", "excludes patients with").
8. Preserve the original wording closely — do not paraphrase or invent facts.
9. For list items within a criterion, emit one triple per item with the same subject and predicate.
10. Output ONLY the triples. No commentary, no markdown.\
"""

def _root_labels(shots):
    top = shots[0]['structured'] if shots else ''
    if '"Inclusion Criteria Set"' in top or '"Exclusion Criteria Set"' in top:
        return 'Inclusion Criteria Set', 'Exclusion Criteria Set'
    return 'Inclusion Criteria', 'Exclusion Criteria'

def _regroup_structured(structured: str) -> str:
    """Reorder a gold structured block so each criterion's reference triple is
    immediately followed by its expansion triples (score-neutral — the metric is
    order-independent). This shows the model the interleaved pattern to imitate,
    instead of 'all references first', which led the 7B to dump references and stop."""
    triples = _parse_triples(structured)
    if not triples:
        return structured
    roots, refs, by_subject, ref_order, leftovers = [], {}, {}, [], []
    for s, p, o in triples:
        if p in ('has inclusion criteria', 'has exclusion criteria'):
            roots.append((s, p, o))
        elif p == 'includes criterion':
            refs.setdefault(o, []).append((s, p, o))
            if o not in ref_order:
                ref_order.append(o)
        else:
            by_subject.setdefault(s, []).append((s, p, o))
    used_subj = set()
    out = list(roots)
    for crit in ref_order:
        out.extend(refs[crit])
        if crit in by_subject:
            out.extend(by_subject[crit]); used_subj.add(crit)
    # nested-definition blocks (subjects that aren't numbered criteria) go last
    for subj, ts in by_subject.items():
        if subj not in used_subj:
            out.extend(ts)
    return '\n\n'.join(f'"{s}" — "{p}" — "{o}"' for s, p, o in out)

def _build_prompt_chatml(criteria_text, shots, inc_label, exc_label):
    # Qwen2.5 ChatML format: instruction as a system message, few-shot as
    # user/assistant turns, final user turn = the query. (Only the wrapper differs
    # from v6 — all pipeline LOGIC, shots, regrouping, post-proc are identical.)
    instr = INSTRUCTION.format(inc_label=inc_label, exc_label=exc_label)
    parts = [f"<|im_start|>system\n{instr}<|im_end|>\n"]
    for shot in shots:
        example_out = _regroup_structured(shot['structured'])
        parts.append(
            f"<|im_start|>user\nEligibility criteria:\n{shot['eligibility_criteria']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example_out}<|im_end|>\n")
    parts.append(
        f"<|im_start|>user\nEligibility criteria:\n{criteria_text}<|im_end|>\n"
        f"<|im_start|>assistant\n")
    return ''.join(parts)

def _build_prompt_gemma(criteria_text, shots, inc_label, exc_label):
    # Gemma-2 chat format: <start_of_turn>user / <start_of_turn>model turns, closed
    # with <end_of_turn>. Gemma-2 has NO system role, so the instruction is folded
    # into the FIRST user turn. llama.cpp prepends <bos> automatically. All pipeline
    # LOGIC (shots, regrouping, post-proc) is identical to the ChatML path.
    instr = INSTRUCTION.format(inc_label=inc_label, exc_label=exc_label)
    parts = []
    for i, shot in enumerate(shots):
        example_out = _regroup_structured(shot['structured'])
        user = f"Eligibility criteria:\n{shot['eligibility_criteria']}"
        if i == 0:
            user = f"{instr}\n\n{user}"
        parts.append(
            f"<start_of_turn>user\n{user}<end_of_turn>\n"
            f"<start_of_turn>model\n{example_out}<end_of_turn>\n")
    final_user = f"Eligibility criteria:\n{criteria_text}"
    if not shots:                      # no-shot fallback: instruction still needed
        final_user = f"{instr}\n\n{final_user}"
    parts.append(
        f"<start_of_turn>user\n{final_user}<end_of_turn>\n"
        f"<start_of_turn>model\n")
    return ''.join(parts)

def _build_prompt(criteria_text, shots, inc_label, exc_label):
    builder = _build_prompt_gemma if CHAT_FORMAT == 'gemma' else _build_prompt_chatml
    return builder(criteria_text, shots, inc_label, exc_label)

# ── Post-processing ─────────────────────────────────────────────────────────────
_TRIPLE_RE = re.compile(r'^"[^"]+"\s*—\s*"[^"]+"\s*—\s*"[^"]+"')

def _parse_triples(raw):
    triples = []
    for line in raw.split('\n'):
        line = line.strip()
        if not line:
            continue
        # strip a leading list-number the model sometimes adds, e.g. '"1. "Study"...'
        # or '3) "Study"...' → '"Study"...'
        line = re.sub(r'^"?\s*\d+[.)]\s*(?=")', '', line)
        line = re.sub(r'"\s*[-–—]+\s*"', '" — "', line)
        if _TRIPLE_RE.match(line):
            s, p, o = [x.strip().strip('"').strip() for x in line.split(' — ')[:3]]
            triples.append((s, p, o))
        else:
            parts = line.split(' — ')
            if len(parts) == 3:
                triples.append(tuple(x.strip().strip('"').strip() for x in parts))
    return triples

def _postprocess(raw, inc_label, exc_label):
    triples = _parse_triples(raw)

    def relabel(x):
        x = re.sub(r'\bInclusion Criteria(?: Set)?\b', inc_label, x)
        x = re.sub(r'\bExclusion Criteria(?: Set)?\b', exc_label, x)
        return x
    triples = [(relabel(s), p, relabel(o)) for s, p, o in triples]

    # near-duplicate dedup: key on case/whitespace-normalised fields
    def norm(t):
        return tuple(re.sub(r'\s+', ' ', f).strip().lower() for f in t)
    seen, uniq = set(), []
    for t in triples:
        k = norm(t)
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    triples = uniq

    roots = [('Study', 'has inclusion criteria', inc_label),
             ('Study', 'has exclusion criteria', exc_label)]
    body = [t for t in triples if t not in roots]
    triples = roots + body
    triples = triples[:MAX_TRIPLES]
    return '\n\n'.join(f'"{s}" — "{p}" — "{o}"' for s, p, o in triples)

# ── Core inference ──────────────────────────────────────────────────────────────
def generate_triples(criteria_text, shots):
    llm = _get_llm()
    inc_label, exc_label = _root_labels(shots)
    cur = list(shots)
    while cur:
        prompt = _build_prompt(criteria_text, cur, inc_label, exc_label)
        try:
            stops = (['<end_of_turn>', '<start_of_turn>'] if CHAT_FORMAT == 'gemma'
                     else ['<|im_end|>', '<|im_start|>'])
            res = llm(prompt, max_tokens=MAX_GEN_TOKENS, temperature=0.0,
                      repeat_penalty=REPEAT_PENALTY, stop=stops)
            return _postprocess(res['choices'][0]['text'].strip(), inc_label, exc_label)
        except ValueError as e:
            if 'exceed context window' in str(e) and len(cur) > 1:
                cur = cur[:-1]
            else:
                raise
    raise RuntimeError("Prompt too large even with 1 shot")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', nargs='?', type=int, const=20, default=None)
    ap.add_argument('--rebuild-index', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='submission_task2_gemma9b')
    args = ap.parse_args()

    train_df = pd.read_excel(DATA_PATH, sheet_name='Train')
    test_df  = pd.read_excel(DATA_PATH, sheet_name='Test')
    _get_index(train_df, rebuild=args.rebuild_index)

    print(f"Model: {os.path.basename(MODEL_PATH)} | format={CHAT_FORMAT} "
          f"gpu_layers={GPU_LAYERS} threads={N_THREADS} ctx={N_CTX} k={K_SHOTS} "
          f"repeat_penalty={REPEAT_PENALTY} struct_shots={STRUCT_SHOTS}")

    if args.validate is not None:
        n = args.validate
        target = train_df.iloc[:n].reset_index(drop=True)
        preds, golds = [], []
        for i, (orig_idx, row) in enumerate(target.iterrows(), 1):
            t0 = time.time()
            shots = _retrieve(row['eligibility_criteria'], train_df, k=K_SHOTS,
                              exclude_idx=orig_idx)
            pred = generate_triples(row['eligibility_criteria'], shots)
            preds.append(pred); golds.append(str(row['structured']))
            print(f"  [{i:3d}/{n}] {time.time()-t0:.0f}s", flush=True)
        from eval_metric import row_score
        scores = [row_score(g, p) for g, p in zip(golds, preds)]
        print(f"\n=== LOCAL SCORE (WordNet proxy) on {n} rows ===")
        print(f"mean={np.mean(scores):.4f}  min={np.min(scores):.4f}  max={np.max(scores):.4f}")
        pd.DataFrame({'eligibility_criteria': target['eligibility_criteria'],
                      'structured_gold': golds, 'structured_pred': preds,
                      'score': scores}).to_csv(
            os.path.join(HERE, f'{args.out}_validation.csv'), index=False)
        print(f"Saved → {args.out}_validation.csv")
        return

    target = test_df if not args.limit else test_df.iloc[:args.limit]
    print(f"Running on {len(target)} test rows …")
    preds = []
    for i, (_, row) in enumerate(target.iterrows(), 1):
        t0 = time.time()
        shots = _retrieve(row['eligibility_criteria'], train_df, k=K_SHOTS)
        preds.append(generate_triples(row['eligibility_criteria'], shots))
        print(f"  [{i:3d}/{len(target)}] {time.time()-t0:.0f}s", flush=True)

    csv_df = test_df.iloc[:len(preds)].copy()
    csv_df['structured'] = preds
    csv_path = os.path.join(HERE, f'{args.out}.csv')
    csv_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\nWrote → {csv_path}")
    txt_path = os.path.join(HERE, f'{args.out}.txt')
    open(txt_path, 'w', encoding='utf-8').write('\n\n'.join(preds))
    print(f"Wrote → {txt_path}")

if __name__ == '__main__':
    main()
