"""
CohortX Task 2 — Local evaluation metric.

Replicates the scoring algorithm described in the competition rules:

  1. Split a solution row (l1) and a submission row (l2) into lists of triples
     (t1, t2) using "\\n\\n", then split each triple into 3 fields using " — ".
  2. Build a similarity matrix where cell (i, j) is the mean of
     sim(e1[k], e2[k]) for k = 0,1,2 (the three fields).
  3. Convert to a cost matrix (cost = -sim) and run the Hungarian algorithm
     (scipy.optimize.linear_sum_assignment) for the optimal one-to-one matching.
  4. Collect matched similarities, pad with zeros for unmatched elements when the
     lists differ in length, and return the mean as the row score.
  5. The overall score is the mean of per-row scores.

IMPORTANT — about the field similarity function `sim`:
The organizers use an "adapted FM3S ontology-based semantic similarity metric"
driven by WordNet 3.0. FM3S (Hadj Taieb et al.) is an information-content based
measure whose exact implementation is not public here. This module therefore uses
a WordNet-based phrase similarity *proxy* (`wordnet_phrase_sim`) so we can measure
RELATIVE improvement between our own submissions. Absolute numbers will differ
from the leaderboard, but improvements here should track improvements there.

Set the env var TASK2_SIM=exact to use a stricter token-overlap fallback only
(no WordNet) for a quick sanity baseline.

Usage:
    python eval_metric.py --pred submission_xxx.csv --gold Task_2.xlsx --sheet Train
    # or compare a predictions column against the structured column in one xlsx
"""
import os, re, argparse, functools
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# ── WordNet setup (offline) ───────────────────────────────────────────────────
_NLTK_DIR = os.path.join(os.path.dirname(__file__), 'nltk_data')
if os.path.isdir(_NLTK_DIR):
    import nltk
    if _NLTK_DIR not in nltk.data.path:
        nltk.data.path.insert(0, _NLTK_DIR)

_wn = None
def _get_wn():
    global _wn
    if _wn is None:
        from nltk.corpus import wordnet as wn
        _ = wn.synsets('test')  # trigger load
        _wn = wn
    return _wn

_TOKEN_RE = re.compile(r"[A-Za-z]+")

def _tokens(s: str):
    return [t.lower() for t in _TOKEN_RE.findall(s)]

@functools.lru_cache(maxsize=200_000)
def _token_sim(a: str, b: str) -> float:
    """WordNet Wu-Palmer similarity between two single tokens (best synset pair)."""
    if a == b:
        return 1.0
    wn = _get_wn()
    sa, sb = wn.synsets(a), wn.synsets(b)
    if not sa or not sb:
        return 1.0 if a == b else 0.0
    best = 0.0
    for x in sa[:4]:
        for y in sb[:4]:
            try:
                v = x.wup_similarity(y) or 0.0
            except Exception:
                v = 0.0
            if v > best:
                best = v
    return best

def wordnet_phrase_sim(a: str, b: str) -> float:
    """Symmetric best-match token alignment between two phrases, WordNet-backed."""
    a, b = a.strip(), b.strip()
    if a == b:
        return 1.0
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        # fall back to raw string equality for numeric/symbolic fields
        return 1.0 if a.lower() == b.lower() else 0.0

    def directional(src, tgt):
        return np.mean([max(_token_sim(s, t) for t in tgt) for s in src])

    return float((directional(ta, tb) + directional(tb, ta)) / 2.0)

def _overlap_sim(a: str, b: str) -> float:
    """Pure token-overlap (Jaccard) fallback — no WordNet."""
    a, b = a.strip().lower(), b.strip().lower()
    if a == b:
        return 1.0
    sa, sb = set(_tokens(a)), set(_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

_SIM = _overlap_sim if os.environ.get('TASK2_SIM') == 'exact' else wordnet_phrase_sim

# ── Triple parsing ──────────────────────────────────────────────────────────────
def _split_triples(block: str):
    triples = []
    for chunk in str(block).split('\n\n'):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip().strip('"').strip() for p in chunk.split(' — ')]
        if len(parts) == 3:
            triples.append(parts)
    return triples

def row_score(gold: str, pred: str) -> float:
    t1 = _split_triples(gold)   # solution
    t2 = _split_triples(pred)   # submission
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    n, m = len(t1), len(t2)
    sim = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            sim[i, j] = np.mean([_SIM(t1[i][k], t2[j][k]) for k in range(3)])
    ri, cj = linear_sum_assignment(-sim)
    matched = [sim[i, j] for i, j in zip(ri, cj)]
    denom = max(n, m)  # pad unmatched with zeros
    return float(sum(matched) / denom)

def corpus_score(golds, preds):
    return float(np.mean([row_score(g, p) for g, p in zip(golds, preds)]))

# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='CSV/xlsx with a "structured" column of predictions')
    ap.add_argument('--gold', required=True, help='xlsx/csv with the gold "structured" column')
    ap.add_argument('--sheet', default='Train', help='sheet name for gold (and pred if xlsx)')
    ap.add_argument('--n', type=int, default=0, help='limit to first N rows (0 = all)')
    args = ap.parse_args()

    def load(path, sheet):
        if path.endswith('.xlsx'):
            return pd.read_excel(path, sheet_name=sheet)
        return pd.read_csv(path)

    gold_df = load(args.gold, args.sheet)
    pred_df = load(args.pred, args.sheet)
    g = gold_df['structured'].astype(str).tolist()
    p = pred_df['structured'].astype(str).tolist()
    n = min(len(g), len(p))
    if args.n:
        n = min(n, args.n)
    g, p = g[:n], p[:n]

    scores = [row_score(gi, pi) for gi, pi in zip(g, p)]
    print(f"Rows scored: {n}")
    print(f"Mean score : {np.mean(scores):.4f}")
    print(f"Min / Max  : {np.min(scores):.4f} / {np.max(scores):.4f}")
    worst = np.argsort(scores)[:5]
    print("Worst 5 rows (index : score):", [(int(i), round(scores[i],3)) for i in worst])

if __name__ == '__main__':
    main()
