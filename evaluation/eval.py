# -*- coding: utf-8 -*-
# =========================================================
# SIGTURK 2026 — CLI Evaluator (Paired files)
# Detection (EN word-level P/R/F1), Correction (Exact Match)
# End-to-End (BLEU, chrF via sacrebleu)
#
# WORKSHOP EDITION — STRICT IDs + MACRO (MEAN) METRICS
# - (paragraph_id, sentence_id) REQUIRED, unique, and matched.
# - Detection: Macro (mean per item) + Micro (corpus) metrics.
# - Correction: Macro (mean per item) + Micro accuracy.
# - End-to-End: Macro (mean sentence BLEU/chrF) + Micro (corpus).
# =========================================================

from __future__ import annotations
import argparse, json, re, sys, unicodedata
from typing import List, Tuple, Dict, Any, Iterable, Optional

# ---------- sacrebleu ----------
try:
    import sacrebleu
except Exception:
    sacrebleu = None


# ---------------- I/O ----------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def as_list(x: Any) -> List[Dict[str, Any]]:
    if x is None: return []
    if isinstance(x, list): return x
    if isinstance(x, dict): return [x]
    raise TypeError("JSON must be an object or an array of objects")


# ----------------- Normalization -----------------
def _norm_text(x: Any) -> str:
    """NFKC normalize, strip, lower, collapse spaces."""
    if x is None: return ""
    s = unicodedata.normalize("NFKC", str(x))
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _norm_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


# ----------------- Tokenization -----------------
WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    if not isinstance(text, str): return []
    return [(m.group(0), m.start(), m.end()) for m in WORD_RE.finditer(text)]


def normalize_spans(spans: Iterable[Tuple[int, int]], text_len: int) -> List[Tuple[int, int]]:
    """Clamp to [0, text_len], swap reversed, dedupe, drop empty."""
    cleaned = []
    for s, e in spans or []:
        si = _norm_int(s);
        ei = _norm_int(e)
        if si is None or ei is None:
            continue
        if si > ei: si, ei = ei, si
        si = max(0, min(si, text_len))
        ei = max(0, min(ei, text_len))
        if si < ei:
            cleaned.append((si, ei))
    return sorted(set(cleaned))


def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    # strict interval overlap, exclusive of endpoints
    return not (a[1] <= b[0] or b[1] <= a[0])


def labels_from_spans(tokens_spans: List[Tuple[str, int, int]], spans: List[Tuple[int, int]]) -> List[int]:
    spans = spans or []
    out = []
    for _, s, e in tokens_spans:
        t = (s, e)
        out.append(1 if any(overlap(t, sp) for sp in spans) else 0)
    return out


def extract_en_spans(term_pairs: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    out = []
    for d in term_pairs or []:
        s = _norm_int(d.get("en_start"))
        e = _norm_int(d.get("en_end"))
        if s is not None and e is not None:
            out.append((s, e))
    return out


# ----------------- Strict alignment -----------------
def _require_ids(rows: List[Dict[str, Any]], tag: str) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Ensure all rows have numeric (paragraph_id, sentence_id)."""
    idx: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for i, r in enumerate(rows):
        gid = _norm_int(r.get("paragraph_id"))
        sid = _norm_int(r.get("sentence_id"))
        if gid is None or sid is None:
            sys.stderr.write(
                f"error: every {tag} entry must include INTEGER 'paragraph_id' and 'sentence_id' "
                f"(missing/invalid at index {i}).\n"
            )
            sys.exit(1)
        key = (gid, sid)
        if key in idx:
            sys.stderr.write(f"error: duplicate (paragraph_id, sentence_id)=({gid},{sid}) in {tag}.\n")
            sys.exit(1)
        idx[key] = r
    return idx


def _pair_rows(golden_set_rows: List[Dict[str, Any]], ans_rows: List[Dict[str, Any]]) -> List[
    Tuple[Dict[str, Any], Dict[str, Any]]]:
    tmap = _require_ids(golden_set_rows, "golden_set")
    amap = _require_ids(ans_rows, "predictions")

    missing = [k for k in tmap if k not in amap]
    extra = [k for k in amap if k not in tmap]
    if missing:
        sys.stderr.write(f"error: {len(missing)} golden_set item(s) missing in predictions, e.g. {missing[:5]}\n")
        sys.exit(1)
    if extra:
        sys.stderr.write(f"error: {len(extra)} extra item(s) in predictions not found in golden_set, e.g. {extra[:5]}\n")
        sys.exit(1)

    # deterministic order
    keys_sorted = sorted(tmap.keys())
    return [(tmap[k], amap[k]) for k in keys_sorted]


# ---------------- Detection Evaluation (EN, word-level) ----------------
def confusion_from_labels(gold: List[int], sysl: List[int]) -> Tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for g, s in zip(gold, sysl):
        if g == 1 and s == 1:
            tp += 1
        elif g == 0 and s == 1:
            fp += 1
        elif g == 0 and s == 0:
            tn += 1
        elif g == 1 and s == 0:
            fn += 1
    return tp, fp, tn, fn


def _safe_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = (2 * P * R / (P + R)) if (P + R) else 0.0
    return P, R, F1


def evaluate_detection(golden_set_rows, ans_rows):
    pairs = _pair_rows(golden_set_rows, ans_rows)

    # Micro counters
    MIC_TP = MIC_FP = MIC_TN = MIC_FN = 0

    # Macro lists
    macro_P, macro_R, macro_F = [], [], []

    for gold_row, sys_row in pairs:
        src = gold_row.get("source_sentence", "") or ""
        gold_terms = gold_row.get("term_pairs", []) or []
        sys_terms = sys_row.get("term_pairs", []) or []

        en_toks = tokenize_with_spans(src)
        gold_sp = normalize_spans(extract_en_spans(gold_terms), len(src))
        sys_sp = normalize_spans(extract_en_spans(sys_terms), len(src))
        gold_lab = labels_from_spans(en_toks, gold_sp)
        sys_lab = labels_from_spans(en_toks, sys_sp)

        tp, fp, tn, fn = confusion_from_labels(gold_lab, sys_lab)
        MIC_TP += tp;
        MIC_FP += fp;
        MIC_TN += tn;
        MIC_FN += fn

        P, R, F1 = _safe_prf(tp, fp, fn)
        macro_P.append(P);
        macro_R.append(R);
        macro_F.append(F1)

    # Macro = mean of per-item metrics
    MacP = sum(macro_P) / len(macro_P) if macro_P else 0.0
    MacR = sum(macro_R) / len(macro_R) if macro_R else 0.0
    MacF = sum(macro_F) / len(macro_F) if macro_F else 0.0

    # Micro from pooled counts
    MicP, MicR, MicF = _safe_prf(MIC_TP, MIC_FP, MIC_FN)

    print("== Detection (EN, word-level) ==")
    print(f"Items={len(pairs)}")
    print(f"Macro_Precision={MacP:.6f} Macro_Recall={MacR:.6f} Macro_F1={MacF:.6f}")
    print(f"Micro_TP={MIC_TP} Micro_FP={MIC_FP} Micro_TN={MIC_TN} Micro_FN={MIC_FN}")
    print(f"Micro_Precision={MicP:.6f} Micro_Recall={MicR:.6f} Micro_F1={MicF:.6f}")


# ---------------- Correction Evaluation (Exact Match) ----------------
def _term_key(d: Dict[str, Any], src_text: str) -> Tuple[int, int, str]:
    """Alignment key: (clamped en_start, en_end, normalized en surface)."""
    en = _norm_text(d.get("en", ""))
    s = _norm_int(d.get("en_start"))
    e = _norm_int(d.get("en_end"))
    if isinstance(s, int) and isinstance(e, int):
        s2 = max(0, min(s, len(src_text)))
        e2 = max(0, min(e, len(src_text)))
        return (s2, e2, en)
    return (-1, -1, en)


def evaluate_correction(golden_set_rows, ans_rows):
    pairs = _pair_rows(golden_set_rows, ans_rows)

    # Macro: per-item accuracies
    per_item_acc = []
    # Micro totals
    MIC_correct = 0
    MIC_total = 0
    items_evaluated = 0

    for gold_row, sys_row in pairs:
        src = gold_row.get("source_sentence", "") or ""
        gmap = {_term_key(g, src): _norm_text(g.get("correction", ""))
                for g in (gold_row.get("term_pairs", []) or [])
                if _norm_text(g.get("correction", ""))}
        smap = {_term_key(s, src): _norm_text(s.get("correction", ""))
                for s in (sys_row.get("term_pairs", []) or [])}

        total = len(gmap)
        if total == 0:
            # skip in macro mean (no evaluable corrections in this item)
            continue

        correct = sum(1 for k, gold_corr in gmap.items() if smap.get(k, "") == gold_corr)

        MIC_correct += correct
        MIC_total += total
        items_evaluated += 1

        per_item_acc.append(correct / total)

    MacAcc = (sum(per_item_acc) / len(per_item_acc)) if per_item_acc else 0.0
    MicAcc = (MIC_correct / MIC_total) if MIC_total else 0.0

    print("== Correction (Exact Match) ==")
    print(f"Items_Evaluated={items_evaluated} (items with at least one gold correction)")
    print(f"Macro_Accuracy={MacAcc:.6f}")
    print(f"Micro_Correct={MIC_correct} Micro_Total={MIC_total}")
    print(f"Micro_Accuracy={MicAcc:.6f}")


# ---------------- End-to-End Evaluation ----------------
def evaluate_end2end(golden_set_rows, ans_rows):
    if sacrebleu is None:
        sys.stderr.write("error: end2end requires sacrebleu. pip install sacrebleu\n")
        sys.exit(1)

    # Strict 1:1 alignment by (paragraph_id, sentence_id)
    pairs = _pair_rows(golden_set_rows, ans_rows)

    # Use ONLY edited_target_sentence from both sides
    refs = [(g.get("edited_target_sentence") or "") for g, _ in pairs]
    hyps = [(s.get("edited_target_sentence") or "") for _, s in pairs]

    if not refs or not hyps:
        print("== End-to-End (sacrebleu) ==")
        print("No valid sentence pairs found.")
        return

    chrf = sacrebleu.CHRF()

    bleu_scores = []
    chrf_scores = []
    for hyp, ref in zip(hyps, refs):
        bleu_scores.append(sacrebleu.sentence_bleu(hyp, [ref]).score)
        chrf_scores.append(chrf.sentence_score(hyp, [ref]).score)

    mean_bleu = sum(bleu_scores) / len(bleu_scores)
    mean_chrf = sum(chrf_scores) / len(chrf_scores)

    print("== End-to-End ==")
    print(f"Items={len(pairs)}")
    print(f"Mean_BLEU={mean_bleu:.6f}")
    print(f"Mean_chrF={mean_chrf:.6f}")


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="SIGTURK 2026 Evaluator (Paired Files, Macro Metrics)")
    ap.add_argument("--task", required=True, choices=["detection", "correction", "end2end"])
    ap.add_argument("--golden_set", required=True, help="Gold golden_set JSON (must have paragraph_id & sentence_id)")
    ap.add_argument("--predictions", required=True, help="System predictions JSON (must have paragraph_id & sentence_id)")
    args = ap.parse_args()

    golden_set_rows = as_list(load_json(args.golden_set))
    ans_rows = as_list(load_json(args.predictions))

    if args.task == "end2end" and sacrebleu is None:
        sys.stderr.write("error: end2end requires sacrebleu. pip install sacrebleu\n")
        sys.exit(1)

    if args.task == "detection":
        evaluate_detection(golden_set_rows, ans_rows)
    elif args.task == "correction":
        evaluate_correction(golden_set_rows, ans_rows)
    else:  # end2end
        evaluate_end2end(golden_set_rows, ans_rows)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"error: {e}\n")
        sys.exit(1)
