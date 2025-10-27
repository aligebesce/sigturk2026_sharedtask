# SIGTURK 2026 — Shared Task: Evaluation Script and Development Data

This repository provides a **command‑line evaluator** for the SIGTURK 2026 shared task and Development Data. It supports three subtasks on paired JSON files containing **golden set** annotations and **system predictions**:

1. **Term Detection (English word‑level)** — Precision/Recall/F1 (Macro + Micro)  
2. **Term Correction (Exact match)** — Accuracy (Macro + Micro)  
3. **End‑to‑End (sentence quality)** — **Sentence‑level** BLEU and chrF (Macro **only**) computed with `sacrebleu`

The evaluator **strictly aligns** records using `(paragraph_id, sentence_id)` and **fails fast** on any mismatch, duplicates, or missing IDs.

---

## Directory Layout

```
sigturk2026_sharedtask/
├─ dev_data/
│  ├─ subtask_1.json
│  ├─ subtask_2.json
│  ├─ subtask_3.json
│  └─ terimler_org_data.json
├─ evaluation/
│  ├─ requirements.txt
│  ├─ eval.py
│  ├─ subtask1_term_detection/
│  │  ├─ golden_set.json
│  │  └─ predictions.json
│  ├─ subtask2_term_correction/
│  │  ├─ golden_set.json
│  │  └─ predictions.json
│  └─ subtask3_end2end/
│     ├─ golden_set.json
│     └─ predictions.json
└─ README.md
```

## Quick Start

### 1) Requirements

```bash
pip install -r evaluation/requirements.txt
```

**requirements.txt** should contain at minimum:
```
sacrebleu
```
(Any 3.x version is fine.)

### 2) Run the evaluator

Use the same interface for all subtasks by switching `--task` and the JSON file paths.

```bash
# Term Detection (EN word-level)
python evaluation/eval.py --task detection --golden_set evaluation/subtask1_term_detection/golden_set.json --predictions evaluation/subtask1_term_detection/predictions.json

# Term Correction (Exact Match)
python evaluation/eval.py --task correction --golden_set evaluation/subtask2_term_correction/golden_set.json --predictions evaluation/subtask2_term_correction/predictions.json

# End-to-End (Sentence BLEU/chrF; sentence-level mean only)
python evaluation/eval.py --task end2end  --golden_set evaluation/subtask3_end2end/golden_set.json --predictions evaluation/subtask3_end2end/predictions.json
```

---

## JSON Schemas (by subtask)

Each JSON file can be **an object** or **an array of objects** (both are supported). All entries **must** include integer `paragraph_id` and `sentence_id`. The evaluator pairs rows by these IDs and **exits with an error** if there are extra or missing pairs.

### Common Keys

- `paragraph_id` *(int, required)*
- `sentence_id` *(int, required)*

### Subtask 1 — Term Detection (EN word‑level)
Gold and predictions both use **English spans** over the **source sentence**.

**golden_set.json**
```json
{
  "paragraph_id": 3,
  "sentence_id": 2,
  "source_sentence": "We discuss p-branes, plane waves, ...",
  "term_pairs": [
    {"en_start": 3, "en_end": 10},
    {"en_start": 12, "en_end": 20}
  ]
}
```
**predictions.json**
```json
{
  "paragraph_id": 3,
  "sentence_id": 2,
  "term_pairs": [
    {"en_start": 3, "en_end": 10}
  ]
}
```

- Spans are interpreted as **half‑open** character ranges `[start, end)` in the `source_sentence`.
- The evaluator normalizes spans (clamps to sentence length, fixes reversed indices, drops empty/invalid, deduplicates).
- Tokens are detected with a Unicode `\w+` regex. Token labels (0/1) are derived by **strict interval overlap** with spans.

**Metrics printed**
- **Macro** (mean across items): Precision / Recall / F1  
- **Micro** (pooled over corpus): TP / FP / TN / FN + Precision / Recall / F1

### Subtask 2 — Term Correction (Exact Match)

**golden_set.json**
```json
{
  "paragraph_id": 3,
  "sentence_id": 2,
  "source_sentence": "We discuss p-branes, plane waves, ...",
  "term_pairs": [
    {"en": "p-branes", "en_start": 10, "en_end": 17, "correction": "p-zarları"}
  ]
}
```
**predictions.json**
```json
{
  "paragraph_id": 3,
  "sentence_id": 2,
  "term_pairs": [
    {"en": "p-branes", "en_start": 10, "en_end": 17, "correction": "p-zarları"}
  ]
}
```

- Keys used for alignment per term: **(clamped `en_start`, `en_end`, normalized `en`)**.  
- **Exact match** on the string value of `correction` (after normalization).

**Metrics printed**
- **Macro Accuracy** (mean per item)  
- **Micro totals**: Correct / Total + Micro Accuracy

### Subtask 3 — End‑to‑End (Sentence BLEU/chrF)

This repository’s *workshop edition* compares **only the edited target sentences** and reports **sentence-level means** (no corpus scores).

**golden_set.json**
```json
{
  "paragraph_id": 3,
  "sentence_id": 2,
  "edited_target_sentence": "Düzlem dalgaları, p-zarları, ..."
}
```
**predictions.json**
```json
{
  "paragraph_id": 3,
  "sentence_id": 2,
  "edited_target_sentence": "Düzlem dalgaları, p-branları, ..."
}
```

**Fields used**
- Gold: `edited_target_sentence`  
- Pred: `edited_target_sentence`

**Metrics printed**
- `Mean_BLEU` — mean of **sentence BLEU** (sacrebleu) over all paired items  
- `Mean_chrF` — mean of **sentence chrF** (sacrebleu) over all paired items

> If either file lacks pairs after alignment or the sentences are missing, the evaluator prints a friendly message and exits.

---

## Design Notes

- **Strict pairing**: `_pair_rows` enforces a 1:1 mapping by `(paragraph_id, sentence_id)`, errors on duplicates, extras, or missing entries.  
- **Normalization**: text is NFKC‑normalized, lowercased, and whitespace‑collapsed; spans are clamped and deduped.  
- **Tokenization** for detection uses `\w+` to get word spans and then converts span overlaps to token labels.  
- **Robust I/O**: both arrays and single JSON objects are accepted (`as_list`).

---

## Examples

Using the sample provided in the prompt (one pair only):

```bash
python evaluation/eval.py --task end2end  --golden_set evaluation/subtask3_end2end/golden_set.json --predictions evaluation/subtask3_end2end/predictions.json
```
Output (format):
```
== End-to-End ==
Items=1
Mean_BLEU=...
Mean_chrF=...
```

---

## Troubleshooting

~~- **“error: … requires sacrebleu”** — run `pip install sacrebleu`.  
- **“duplicate … in golden_set/predictions”** — ensure each `(paragraph_id, sentence_id)` is unique per file.  
- **“missing … in predictions” / “extra … in predictions”** — files must contain the **same set** of `(paragraph_id, sentence_id)`.  
- **Zero items evaluated** — check that required fields (`term_pairs`, `edited_target_sentence`, etc.) are present for your task.~~

---

## Contact

For any questions regarding the shared task, please contact: sigturk2026.sharedtask@gmail.com
