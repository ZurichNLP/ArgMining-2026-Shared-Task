# ArgMine 2026: Argument Mining in UN Resolutions

Shared Task at the 13th Workshop on Argument Mining and Reasoning ([ArgMining @ ACL 2026](https://argmining-org.github.io/2026/))

## Overview

UN resolutions and recommendations are among the most structurally constrained argumentative texts in international discourse. Their paragraphs follow a strict division: **preambular** paragraphs establish the rationale ("Noting that…", "Considering that…"), while **operative** paragraphs enact decisions ("Recommends…", "Urges…"). This formulaic structure encodes reasoning patterns that are rarely studied computationally.

ArgMine 2026 challenges participants to reconstruct this argumentative structure at the paragraph level — classifying paragraph types, assigning thematic policy tags, and predicting semantic relations between paragraphs — using only open-weight language models with at most 8B parameters.

The test corpus consists of 45 documents from the UNESCO International Bureau of Education's **International Conference on Education (1934–2008)**, spanning nearly a century of multilingual educational policy discourse.

---

## Task

The shared task comprises two subtasks, each with two components.

### Subtask 1: Paragraph Classification

Given a UN resolution in French (with English machine translations provided), classify each paragraph along two dimensions:

**1a. Paragraph Type**

Assign one of three types to each paragraph:

| Type | Description | Signal words |
|---|---|---|
| `preambular` | Justifies or motivates the resolution | *Considering, Noting, Recognizing, Recalling, Aware that…* |
| `operative` | Enacts a decision or recommendation | *Recommends, Urges, Invites, Decides, Requests…* |
| `null` | Structural paragraph (title, header, preamble opener) | — |

**1b. Education Policy Tags**

Assign zero or more thematic tags from the **IBE education dimensions ontology** (143 tags across 15 dimensions) to each paragraph. Tags cover ISCED levels, pedagogical approaches, curriculum content, populations, governance, infrastructure, and more. The full ontology is in [`data/education_dimensions_updated.csv`](data/education_dimensions_updated.csv).

---

### Subtask 2: Argumentative Relation Prediction

Identify semantic relationships between paragraphs within a document.

**2a. Related Paragraph Pair Identification**

For each paragraph, predict which earlier paragraphs it is semantically related to. Relations are directed: a later paragraph references an earlier one. Pairs are evaluated as unordered sets (directionality is not scored).

**2b. Relation Label Classification**

For each identified related pair, assign one of four relation types:

| Label | Description |
|---|---|
| `complemental` | One paragraph adds parallel information to themes in another |
| `modifying` | One paragraph narrows, qualifies, or adjusts themes in another |
| `supporting` | One paragraph provides evidence or examples that reinforce another |
| `contradictive` | One paragraph introduces tension or contradiction with another |

Label distribution in the test corpus: `modifying` (55%), `complemental` (26%), `none` (13%), `supporting` (5%), `contradictive` (<1%).

---

## Data

**Dataset on HuggingFace:** [ZurichNLP/ArgMining-2026-UZH-Shared-Task](https://huggingface.co/datasets/ZurichNLP/ArgMining-2026-UZH-Shared-Task)

| Split | Source | Size | Labels |
|---|---|---|---:|
| Train | UN-RES corpus (Gao et al., 2025), UN General Assembly resolutions | 2,695 documents | raw text only |
| Test (participant release) | UNESCO IBE International Conferences on Education (1934–2008) | 92 resolutions | none |

All documents are in **French** with machine-generated English translations (`gpt-4.1-mini` for test, `Helsinki-NLP/opus-mt-fr-en` for train).

### JSON Format

Each document is a single `.json` file. The system must fill in the fields marked with `*`:

```json
{
  "TEXT_ID": "ICPE-25-1962_RES1-FR_res_54",
  "RECOMMENDATION": 54,
  "TITLE": "LA PLANIFICATION DE L'ÉDUCATION",
  "METADATA": {
    "structure": {
      "doc_title": "ICPE-25-1962_RES1-FR",
      "nb_paras": 58,
      "preambular_para": [],   // * list of preambular paragraph numbers (int)
      "operative_para":  [],   // * list of operative paragraph numbers (int)
      "think": ""              // * reasoning trace (string)
    }
  },
  "body": {
    "paragraphs": [
      {
        "para_number": 3,
        "para":    "Que, par ailleurs, elle ne peut...",
        "para_en": "Moreover, it can no longer limit...",
        "type":         "preambular",    // * "preambular" | "operative" | null
        "tags":         ["CUR_CONTENT"], // * list of tag codes (may be empty)
        "matched_paras": {"2": "complemental"}, // * {para_number (str): relation_label}
        "think": ""                      // * reasoning trace (string)
      }
    ]
  }
}
```

**Relation label values:** `"complemental"`, `"modifying"`, `"supporting"`, `"contradictive"`

### Ground Truth Construction

The ground truth is **silver-standard** (LLM-generated) with human validation on subsets:

| Annotation | Method | Human validation |
|---|---|---|
| Paragraph types | `gpt-5.4-mini` (temp=0, prompt v10) | 97.0% accuracy on 363 paragraphs (11 docs); 97.4% on 352 paragraphs (7 docs) |
| Tags | `gpt-5.4` ∩ `claude-opus-4.6` intersection | Expert review of 178 paragraphs |
| Relations | `gpt-5.4` (best of 6 prompt iterations) | Two-round annotation of 300 pairs; κ = 0.540 (honest ceiling), κ = 0.664 (all pairs) |

Human expert annotations are included in [`human-annotations/`](human-annotations/).

---

## Evaluation

### Setup

```bash
pip install -r requirements.txt
```

### Running the Evaluation

Evaluate a submission directory against the silver-standard:

```bash
python evaluate.py --submission path/to/submission/dir
```

Evaluate against the expert human gold subsets:

```bash
python evaluate.py --submission path/to/submission/dir --human-gold
```

Show per-tag F1 and paragraph-level warnings:

```bash
python evaluate.py --submission path/to/submission/dir --verbose
```

### Metrics

| Task | Metric |
|---|---|
| ST1a — Paragraph type | Macro F1 (`preambular`, `operative`) |
| ST1b — Tags | Micro F1 and macro F1 (multi-label) |
| ST2a — Pair identification | Precision, Recall, F1 (unordered pairs) |
| ST2b — Relation labels | F1 per relation type (on correctly identified pairs) |

**Final ranking formula:** `0.4 × F1-rank(ST1) + 0.6 × F1-rank(ST2)`, combined with LLM-as-judge rank (weight 0.4).

---

## Results

Teams were restricted to open-weight models with ≤ 8B parameters. The final ranking combines F1-based rank and LLM-as-judge rank (0.6 / 0.4 weighting).

### Silver-Standard Evaluation

Ordered by final rank. μF1 = micro-averaged; mF1 = macro-averaged. Pair F1 measures unordered pair identification; Rel F1 is weighted F1 over relation labels on correctly identified pairs only.

| # | Team | Type mF1 | Tag μF1 | Tag mF1 | Pair F1 | Rel F1 |
|---|---|---|---|---|---|---|
| 1 | LLM-Instruct | 0.815 | 0.396 | **0.294** | 0.322 | 0.366 |
| 2 | Prompteam | 0.587 | 0.226 | 0.169 | 0.222 | 0.413 |
| 3 | Argchestrators | **0.936** | 0.327 | 0.285 | 0.206 | **0.440** |
| 3 | HybridArguer | 0.891 | 0.380 | 0.224 | 0.279 | 0.389 |
| 5 | POINTERS | 0.762 | **0.459** | **0.357** | **0.330** | 0.286 |
| 6 | LLM-Instruct-2 | 0.891 | 0.329 | 0.254 | 0.202 | 0.350 |
| 7 | ResolveNow | 0.910 | 0.344 | 0.236 | — | — |
| 8 | TypeCoT | 0.913 | 0.280 | 0.278 | 0.123 | 0.329 |
| 9 | Ockham | 0.445 | 0.205 | 0.045 | 0.289 | 0.328 |

### Final Rankings

| Team | F1 Rank | LLM-Judge Rank | Final Rank |
|---|---|---|---|
| **LLM-Instruct** | 1 | 5 | **1** |
| Prompteam | 5 | 1 | 2 |
| Argchestrators | 2 | 6 | 3 |
| HybridArguer | 4 | 3 | 3 |
| POINTERS | 3 | 9 | 5 |
| LLM-Instruct-2 | 7 | 4 | 6 |
| ResolveNow | 9 | 2 | 7 |
| TypeCoT | 6 | 8 | 8 |
| Ockham | 8 | 7 | 9 |

**Human ceiling (ST2 relations):** κ = 0.540 on 233 blind re-annotated pairs. The best system reached κ = 0.354 (65% of the human ceiling).

**Key findings:**
- Type classification is largely solved by deterministic lexical rules; every team using bilingual pattern matching achieves ≥ 0.910 F1.
- Tag prediction remains the hardest subtask; retrieval-based pre-filtering of the tag space systematically hurts rare-label recall.
- All teams over-predict argumentative pairs regardless of architecture; the precision–recall gap reflects an inherent LLM tendency to over-connect.
- F1 and LLM-as-judge rankings diverge substantially: Prompteam ranks 5th on F1 but 1st on reasoning quality; POINTERS ranks 3rd on F1 but 9th on judge scores.

Full results: [`results/evaluation-results.csv`](results/evaluation-results.csv)

---

## Annotation Prompts

The prompts used to construct the ground truth are in [`prompts/`](prompts/). We iterated through several relation and type prompts; the final versions (`prompt-type-v10.txt`, `prompt-relations-v6.txt`) are included. The best type prompt uses monotonicity constraints and structural rules; the best relation prompt uses a content-type decision tree (Principled / Evidentiary / Propositional) to determine relation labels, with worked examples for each type combination.

---

## Citations

If you use this dataset or evaluation code, please cite our overview paper:

```bibtex
@inproceedings{shaitarova-etal-2026-argmine,
  title     = {Overview of the {UZH} Shared Task 2026 on Reconstructing the Reasoning in {United Nations} Resolutions},
  author    = {Shaitarova, Anastassia and Gao, Yingqiang and Rezkellah, Fatma-Zohra and Gubelmann, Reto and Montjourid{\`e}s, Patrick},
  booktitle = {Proceedings of the 13th Workshop on Argument Mining and Reasoning},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
}
```

The citation for the training data:

```bibtex
@inproceedings{gao2025spiritrag,
  title={{SpiritRAG: A Q\&A System for Religion and Spirituality in the United Nations Archive}},
  author={Gao, Yingqiang and Winiger, Fabian and Montjourides, Patrick and Shaitarova, Anastassia and Gu, Nianlong and Peng-Keller, Simon and Schneider, Gerold},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  pages={26--41},
  year={2025}
}
```

---

## Organizers

[Anastassia Shaitarova](https://github.com/shaitarAn)

[Yingqiang Gao](https://github.com/CharizardAcademy)

[Reto Gubelmann](https://retogubelmann.net/)

[Patrick Montjouridès](https://www.ife.uzh.ch/en/research/ydesen/Staff/montjouridespatrick.html)

[Fatma-Zohra Rezkellah](https://www.cl.uzh.ch/de/about-us/people/team/compling/rezkellah.html)

Questions and feedback: open an [issue](../../issues) or contact us at the addresses listed in the paper.
