# ArgMine 2026: Argument Mining in UN Resolutions

**Shared Task at the 13th Workshop on Argument Mining and Reasoning ([ArgMining @ ACL 2026](https://argmining-org.github.io/2026/))**
---

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
| Test (participant release) | UNESCO IBE International Conferences on Education (1934–2008) | 92 resolutions/recommendations | none |
| Test (gold standard) | same as above | 92 resolutions/recommendations | types, tags, relations |

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

Final leaderboard (silver-standard evaluation). Teams were restricted to open-weight models with ≤ 8B parameters.

| Team | Type F1 | Tag Micro F1 | Tag Macro F1 | Pair P | Pair R | Pair F1 | Rel F1 |
|---|---|---|---|---|---|---|---|
| **Argchestrators** | **0.936** | 0.327 | 0.285 | 0.119 | 0.740 | 0.206 | **0.440** |
| ResolveNow | 0.910 | 0.344 | 0.236 | 0.000 | 0.000 | 0.000 | — |
| TypeCoT | **0.913** | 0.280 | 0.278 | 0.072 | 0.401 | 0.123 | 0.329 |
| LLM-Instruct (run 2) | 0.891 | 0.329 | 0.254 | 0.162 | 0.270 | 0.202 | 0.350 |
| HybridArguer | 0.891 | **0.380** | 0.224 | 0.173 | 0.713 | 0.279 | 0.389 |
| LLM-Instruct (run 1) | 0.815 | 0.396 | **0.294** | 0.205 | 0.748 | 0.322 | 0.366 |
| POINTERS | 0.762 | **0.459** | **0.357** | **0.208** | **0.796** | **0.330** | 0.286 |
| Prompteam | 0.587 | 0.226 | 0.169 | 0.136 | 0.611 | 0.222 | 0.413 |
| Ockham | 0.445 | 0.205 | 0.045 | 0.194 | 0.569 | 0.289 | 0.328 |

**Human ceiling (ST2):** κ = 0.540 on 233 re-annotated pairs. The best system reached κ = 0.354 (65% of human ceiling).

**Key findings:**
- Paragraph type classification is largely solvable by lexical heuristics (6 of 9 runs exceed 0.89 F1).
- Tag prediction remains difficult due to class imbalance and ontology coverage gaps.
- All teams over-predicted relations (high recall, low precision); label ambiguity between `complemental` and `modifying` constrains the ceiling.
- F1-based and LLM-as-judge rankings diverged substantially, suggesting complementary evaluation perspectives.

Full results: [`results/evaluation-results.csv`](results/evaluation-results.csv)

---

## Annotation Prompts

The prompts used to construct the ground truth are in [`prompts/`](prompts/). We iterated through several relation and type prompts; the final versions (`prompt-type-v10.txt`, `prompt-relations-v6.txt`) are included. The best type prompt uses monotonicity constraints and structural rules; the best relation prompt uses a content-type decision tree (Principled / Evidentiary / Propositional) to determine relation labels, with worked examples for each type combination.

---

## Citations

If you use this dataset or evaluation code, please cite our overview paper:

```bibtex
@inproceedings{shaitarova-etal-2026-argmine,
  title     = {{ArgMine} 2026: Reconstructing Argumentative Structure in {UN} Resolutions},
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

| Name | Affiliation |
|---|---|
| [Anastassia Shaitarova](https://github.com/shaitarova) | University of Zurich |
| Yingqiang Gao | University of Zurich |
| [Reto Gubelmann](https://reto.gubelmann.com) | University of Zurich |
| Patrick Montjouridès | UNESCO International Bureau of Education |
| Fatma-Zohra Rezkellah | University of Zurich |

Questions and feedback: open an [issue](../../issues) or contact us at the addresses listed in the paper.
