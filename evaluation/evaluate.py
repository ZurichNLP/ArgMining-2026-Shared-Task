"""
Evaluation script for ArgMine shared task submissions.

Two evaluation modes:

1. Silver-standard (default): compares against the full corpus ground truth in
   data/test-data-full-final/ (model-generated labels). Reports:
     - Paragraph type classification (preambular / operative): F1 per class, macro F1
     - Tag prediction (multi-label): micro F1, macro F1, per-tag F1
     - Related paragraph pair identification: precision, recall, F1 over unordered pairs
     - Relation label classification (complemental / contradictive / modifying / supporting):
       F1 per type on correctly identified pairs

2. Human gold (--human-gold): compares against the small expert-annotated subsets.
   Each task is evaluated independently against its own gold file:
     - Types: human-annotations/types/para-type-annos-363p-expert.csv  (363 paras, 11 files)
     - Tags:  human-annotations/tags/para-tags-annos-178p-silver-expert-reviewed.csv  (178 paras)
     - Relations: human-annotations/relations/sample-review-ground-truth-v2.csv  (300 pairs)
   Only paragraphs / pairs present in the respective gold file are evaluated.

Usage:
    python evaluate.py --submission path/to/submission/dir
    python evaluate.py --submission path/to/submission/dir --verbose
    python evaluate.py --submission path/to/submission/dir --gt-dir data/test-data-full-final
    python evaluate.py --submission path/to/submission/dir --human-gold
    python evaluate.py --submission path/to/submission/dir --human-gold --verbose
"""

import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

HUMAN_TYPES_CSVS   = [
    Path("human-annotations/types/para-type-annos-363p-expert.csv"),       # 11 files
    Path("human-annotations/types/para-type-annos-352p-annotator2.csv"),   # 7 files, no overlap
]
HUMAN_TAGS_CSV     = Path("human-annotations/tags/para-tags-annos-178p-silver-expert-reviewed.csv")
HUMAN_RELS_CSV     = Path("human-annotations/relations/sample-review-ground-truth-v2.csv")

FULL_DIR = Path("data/test-data-full-final")


def evaluate(submission_dir: Path, verbose: bool = False, gt_dir: Path = FULL_DIR):
    sub_files = {p.name: p for p in submission_dir.glob("*.json")}
    gt_files = {p.name: p for p in gt_dir.glob("*.json")}

    matched = sorted(set(sub_files) & set(gt_files))
    only_in_sub = set(sub_files) - set(gt_files)
    only_in_gt = set(gt_files) - set(sub_files)

    if only_in_sub:
        print(f"{len(only_in_sub)} file(s) in submission not in ground truth (ignored)")
    if only_in_gt:
        print(f"{len(only_in_gt)} file(s) missing from submission")

    print(f"Evaluating {len(matched)} matched files...\n")

    gt_types, sub_types = [], []
    gt_tags_list, sub_tags_list = [], []
    # Each entry is a set of (source_para_num, target_para_num) pairs for one file
    gt_pair_sets: list[set] = []
    sub_pair_sets: list[set] = []
    # Parallel label lists for pairs present in both GT and submission
    gt_rel_labels: list[str] = []
    sub_rel_labels: list[str] = []

    for fname in matched:
        with open(gt_files[fname],  encoding="utf-8") as f:
            gt = json.load(f)
        with open(sub_files[fname], encoding="utf-8") as f:
            sub = json.load(f)

        gt_paras = {p["para_number"]: p for p in gt["body"]["paragraphs"]}
        sub_paras = {p["para_number"]: p for p in sub["body"]["paragraphs"]}

        gt_pairs: set[tuple] = set()
        sub_pairs: set[tuple] = set()

        for num, gt_para in gt_paras.items():
            sub_para = sub_paras.get(num)
            if sub_para is None:
                if verbose:
                    print(f"  {fname}: missing para_number={num} in submission")
                continue

            gt_types.append(gt_para.get("type") or "none")
            sub_types.append(sub_para.get("type") or "none")

            gt_tags_list.append(set(gt_para.get("tags") or []))
            sub_tags_list.append(set(sub_para.get("tags") or []))

            # Collect related paragraph pairs as unordered (max, min) tuples
            # so that (5, "3") and (3, "5") are treated as the same pair
            gt_mp = gt_para.get("matched_paras") or {}
            sub_mp = sub_para.get("matched_paras") or sub_para.get("matched_pars") or {}
            for target_num in gt_mp:
                gt_pairs.add((max(num, int(target_num)), min(num, int(target_num))))
            for target_num in sub_mp:
                sub_pairs.add((max(num, int(target_num)), min(num, int(target_num))))
            # Collect labels for pairs both identified (using original directed keys for label lookup)
            for target_num in set(gt_mp) & set(sub_mp):
                gt_rel_labels.append(gt_mp[target_num])
                sub_rel_labels.append(sub_mp[target_num])

        gt_pair_sets.append(gt_pairs)
        sub_pair_sets.append(sub_pairs)

    # ------------------------------------------------------------------ #
    #  Type classification
    # ------------------------------------------------------------------ #
    print("=" * 50)
    print("PARAGRAPH TYPE CLASSIFICATION")
    print("=" * 50)
    print(classification_report(
        gt_types, sub_types,
        labels=["preambular", "operative"],
        digits=3,
        zero_division=0,
    ))

    # ------------------------------------------------------------------ #
    #  Tag prediction (multi-label)
    # ------------------------------------------------------------------ #
    print("=" * 50)
    print("TAG PREDICTION")
    print("=" * 50)

    # Fit on union of GT and sub labels so that extra predicted tags count as FP
    mlb = MultiLabelBinarizer()
    mlb.fit(gt_tags_list + sub_tags_list)
    gt_bin = mlb.transform(gt_tags_list)
    sub_bin = mlb.transform(sub_tags_list)

    if gt_bin.shape[1] == 0:
        print("  Micro F1: N/A (no tags in ground truth or submission)")
        print("  Macro F1: N/A")
    else:
        micro = f1_score(gt_bin, sub_bin, average="micro", zero_division=0)
        macro = f1_score(gt_bin, sub_bin, average="macro", zero_division=0)
        print(f"  Micro F1: {micro:.3f}")
        print(f"  Macro F1: {macro:.3f}")

        if verbose:
            per_tag = f1_score(gt_bin, sub_bin, average=None, zero_division=0)
            rows = [
                (tag, float(score), int(gt_bin[:, i].sum()))
                # type: ignore[arg-type]
                for i, (tag, score) in enumerate(zip(mlb.classes_, per_tag))
            ]
            print()
            print("  Per-tag breakdown:")
            for tag, score, support in sorted(rows, key=lambda x: -x[1]):
                print(f"    {tag:<30}  F1={score:.3f}  (support={support})")

    # ------------------------------------------------------------------ #
    #  Related paragraph pair identification
    # ------------------------------------------------------------------ #
    print("=" * 50)
    print("RELATED PARAGRAPH PAIR IDENTIFICATION")
    print("=" * 50)

    tp = fp = fn = 0
    for gt_pairs, sub_pairs in zip(gt_pair_sets, sub_pair_sets):
        tp += len(gt_pairs & sub_pairs)
        fp += len(sub_pairs - gt_pairs)
        fn += len(gt_pairs - sub_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  (TP={tp}, FP={fp}, FN={fn})")

    if verbose:
        print()
        print("  Per-file breakdown:")
        for fname, gt_pairs, sub_pairs in zip(matched, gt_pair_sets, sub_pair_sets):
            file_tp = len(gt_pairs & sub_pairs)
            file_fp = len(sub_pairs - gt_pairs)
            file_fn = len(gt_pairs - sub_pairs)
            missing = gt_pairs - sub_pairs
            extra = sub_pairs - gt_pairs
            if missing or extra:
                print(f"    {fname}:  TP={file_tp} FP={file_fp} FN={file_fn}")
                if missing:
                    print(f"      missed pairs: {sorted(missing)}")
                if extra:
                    print(f"      extra pairs:  {sorted(extra)}")

    # ------------------------------------------------------------------ #
    #  Relation label classification (on correctly identified pairs)
    # ------------------------------------------------------------------ #
    print("=" * 50)
    print("RELATION LABEL CLASSIFICATION")
    print("=" * 50)

    if not gt_rel_labels:
        print("  No overlapping pairs to evaluate labels on.")
    else:
        rel_types = ["complemental", "contradictive", "modifying", "supporting"]
        print(classification_report(
            gt_rel_labels, sub_rel_labels,
            labels=rel_types,
            digits=3,
            zero_division=0,
        ))

    print()


def evaluate_human_gold(submission_dir: Path, verbose: bool = False,
                        types_csvs: list = HUMAN_TYPES_CSVS,
                        tags_csv: Path = HUMAN_TAGS_CSV,
                        rels_csv: Path = HUMAN_RELS_CSV):
    """
    Evaluate a submission against the expert human gold standard subsets.

    Each task uses its own CSV gold file(s); only paragraphs / pairs present in the
    gold are evaluated. Submission paragraphs not in the gold are ignored.

    Types are evaluated against both expert (11 files) and annotator2 (7 files),
    which cover 18 distinct files with no overlap (~715 paragraphs total).
    """
    sub_files = {p.name: p for p in submission_dir.glob("*.json")}

    def load_sub_paras(text_id: str) -> dict[int, dict]:
        fname = text_id + ".json"
        if fname not in sub_files:
            return {}
        with open(sub_files[fname], encoding="utf-8") as f:
            doc = json.load(f)
        return {p["para_number"]: p for p in doc["body"]["paragraphs"]}

    # ------------------------------------------------------------------ #
    #  Type classification
    # ------------------------------------------------------------------ #
    print("=" * 50)
    print("PARAGRAPH TYPE CLASSIFICATION (human gold)")
    print("=" * 50)

    gt_types, sub_types = [], []
    skipped_types = 0
    for types_csv in types_csvs:
        with open(types_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gold_type = (row.get("type") or row.get("type_manual") or "").strip()
                if not gold_type:
                    continue
                text_id = row["text_id"].strip()
                para_num = int(row["para_number"].strip())
                paras = load_sub_paras(text_id)
                if para_num not in paras:
                    skipped_types += 1
                    continue
                gt_types.append(gold_type)
                sub_types.append(paras[para_num].get("type") or "none")

    if skipped_types:
        print(f"  Warning: {skipped_types} gold paragraph(s) missing from submission")
    if gt_types:
        print(classification_report(
            gt_types, sub_types,
            labels=["preambular", "operative"],
            digits=3,
            zero_division=0,
        ))
    else:
        print("  No matching paragraphs found.")

    # ------------------------------------------------------------------ #
    #  Tag prediction (multi-label)
    # ------------------------------------------------------------------ #
    print("=" * 50)
    print("TAG PREDICTION (human gold)")
    print("=" * 50)

    gt_tags_list, sub_tags_list = [], []
    skipped_tags = 0
    with open(tags_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            agreed_raw = row["agreed"].strip()
            gold_tags = set(t.strip() for t in agreed_raw.splitlines() if t.strip())
            text_id = row["file"].strip().replace(".json", "")
            para_num = int(row["para_number"].strip())
            paras = load_sub_paras(text_id)
            if para_num not in paras:
                skipped_tags += 1
                continue
            gt_tags_list.append(gold_tags)
            sub_tags_list.append(set(paras[para_num].get("tags") or []))

    if skipped_tags:
        print(f"  Warning: {skipped_tags} gold paragraph(s) missing from submission")
    if gt_tags_list:
        mlb = MultiLabelBinarizer()
        mlb.fit(gt_tags_list + sub_tags_list)
        gt_bin = mlb.transform(gt_tags_list)
        sub_bin = mlb.transform(sub_tags_list)
        micro = f1_score(gt_bin, sub_bin, average="micro", zero_division=0)
        macro = f1_score(gt_bin, sub_bin, average="macro", zero_division=0)
        print(f"  Micro F1: {micro:.3f}")
        print(f"  Macro F1: {macro:.3f}")
        if verbose:
            per_tag = f1_score(gt_bin, sub_bin, average=None, zero_division=0)
            rows = [(tag, float(score), int(gt_bin[:, i].sum()))
                    for i, (tag, score) in enumerate(zip(mlb.classes_, per_tag))]  # type: ignore[arg-type]
            print()
            print("  Per-tag breakdown:")
            for tag, score, support in sorted(rows, key=lambda x: -x[1]):
                print(f"    {tag:<30}  F1={score:.3f}  (support={support})")
    else:
        print("  No matching paragraphs found.")

    # ------------------------------------------------------------------ #
    #  Related paragraph pair identification + relation labels
    # ------------------------------------------------------------------ #
    print("=" * 50)
    print("RELATED PARAGRAPH PAIR IDENTIFICATION (human gold)")
    print("=" * 50)

    gt_pair_sets: list[set] = []
    sub_pair_sets: list[set] = []
    gt_rel_labels: list[str] = []
    sub_rel_labels: list[str] = []

    # Group gold rows by text_id
    from collections import defaultdict
    gold_by_doc: dict[str, list[dict]] = defaultdict(list)
    with open(rels_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gold_by_doc[row["text_id"].strip()].append(row)

    for text_id, gold_rows in sorted(gold_by_doc.items()):
        paras = load_sub_paras(text_id)
        gt_pairs: set[tuple] = set()
        sub_pairs: set[tuple] = set()

        for row in gold_rows:
            gold_label = row["final_label"].strip()
            if gold_label in ("", "no label"):
                continue
            px = int(row["Para X"].strip())  # later para
            py = int(row["Para Y"].strip())  # earlier para
            pair = (max(px, py), min(px, py))
            gt_pairs.add(pair)

            # Look up submission label for this pair
            if px in paras:
                sub_mp = paras[px].get("matched_paras") or paras[px].get("matched_pars") or {}
                sub_mp_str = {str(k): v for k, v in sub_mp.items()}
                if str(py) in sub_mp_str:
                    sub_pairs.add(pair)
                    gt_rel_labels.append(gold_label)
                    sub_label = sub_mp_str[str(py)]
                    sub_rel_labels.append(sub_label[0] if isinstance(sub_label, list) else sub_label)
            if py in paras:
                sub_mp = paras[py].get("matched_paras") or paras[py].get("matched_pars") or {}
                sub_mp_str = {str(k): v for k, v in sub_mp.items()}
                if str(px) in sub_mp_str and pair not in sub_pairs:
                    sub_pairs.add(pair)
                    gt_rel_labels.append(gold_label)
                    sub_label = sub_mp_str[str(px)]
                    sub_rel_labels.append(sub_label[0] if isinstance(sub_label, list) else sub_label)

        # Count all submission pairs for this doc (for FP calculation)
        all_sub_pairs: set[tuple] = set()
        for para_num, para in paras.items():
            sub_mp = para.get("matched_paras") or para.get("matched_pars") or {}
            for tgt in sub_mp:
                all_sub_pairs.add((max(para_num, int(tgt)), min(para_num, int(tgt))))

        gt_pair_sets.append(gt_pairs)
        sub_pair_sets.append(all_sub_pairs)

    tp = fp = fn = 0
    for gt_pairs, sub_pairs in zip(gt_pair_sets, sub_pair_sets):
        tp += len(gt_pairs & sub_pairs)
        fp += len(sub_pairs - gt_pairs)
        fn += len(gt_pairs - sub_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)

    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  (TP={tp}, FP={fp}, FN={fn})")
    print(f"  Note: FP counts all submission pairs in these documents not in human gold,")
    print(f"  including pairs not covered by the gold sample (low precision expected).")

    print("=" * 50)
    print("RELATION LABEL CLASSIFICATION (human gold)")
    print("=" * 50)

    if not gt_rel_labels:
        print("  No overlapping pairs to evaluate labels on.")
    else:
        rel_types = ["complemental", "contradictive", "modifying", "supporting"]
        print(classification_report(
            gt_rel_labels, sub_rel_labels,
            labels=rel_types,
            digits=3,
            zero_division=0,
        ))

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True,
                        help="Path to participant submission directory")
    parser.add_argument("--gt-dir", type=Path, default=FULL_DIR,
                        help=f"Ground truth directory for silver eval (default: {FULL_DIR})")
    parser.add_argument("--human-gold", action="store_true",
                        help="Evaluate against expert human gold CSVs instead of silver GT")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-tag breakdown and paragraph-level warnings")
    args = parser.parse_args()

    if args.human_gold:
        evaluate_human_gold(Path(args.submission), verbose=args.verbose)
    else:
        evaluate(Path(args.submission), verbose=args.verbose, gt_dir=args.gt_dir)
