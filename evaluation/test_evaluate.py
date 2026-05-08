"""
Tests for evaluate.py

Each test writes minimal JSON files into tmp_path subdirs and calls evaluate(),
capturing stdout to assert on the reported scores.
"""

import json
import re
import sys
from pathlib import Path

import pytest

from evaluate import evaluate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(paragraphs: list[dict]) -> dict:
    return {"body": {"paragraphs": paragraphs}}


def _para(
    num: int,
    type_: str,
    tags: list[str],
    matched_paras: dict | None = None,
) -> dict:
    return {
        "para_number": num,
        "para": f"Paragraph {num} text.",
        "type": type_,
        "tags": tags,
        "matched_paras": matched_paras or {},
    }


def _write(path: Path, doc: dict):
    path.write_text(json.dumps(doc), encoding="utf-8")


def _capture(tmp_path, gt_doc, sub_doc, verbose=False) -> str:
    """Write one GT and one submission file, run evaluate, return stdout."""
    gt_dir  = tmp_path / "gt"
    sub_dir = tmp_path / "sub"
    gt_dir.mkdir()
    sub_dir.mkdir()
    fname = "doc_001.json"
    _write(gt_dir / fname, gt_doc)
    _write(sub_dir / fname, sub_doc)

    captured = []
    orig_print = __builtins__["print"] if isinstance(__builtins__, dict) else print

    import builtins
    original = builtins.print
    builtins.print = lambda *a, **kw: captured.append(" ".join(str(x) for x in a))
    try:
        evaluate(sub_dir, verbose=verbose, gt_dir=gt_dir)
    finally:
        builtins.print = original

    return "\n".join(captured)


def _extract(output: str, label: str) -> float:
    """Pull a float value from a line like '  Label: 0.857'."""
    pattern = rf"{re.escape(label)}\s*:?\s*([0-9]\.[0-9]+)"
    m = re.search(pattern, output, re.IGNORECASE)
    assert m, f"Could not find '{label}' in output:\n{output}"
    return float(m.group(1))


# ---------------------------------------------------------------------------
# Type classification
# ---------------------------------------------------------------------------

class TestTypeClassification:

    def test_perfect_predictions(self, tmp_path):
        gt  = _make_doc([_para(1, "preambular", []), _para(2, "operative", [])])
        sub = _make_doc([_para(1, "preambular", []), _para(2, "operative", [])])
        out = _capture(tmp_path, gt, sub)
        # macro avg f1-score should be 1.000
        assert "1.000" in out

    def test_all_wrong_type(self, tmp_path):
        gt  = _make_doc([_para(1, "preambular", []), _para(2, "operative", [])])
        sub = _make_doc([_para(1, "operative", []),  _para(2, "preambular", [])])
        out = _capture(tmp_path, gt, sub)
        # macro avg precision/recall/f1 should all be 0.000
        assert "0.000" in out

    def test_partial_correct_type(self, tmp_path):
        gt  = _make_doc([
            _para(1, "preambular", []),
            _para(2, "preambular", []),
            _para(3, "operative",  []),
            _para(4, "operative",  []),
        ])
        sub = _make_doc([
            _para(1, "preambular", []),   # correct
            _para(2, "operative",  []),   # wrong
            _para(3, "operative",  []),   # correct
            _para(4, "preambular", []),   # wrong
        ])
        out = _capture(tmp_path, gt, sub)
        # preambular: 1 TP, 1 FP, 1 FN → precision=0.5, recall=0.5, f1=0.5
        assert "0.500" in out


# ---------------------------------------------------------------------------
# Tag prediction (multi-label)
# ---------------------------------------------------------------------------

class TestTagPrediction:

    def test_perfect_tags(self, tmp_path):
        gt  = _make_doc([_para(1, "preambular", ["A", "B"]), _para(2, "operative", ["C"])])
        sub = _make_doc([_para(1, "preambular", ["A", "B"]), _para(2, "operative", ["C"])])
        out = _capture(tmp_path, gt, sub)
        assert _extract(out, "Micro F1") == pytest.approx(1.0, abs=0.001)
        assert _extract(out, "Macro F1") == pytest.approx(1.0, abs=0.001)

    def test_no_tags_predicted(self, tmp_path):
        gt  = _make_doc([_para(1, "preambular", ["A", "B"])])
        sub = _make_doc([_para(1, "preambular", [])])
        out = _capture(tmp_path, gt, sub)
        assert _extract(out, "Micro F1") == pytest.approx(0.0, abs=0.001)

    def test_partial_tags(self, tmp_path):
        # GT has A+B, sub predicts only A → recall=0.5, precision=1.0, F1=0.667
        gt  = _make_doc([_para(1, "preambular", ["A", "B"])])
        sub = _make_doc([_para(1, "preambular", ["A"])])
        out = _capture(tmp_path, gt, sub)
        assert _extract(out, "Micro F1") == pytest.approx(0.667, abs=0.001)

    def test_extra_tags_predicted(self, tmp_path):
        # GT has A, sub predicts A+B → precision=0.5, recall=1.0, F1=0.667
        gt  = _make_doc([_para(1, "preambular", ["A"])])
        sub = _make_doc([_para(1, "preambular", ["A", "B"])])
        out = _capture(tmp_path, gt, sub)
        assert _extract(out, "Micro F1") == pytest.approx(0.667, abs=0.001)


# ---------------------------------------------------------------------------
# Matched para pair identification
# ---------------------------------------------------------------------------

class TestMatchedParaPairs:

    def test_perfect_pairs(self, tmp_path):
        # Para 3 matches paras 1 and 2
        gt  = _make_doc([
            _para(1, "preambular", []),
            _para(2, "preambular", []),
            _para(3, "operative",  [], matched_paras={"1": "supporting", "2": "supporting"}),
        ])
        sub = _make_doc([
            _para(1, "preambular", []),
            _para(2, "preambular", []),
            _para(3, "operative",  [], matched_paras={"1": "supporting", "2": "supporting"}),
        ])
        out = _capture(tmp_path, gt, sub)
        assert _extract(out, "Precision") == pytest.approx(1.0, abs=0.001)
        assert _extract(out, "Recall")    == pytest.approx(1.0, abs=0.001)
        assert _extract(out, "F1")        == pytest.approx(1.0, abs=0.001)

    def test_no_pairs_predicted(self, tmp_path):
        gt  = _make_doc([
            _para(1, "preambular", []),
            _para(2, "operative",  [], matched_paras={"1": "modifying"}),
        ])
        sub = _make_doc([
            _para(1, "preambular", []),
            _para(2, "operative",  [], matched_paras={}),   # missed the pair
        ])
        out = _capture(tmp_path, gt, sub)
        assert _extract(out, "Recall") == pytest.approx(0.0, abs=0.001)
        assert _extract(out, "F1")     == pytest.approx(0.0, abs=0.001)

    def test_extra_pairs_predicted(self, tmp_path):
        # GT has no pairs, submission invents one → precision=0, F1=0
        gt  = _make_doc([
            _para(1, "preambular", []),
            _para(2, "operative",  [], matched_paras={}),
        ])
        sub = _make_doc([
            _para(1, "preambular", []),
            _para(2, "operative",  [], matched_paras={"1": "supporting"}),
        ])
        out = _capture(tmp_path, gt, sub)
        assert _extract(out, "Precision") == pytest.approx(0.0, abs=0.001)
        assert _extract(out, "F1")        == pytest.approx(0.0, abs=0.001)

    def test_partial_pairs(self, tmp_path):
        # GT: para 4 → {1, 2, 3}.  Sub: para 4 → {1, 2}.  Miss para 3.
        # TP=2, FP=0, FN=1 → precision=1.0, recall=0.667, F1=0.800
        gt  = _make_doc([
            _para(1, "preambular", []),
            _para(2, "preambular", []),
            _para(3, "preambular", []),
            _para(4, "operative",  [], matched_paras={"1": "m", "2": "m", "3": "m"}),
        ])
        sub = _make_doc([
            _para(1, "preambular", []),
            _para(2, "preambular", []),
            _para(3, "preambular", []),
            _para(4, "operative",  [], matched_paras={"1": "m", "2": "m"}),
        ])
        out = _capture(tmp_path, gt, sub)
        assert _extract(out, "Precision") == pytest.approx(1.0,   abs=0.001)
        assert _extract(out, "Recall")    == pytest.approx(0.667, abs=0.001)
        assert _extract(out, "F1")        == pytest.approx(0.800, abs=0.001)

    def test_pairs_ignore_relation_type(self, tmp_path):
        # Pair identification ignores relation label — only para numbers matter
        gt  = _make_doc([
            _para(1, "preambular", []),
            _para(2, "operative",  [], matched_paras={"1": "supporting"}),
        ])
        sub = _make_doc([
            _para(1, "preambular", []),
            _para(2, "operative",  [], matched_paras={"1": "completely_wrong_label"}),
        ])
        out = _capture(tmp_path, gt, sub)
        assert _extract(out, "F1") == pytest.approx(1.0, abs=0.001)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_missing_para_in_submission(self, tmp_path):
        # Submission is missing para 2 — should not crash
        gt  = _make_doc([_para(1, "preambular", []), _para(2, "operative", [])])
        sub = _make_doc([_para(1, "preambular", [])])
        out = _capture(tmp_path, gt, sub, verbose=True)
        assert "missing" in out.lower() or "1" in out   # just check no exception

    def test_no_matched_paras_field(self, tmp_path):
        # Para with no matched_paras key at all — should count as empty
        para = {"para_number": 1, "para": "text", "type": "preambular", "tags": []}
        gt  = {"body": {"paragraphs": [para]}}
        sub = {"body": {"paragraphs": [para]}}
        out = _capture(tmp_path, gt, sub)
        assert _extract(out, "F1") == pytest.approx(0.0, abs=0.001)  # no pairs → 0/0 → 0
