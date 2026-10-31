import math

from s1geo.metrics import fit_temperatures, macro_f1, score_rows


def _row(task, logits, gold, kind="choice"):
    labels = [f"c{i}" for i in range(len(logits))]
    target = [1.0 if i == gold else 0.0 for i in range(len(logits))]
    return {"task": task, "kind": kind, "labels": labels, "logits": logits, "target": target}


def test_macro_f1_perfect_and_partial():
    assert macro_f1([0, 1, 2], [0, 1, 2], 3) == 1.0
    # class 0: tp1 fp0 fn1 -> 2/3; class 1: tp1 fp1 fn0 -> 2/3
    assert math.isclose(macro_f1([0, 0, 1], [0, 1, 1], 2), 2 / 3)


def test_score_rows_accuracy_and_ece():
    rows = [_row("lc", [5.0, 0.0], 0), _row("lc", [5.0, 0.0], 1)]
    m = score_rows(rows)["lc"]
    assert m["acc"] == 0.5
    assert m["ece15"] > 0.4  # confident (~0.993) but 50% accurate


def test_temperature_softens_overconfident_logits():
    rows = [_row("lc", [4.0, 0.0], 0) for _ in range(6)] + [_row("lc", [4.0, 0.0], 1) for _ in range(4)]
    temps = fit_temperatures(rows)
    assert temps["choice:2"] > 1.0
    assert score_rows(rows, temps)["lc"]["nll"] < score_rows(rows)["lc"]["nll"]
