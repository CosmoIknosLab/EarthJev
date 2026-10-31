"""Offline metrics on dumped slot logits (s1geo.onejev_eval output).

    python -m s1geo.metrics --val runs/eval/x/val.logits.jsonl --test runs/eval/x/test.logits.jsonl --out m.json

Per task and overall: accuracy, macro-F1, NLL, Brier, top-label ECE (15 equal-width bins, as
qev.calibrate.ece). Temperatures are fitted per (kind, K) on --val with qev.calibrate.fit_temperature
(OneJev's calibration) and applied to --test; both uncalibrated (T=1) and calibrated numbers are reported.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from qev.calibrate import brier, ece, fit_temperature, nll, softmax


def load(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def macro_f1(golds: list[int], preds: list[int], k: int) -> float:
    f1s = []
    for c in range(k):
        tp = sum(1 for g, p in zip(golds, preds) if g == c and p == c)
        fp = sum(1 for g, p in zip(golds, preds) if g != c and p == c)
        fn = sum(1 for g, p in zip(golds, preds) if g == c and p != c)
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s) if f1s else float("nan")


def score_rows(rows: list[dict], temps: dict[str, float] | None = None) -> dict:
    out: dict = {}
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["task"]].append(r)
        groups["all"].append(r)
    for task, rs in groups.items():
        accs, nlls, briers, confs, golds, preds = [], [], [], [], [], []
        for r in rs:
            t = (temps or {}).get(f"{r['kind']}:{len(r['labels'])}", 1.0)
            p = softmax(r["logits"], t)
            gold = max(range(len(r["target"])), key=r["target"].__getitem__)
            pred = max(range(len(p)), key=p.__getitem__)
            accs.append(float(pred == gold))
            nlls.append(nll(p, r["target"]))
            briers.append(brier(p, r["target"]))
            confs.append(max(p))
            # F1 over the label names, not slot positions
            golds.append(r["labels"][gold])
            preds.append(r["labels"][pred])
        names = sorted(set(golds) | set(preds))
        idx = {n: i for i, n in enumerate(names)}
        n = len(rs)
        out[task] = {
            "n": n,
            "acc": sum(accs) / n,
            "macro_f1": macro_f1([idx[g] for g in golds], [idx[p] for p in preds], len(names)) if task != "all" else None,
            "nll": sum(nlls) / n,
            "brier": sum(briers) / n,
            "ece15": ece(confs, accs, n_bins=15),
            "mean_conf": sum(confs) / n,
        }
    return out


def fit_temperatures(rows: list[dict]) -> dict[str, float]:
    by = collections.defaultdict(lambda: ([], []))
    for r in rows:
        logits, targets = by[f"{r['kind']}:{len(r['labels'])}"]
        logits.append(r["logits"])
        targets.append(r["target"])
    return {k: fit_temperature(lg, tg) for k, (lg, tg) in by.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--val", default=None, help="fit temperatures here; omit to report T=1 only")
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    test = load(args.test)
    val = load(args.val) if args.val else None
    temps = fit_temperatures(val) if val else {}
    res = {"temperatures": temps, "test_T1": score_rows(test), "test_calibrated": score_rows(test, temps)}
    if val:
        res["val_T1"] = score_rows(val)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    print("temperatures:", {k: round(v, 3) for k, v in temps.items()})
    print(f"{'task':22s} {'n':>6s} {'acc':>7s} {'mF1':>7s} {'NLL':>7s} {'Brier':>7s} {'ECE':>7s} | {'NLL_T':>7s} {'ECE_T':>7s}")
    for task in sorted(res["test_T1"], key=lambda t: (t != "all", t)):
        a, c = res["test_T1"][task], res["test_calibrated"][task]
        f1 = f"{a['macro_f1']:.4f}" if a["macro_f1"] is not None else "   -   "
        print(f"{task:22s} {a['n']:6d} {a['acc']:.4f} {f1} {a['nll']:.4f} {a['brier']:.4f} {a['ece15']:.4f} | "
              f"{c['nll']:.4f} {c['ece15']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
