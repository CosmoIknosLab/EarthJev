"""Paired image-level bootstrap, equivalence / superiority tests and calibration, as pre-specified in the analysis plan.

The spec (YAML) lists the test sets, the models (one prediction file per seed and test set) and the hypotheses:

    reference: earthjev
    test_sets: {corine: data/earthjev/<..>/test.jsonl, rsvqaxben: data/earthjev/<..>/presence.jsonl}
    models:
      earthjev: {name: EarthJev-MSI, corine: [runs/eval/<s0>/probe.logits.jsonl, ...], rsvqaxben: [...]}
      gensft:   {name: ..., corine: [runs/eval/<s0>_test, ...]}                  # greedy: dir with predictions.jsonl
      gensft_sum: {name: ..., field: logits, corine: [runs/eval/<s0>_test/option_scores.jsonl, ...]}
    hypotheses:
      H1: {type: equivalence, margin: 1.0, test_set: corine, compare: [qwen3vl, internvl3, earthdial, gensft]}
      H2: {type: superiority, test_set: corine, compare: [resnet50, swin_t, mambavision_t]}

Prediction files: `probe.logits.jsonl` / `option_scores.jsonl` (`logits`, or `field: logits_norm`; softmax over the
options gives the probabilities, so NLL and Brier exist), or a directory / file `predictions.jsonl` (`correct`,
`answer_prob` = probability of the generated sequence, null for the rules baseline; accuracy and ECE only).

Statistics, as pre-specified:
- clusters = test images (meta.item_id of the questions file): every replicate resamples images with replacement and
  keeps all their questions; 10,000 replicates; the same resampled images for every model (paired);
- seed-averaged predictions: per question, mean over seeds of the correctness indicator and of the top-answer
  probability (ECE 15 equal-width bins on the averaged probability; averaging acts as a small ensemble, so this ECE
  is lower than the mean of the per-seed ECEs, which is reported next to it);
- equivalence (TOST): difference reference - other inside [-margin, +margin] points; p = max of the two one-sided
  bootstrap p-values; 90% percentile interval reported. Superiority: one-sided, H0 difference <= 0;
- Holm within each hypothesis family; claim rule: rejected on the seed-averaged predictions AND for at least two of
  the three seeds (seed i of the reference against seed i of the other model; a one-seed model is used for all).
p-values are (count + 1) / (B + 1).

    PYTHONPATH=third_party/OneJev python scripts/analysis/paired_bootstrap.py --spec configs/analysis/msi.yaml \
        --out runs/analysis/msi/statistics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
N_BINS = 15


def questions(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    """(question ids, image index per question, task per question) in file order."""
    ids, items, tasks = [], [], []
    for line in open(path):
        r = json.loads(line)
        ids.append(r["id"]); items.append(r["meta"]["item_id"]); tasks.append(r.get("task"))
    _, img = np.unique(np.array(items), return_inverse=True)
    return ids, img, tasks


def load_seed(path: Path, ids: list[str], field: str) -> dict[str, np.ndarray]:
    """correct, conf (top-answer probability), nll, brier per question in `ids` order (nan where undefined)."""
    if path.is_dir():
        path = path / "predictions.jsonl"
    pos = {q: i for i, q in enumerate(ids)}
    n = len(ids)
    out = {k: np.full(n, np.nan) for k in ("correct", "conf", "nll", "brier")}
    seen = np.zeros(n, bool)
    for line in open(path):
        r = json.loads(line)
        i = pos.get(r["id"])
        if i is None:
            continue
        seen[i] = True
        if "correct" in r:
            out["correct"][i] = float(r["correct"])
            if r.get("answer_prob") is not None:
                out["conf"][i] = r["answer_prob"]
            continue
        z = np.asarray(r[field], dtype=np.float64)
        p = np.exp(z - z.max()); p /= p.sum()
        t = np.asarray(r["target"], dtype=np.float64)
        gold, top = int(t.argmax()), int(p.argmax())
        out["correct"][i] = float(top == gold)
        out["conf"][i] = p[top]
        out["nll"][i] = -np.log(max(p[gold], 1e-12))
        out["brier"][i] = float(((p - np.eye(len(p))[gold]) ** 2).sum())
    if not seen.all():
        raise ValueError(f"{path}: {int((~seen).sum())} of {n} questions missing")
    return out


class Clustered:
    """Per-image sufficient statistics of one prediction set, so a replicate is a weighted sum over images."""

    def __init__(self, pred: dict[str, np.ndarray], img: np.ndarray, n_img: int):
        c, p = pred["correct"], pred["conf"]
        self.n = np.bincount(img, minlength=n_img).astype(np.float64)
        self.correct = np.bincount(img, weights=c, minlength=n_img)
        self.has_conf = not np.isnan(p).any()
        if self.has_conf:
            b = np.minimum((p * N_BINS).astype(int), N_BINS - 1)
            self.bin_n = np.zeros((n_img, N_BINS)); self.bin_conf = np.zeros((n_img, N_BINS))
            self.bin_corr = np.zeros((n_img, N_BINS))
            np.add.at(self.bin_n, (img, b), 1.0); np.add.at(self.bin_conf, (img, b), p)
            np.add.at(self.bin_corr, (img, b), c)

    def stats(self, w: np.ndarray) -> dict[str, np.ndarray]:
        """w: [B, n_img] image counts -> accuracy and ECE per replicate."""
        tot = w @ self.n
        res = {"acc": (w @ self.correct) / tot}
        if self.has_conf:
            res["ece"] = np.abs(w @ self.bin_conf - w @ self.bin_corr).sum(1) / tot
        return res


def point(pred: dict[str, np.ndarray]) -> dict[str, float]:
    c, p = pred["correct"], pred["conf"]
    res = {"acc": float(c.mean())}
    if not np.isnan(p).any():
        b = np.minimum((p * N_BINS).astype(int), N_BINS - 1)
        res["ece"] = float(sum(abs(p[b == k].sum() - c[b == k].sum()) for k in range(N_BINS)) / len(c))
    for k in ("nll", "brier"):
        if not np.isnan(pred[k]).any():
            res[k] = float(pred[k].mean())
    return res


def holm(p: dict[str, float]) -> dict[str, float]:
    order = sorted(p, key=p.get)
    adj, run = {}, 0.0
    for j, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - j) * p[k]))
        adj[k] = run
    return adj


def pct(x: np.ndarray, lo: float, hi: float) -> list[float]:
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output prefix: <out>.json and <out>.md")
    ap.add_argument("--replicates", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    spec = yaml.safe_load(open(args.spec))
    ref, models, B = spec["reference"], spec["models"], args.replicates
    result = {"spec": str(args.spec), "replicates": B, "seed": args.seed, "test_sets": {}, "hypotheses": {}}
    boot = {}   # (test set, model, variant) -> {"acc": [B], "ece": [B]}

    for ts, qpath in spec["test_sets"].items():
        ids, img, tasks = questions(REPO / qpath)
        n_img = int(img.max()) + 1
        tasks = np.array(tasks)
        per_model = {}
        for key, m in models.items():
            if ts not in m:
                continue
            seeds = [load_seed(REPO / p, ids, m.get("field", "logits")) for p in m[ts]]
            avg = {k: np.mean([s[k] for s in seeds], 0) for k in seeds[0]}
            variants = {"mean": avg} | {f"s{i}": s for i, s in enumerate(seeds)}
            per_model[key] = {"variants": variants, "clustered": {v: Clustered(d, img, n_img) for v, d in variants.items()}}
            seed_acc = [float(s["correct"].mean()) for s in seeds]
            entry = {"name": m.get("name", key), "seeds": len(seeds), "seed_acc": seed_acc,
                     "seed_std": float(np.std(seed_acc, ddof=1)) if len(seeds) > 1 else None,
                     "seed_avg": point(avg),
                     "seed_ece": [point(s).get("ece") for s in seeds],
                     "by_task": {t: float(avg["correct"][tasks == t].mean()) for t in sorted(set(tasks))}}
            per_model[key]["entry"] = entry
        rng = np.random.default_rng(args.seed)
        acc = {k: {v: [] for v in d["variants"]} for k, d in per_model.items()}
        ece = {k: {v: [] for v in d["variants"]} for k, d in per_model.items()}
        chunk = 500
        for start in range(0, B, chunk):
            b = min(chunk, B - start)
            draw = rng.integers(0, n_img, size=(b, n_img)) + n_img * np.arange(b)[:, None]
            w = np.bincount(draw.ravel(), minlength=b * n_img).reshape(b, n_img).astype(np.float64)
            for k, d in per_model.items():
                for v, cl in d["clustered"].items():
                    s = cl.stats(w)
                    acc[k][v].append(s["acc"])
                    if "ece" in s:
                        ece[k][v].append(s["ece"])
        for k in per_model:
            for v in acc[k]:
                boot[(ts, k, v)] = {"acc": np.concatenate(acc[k][v])}
                if ece[k][v]:
                    boot[(ts, k, v)]["ece"] = np.concatenate(ece[k][v])
            e = per_model[k]["entry"]
            e["acc_ci95"] = pct(boot[(ts, k, "mean")]["acc"], 2.5, 97.5)
            if "ece" in boot[(ts, k, "mean")]:
                e["ece_ci95"] = pct(boot[(ts, k, "mean")]["ece"], 2.5, 97.5)
        diffs = {}
        if ref in per_model:
            for k in per_model:
                if k == ref:
                    continue
                d = {"acc_diff": float(per_model[ref]["entry"]["seed_avg"]["acc"] - per_model[k]["entry"]["seed_avg"]["acc"]),
                     "acc_diff_ci95": pct(boot[(ts, ref, "mean")]["acc"] - boot[(ts, k, "mean")]["acc"], 2.5, 97.5)}
                if "ece" in boot[(ts, ref, "mean")] and "ece" in boot[(ts, k, "mean")]:
                    d["ece_diff"] = per_model[ref]["entry"]["seed_avg"]["ece"] - per_model[k]["entry"]["seed_avg"]["ece"]
                    d["ece_diff_ci95"] = pct(boot[(ts, ref, "mean")]["ece"] - boot[(ts, k, "mean")]["ece"], 2.5, 97.5)
                diffs[k] = d
        result["test_sets"][ts] = {"questions": qpath, "n_questions": len(ids), "n_images": n_img,
                                   "models": {k: d["entry"] for k, d in per_model.items()},
                                   f"differences_vs_{ref}": diffs}

    for hname, h in (spec.get("hypotheses") or {}).items():
        ts, kind = h["test_set"], h["type"]
        margin = h.get("margin", 0.0) / 100
        n_ref = len(models[ref][ts])
        variants = ["mean"] + [f"s{i}" for i in range(n_ref)]
        tests = {}
        for v in variants:
            pvals, detail = {}, {}
            for k in h["compare"]:
                n_k = len(models[k][ts])
                vk = v if v == "mean" or n_k > 1 else "s0"
                d = boot[(ts, ref, v)]["acc"] - boot[(ts, k, vk)]["acc"]
                if kind == "equivalence":
                    p_lo = ((d <= -margin).sum() + 1) / (B + 1)
                    p_hi = ((d >= margin).sum() + 1) / (B + 1)
                    pvals[k] = float(max(p_lo, p_hi))
                    detail[k] = {"diff_ci90": pct(d, 5, 95)}
                elif kind == "superiority":
                    pvals[k] = float(((d <= 0).sum() + 1) / (B + 1))
                    detail[k] = {"diff_ci95": pct(d, 2.5, 97.5)}
                else:
                    raise ValueError(kind)
            adj = holm(pvals)
            for k in h["compare"]:
                detail[k].update({"p": pvals[k], "p_holm": adj[k], "rejected": adj[k] < 0.05})
            tests[v] = detail
        verdict = {}
        for k in h["compare"]:
            n_seeds_ok = sum(tests[v][k]["rejected"] for v in variants if v != "mean")
            verdict[k] = {"seed_mean": tests["mean"][k]["rejected"], "seeds_rejected": n_seeds_ok,
                          "supported": tests["mean"][k]["rejected"] and n_seeds_ok >= 2}
        result["hypotheses"][hname] = {**h, "tests": tests, "verdict": verdict}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{args.out}.json").write_text(json.dumps(result, indent=1))
    Path(f"{args.out}.md").write_text(markdown(result, ref))
    print(markdown(result, ref))
    return 0


def markdown(res: dict, ref: str) -> str:
    f = lambda x: f"{100 * x:.1f}"
    ci = lambda c: f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"
    cid = lambda c: f"[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"
    out = [f"# Paired image-level bootstrap ({res['replicates']:,} replicates)", "",
           f"Spec: `{res['spec']}`. Seed-averaged predictions; 95% percentile intervals; differences = {ref} - model.", ""]
    for ts, d in res["test_sets"].items():
        out += [f"## {ts}: {d['n_questions']:,} questions, {d['n_images']:,} images", "",
                "| Model | Seeds | Acc. [95% CI] | Seed std | ECE [95% CI] | ECE, mean of seeds | NLL | Brier | Diff. acc. [95% CI] "
                "| Diff. ECE [95% CI] |", "|---|---|---|---|---|---|---|---|---|---|"]
        diffs = d[f"differences_vs_{ref}"]
        for k, e in d["models"].items():
            sa = e["seed_avg"]
            dd = diffs.get(k, {})
            out.append("| " + " | ".join([
                e["name"], str(e["seeds"]), f"{f(sa['acc'])} {ci(e['acc_ci95'])}",
                f(e["seed_std"]) if e["seed_std"] is not None else "--",
                f"{sa['ece']:.3f} [{e['ece_ci95'][0]:.3f}, {e['ece_ci95'][1]:.3f}]" if "ece" in sa else "--",
                f"{np.mean(e['seed_ece']):.3f}" if "ece" in sa else "--",
                f"{sa['nll']:.3f}" if "nll" in sa else "--", f"{sa['brier']:.3f}" if "brier" in sa else "--",
                f"{100 * dd['acc_diff']:+.1f} {cid(dd['acc_diff_ci95'])}" if dd else "--",
                f"{dd['ece_diff']:+.3f} [{dd['ece_diff_ci95'][0]:+.3f}, {dd['ece_diff_ci95'][1]:+.3f}]"
                if "ece_diff" in dd else "--"]) + " |")
        out.append("")
    for hname, h in res["hypotheses"].items():
        names = {k: res["test_sets"][h["test_set"]]["models"][k]["name"] for k in h["compare"]}
        crit = f"margin ±{h['margin']} points, 90% CI" if h["type"] == "equivalence" else "one-sided, 95% CI"
        out += [f"## {hname}: {h['type']} on {h['test_set']} ({crit}; Holm over {len(h['compare'])})", "",
                "| Model | Seed mean: CI, p Holm | Seeds rejected | Supported |", "|---|---|---|---|"]
        for k in h["compare"]:
            t = h["tests"]["mean"][k]
            c = t.get("diff_ci90", t.get("diff_ci95"))
            v = h["verdict"][k]
            out.append(f"| {names[k]} | {cid(c)}, {t['p_holm']:.4f} | {v['seeds_rejected']} of "
                       f"{len(h['tests']) - 1} | {'yes' if v['supported'] else 'no'} |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
