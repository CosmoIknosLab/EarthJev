"""RSVQAxBEN test questions on our reBEN test patches, as OneJev rows for EarthJev.

Input: data/rsvqaxben/test_on_reben5pct_test.json (RSVQAxBEN test questions whose image is in our reBEN v2 test
subset, so EarthJev never saw these images). Output in <out>/:
  presence.jsonl  one noul row per RSVQAxBEN presence question (question text verbatim)
With --vocab (land-cover classes per CORINE level), also:
  classes.jsonl   one noul row per (image, class) for every class in the RSVQAxBEN vocabulary
                  ("Are there some {class} in the image?"), used to compose answers to LC questions
  lc.json         the LC questions with their level (1/2/3 or null for "in addition to X") and gold class set

    python scripts/data/make_rsvqaxben_eval.py --out data/earthjev/rsvqaxben_test
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from s1geo.rsvqa import parse_classes

REPO = Path(__file__).resolve().parents[2]


def state_for(pid: str, country: str) -> dict:
    date = pid.split("_")[2][:8]
    state = {"sensor": "Sentinel-2 L2A, true-colour composite (bands B4, B3, B2)", "ground_sample_distance": "10 m",
             "patch": "120 x 120 pixels (1.2 km x 1.2 km), north up",
             "acquisition_date": f"{date[:4]}-{date[4:6]}-{date[6:]}", "country": country}
    state["image"] = "<image:1>"
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=REPO / "data/rsvqaxben/test_on_reben5pct_test.json")
    ap.add_argument("--vocab", type=Path, default=None, help="classes per level, for classes.jsonl and lc.json")
    ap.add_argument("--subset", type=Path, default=REPO / "data/reben/subset_5pct")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    src = json.loads(args.src.read_text())
    country = dict(pd.read_parquet(args.subset / "test_index.parquet")[["patch_id", "country"]].values)
    img2pid = {int(k): v for k, v in src["img_to_v2"].items()}

    def row(rid, pid, question, target, task):
        return {"id": rid, "source": "rsvqaxben", "task": task, "split": "test", "license": "cc-by-4.0",
                "state": state_for(pid, country[pid]), "question": question, "target": target,
                "media": [{"type": "image", "path": f"rgb/test/{pid}.png", "w": 120, "h": 120}],
                "meta": {"item_id": pid, "judge": "rsvqaxben_label"}}

    n_pres = 0
    with open(args.out / "presence.jsonl", "w") as fh:
        for r in src["rows"]:
            if r["type"] != "presence":
                continue
            yes = r["answer"] == "yes"
            fh.write(json.dumps(row(f"rsvqaxben/{r['qid']}", img2pid[r["img_id"]],
                                    {"type": "noul", "instructions": r["question"]},
                                    {"yes": 1.0 if yes else 0.0, "no": 0.0 if yes else 1.0}, "presence"),
                                ensure_ascii=False) + "\n")
            n_pres += 1
    if args.vocab is None:
        print(f"presence {n_pres}")
        return
    vocab = {int(k): v for k, v in json.loads(args.vocab.read_text()).items()}
    all_classes = sorted(set().union(*vocab.values()))
    pids = sorted({img2pid[r["img_id"]] for r in src["rows"]})
    with open(args.out / "classes.jsonl", "w") as fh:
        for pid in pids:
            for c in all_classes:
                fh.write(json.dumps(row(f"class/{pid}/{c}", pid, {"type": "noul", "instructions": f"Are there some {c} in the image?"},
                                        {"yes": 0.5, "no": 0.5}, "class_presence"), ensure_ascii=False) + "\n")
    lc = []
    for r in src["rows"]:
        if r["type"] != "LC":
            continue
        m = re.search(r"\bL([123])\b", r["question"])
        x = re.match(r"(?:In addition to|Besides) (.+?), ", r["question"])
        gold, rest = parse_classes(r["answer"])
        assert not rest, r
        lc.append({"qid": r["qid"], "pid": img2pid[r["img_id"]], "question": r["question"],
                   "level": int(m.group(1)) if m else None, "given": x.group(1).lower() if x else None,
                   "gold": sorted(gold)})
    (args.out / "lc.json").write_text(json.dumps({"vocab": vocab, "questions": lc}, indent=0))
    print(f"presence {n_pres}, class rows {len(pids) * len(all_classes)} ({len(pids)} images x {len(all_classes)} classes), "
          f"LC {len(lc)} ({sum(q['level'] is not None for q in lc)} with explicit level)")


if __name__ == "__main__":
    main()
