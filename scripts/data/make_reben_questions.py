"""Typed questions (OneJev unified JSONL) from reBEN CORINE masks: dominant class, presence, share,
location and comparison, with English and Italian templates.

Answers come from the CORINE level-3 reference mask of each patch. Ambiguous cases are skipped rather
than labelled (thresholds below). Templates are split into "seen" (used in train/validation) and
"unseen" (held out): the test set is written twice with the same question instances, once per template
set, to measure generalisation to new phrasings.

    python scripts/data/make_reben_questions.py --subset data/reben/subset_5pct --out data/earthjev/reben5pct
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from s1geo.clc import LEVEL_CODES, NAMES, fractions, to_level

PRESENT_MIN = 0.05          # presence: >= 5% of valid pixels; absence: exactly 0
SHARE_BINS = [0.10, 0.30, 0.60]
SHARE_MARGIN = 0.02         # skip shares within +-2 points of a bin edge (and in (0, 2%))
DOMINANT_MARGIN = 0.05      # skip dominant-class questions when top-2 differ by less
LOCATION_MIN, LOCATION_PEAK = 0.10, 0.5
COMPARE_GAP, COMPARE_MIN = 0.15, 0.10

# Italian names: our translation of the CLC level-1/2 headings, to be checked against ISPRA's nomenclature.
NAMES_IT = {
    1: "Superfici artificiali", 2: "Superfici agricole", 3: "Territori boscati e ambienti seminaturali",
    4: "Zone umide", 5: "Corpi idrici",
    11: "Zone urbanizzate", 12: "Zone industriali, commerciali e reti di comunicazione",
    13: "Zone estrattive, discariche e cantieri", 14: "Zone verdi artificiali non agricole", 21: "Seminativi",
    22: "Colture permanenti", 23: "Prati stabili e pascoli", 24: "Zone agricole eterogenee", 31: "Zone boscate",
    32: "Zone con vegetazione arbustiva e/o erbacea", 33: "Zone aperte con vegetazione rada o assente",
    41: "Zone umide interne", 42: "Zone umide marittime", 51: "Acque continentali", 52: "Acque marittime",
}
SHARE_KEYS = ["none", "under_10", "10_30", "30_60", "over_60"]
SHARE_DESC = {"en": ["None (0%).", "Less than 10%.", "Between 10% and 30%.", "Between 30% and 60%.", "More than 60%."],
              "it": ["Nessuna (0%).", "Meno del 10%.", "Tra il 10% e il 30%.", "Tra il 30% e il 60%.", "Più del 60%."]}
QUADRANTS = ["north_west", "north_east", "south_west", "south_east"]
QUAD_DESC = {"en": ["North-west (top left).", "North-east (top right).", "South-west (bottom left).", "South-east (bottom right)."],
             "it": ["Nord-ovest (in alto a sinistra).", "Nord-est (in alto a destra).", "Sud-ovest (in basso a sinistra).",
                    "Sud-est (in basso a destra)."]}

# (seen templates, unseen templates) per language and question type. {c}, {a}, {b} = class names.
T = {
    "dominant": {"en": (["Which land-cover class covers most of this image?",
                         "What is the dominant land cover in this Sentinel-2 patch?",
                         "Which CORINE land-cover class occupies the largest area here?"],
                        ["Looking at this satellite image, which kind of land covers the biggest part of it?"]),
                 "it": (["Quale classe di copertura del suolo occupa la maggior parte di questa immagine?",
                         "Qual è la copertura del suolo prevalente in questa patch Sentinel-2?"],
                        ["Guardando questa immagine satellitare, che tipo di territorio ne occupa la parte più grande?"])},
    "presence": {"en": (["This image contains {c}.", "Is there any {c} in this patch?", "Does {c} appear in this satellite image?"],
                        ["Can you see {c} anywhere in this picture?"]),
                 "it": (["In questa immagine sono presenti {c}.", "Ci sono {c} in questa patch?"],
                        ["Si vedono {c} da qualche parte in questa immagine?"])},
    "share": {"en": (["What share of the image is covered by {c}?", "How much of this patch is {c}?",
                      "Which fraction of the area is occupied by {c}?"],
                     ["Roughly what percentage of this scene is {c}?"]),
              "it": (["Quale quota dell'immagine è coperta da {c}?", "Quanta parte di questa patch è costituita da {c}?"],
                     ["Più o meno, che percentuale di questa scena è occupata da {c}?"])},
    "location": {"en": (["In which part of the image is most of the {c}?", "Where in this patch is the {c} mainly located?",
                         "Which quadrant contains the largest part of the {c}?"],
                        ["If you split the image into four, where would you find most of the {c}?"]),
                 "it": (["In quale parte dell'immagine si trova la maggior parte della classe «{c}»?", "Dove si trova principalmente la classe «{c}» in questa patch?"],
                        ["Dividendo l'immagine in quattro parti, dove si trova soprattutto la classe «{c}»?"])},
    "compare": {"en": (["Which covers more of the image: {a} or {b}?", "Is there more {a} or more {b} in this patch?",
                        "Which class has the larger area here, {a} or {b}?"],
                       ["Comparing {a} and {b}, which one takes up more space in this scene?"]),
                "it": (["Quale classe copre una parte maggiore dell'immagine: «{a}» o «{b}»?", "In questa patch occupa più area la classe «{a}» o la classe «{b}»?"],
                       ["Confrontando le classi «{a}» e «{b}», quale occupa più spazio in questa scena?"])},
}


def name(code: int, lang: str) -> str:
    return (NAMES[code] if lang == "en" else NAMES_IT[code]).lower()


def key(code: int) -> str:
    return NAMES[code].lower().replace(",", "").replace("/", " or ").replace(" ", "_")


def class_choice(level: int, lang: str) -> dict[str, str]:
    return {key(c): (NAMES[c] if lang == "en" else NAMES_IT[c]) + "." for c in LEVEL_CODES[level]}


def share_bin(f: float) -> int | None:
    if f == 0:
        return 0
    if f < SHARE_MARGIN or any(abs(f - e) < SHARE_MARGIN for e in SHARE_BINS):
        return None
    return 1 + sum(f > e for e in SHARE_BINS)


def quadrant_shares(lv: np.ndarray, code: int) -> np.ndarray:
    h, w = lv.shape
    m = lv == code
    q = np.array([m[:h // 2, :w // 2].sum(), m[:h // 2, w // 2:].sum(), m[h // 2:, :w // 2].sum(), m[h // 2:, w // 2:].sum()],
                 dtype=float)
    return q / max(q.sum(), 1)


def instances(clc: np.ndarray, rng: random.Random) -> list[dict]:
    """Question instances (language/template chosen later) for one mask."""
    out = []
    fr = {lvl: fractions(clc, lvl) for lvl in (1, 2)}
    for lvl in (1, 2):
        ranked = sorted(fr[lvl].items(), key=lambda kv: -kv[1])
        if ranked and (len(ranked) == 1 or ranked[0][1] - ranked[1][1] >= DOMINANT_MARGIN):
            out.append({"qtype": "dominant", "level": lvl, "answer": key(ranked[0][0]), "classes": [ranked[0][0]]})
    for want_present in (True, False):
        lvl = rng.choice((1, 2))
        pool = [c for c in LEVEL_CODES[lvl] if (fr[lvl].get(c, 0) >= PRESENT_MIN if want_present else fr[lvl].get(c, 0) == 0)]
        if pool:
            c = rng.choice(pool)
            out.append({"qtype": "presence", "level": lvl, "answer": "yes" if want_present else "no", "classes": [c],
                        "fraction": fr[lvl].get(c, 0.0)})
    pool = [(c, share_bin(fr[1].get(c, 0.0))) for c in LEVEL_CODES[1]]
    pool = [(c, b) for c, b in pool if b is not None]
    present_pool = [(c, b) for c, b in pool if b > 0]
    if present_pool and rng.random() < 0.5:  # half of the share questions ask about a class that is present
        pool = present_pool
    if pool:
        c, b = rng.choice(pool)
        out.append({"qtype": "share", "level": 1, "answer": SHARE_KEYS[b], "classes": [c], "fraction": fr[1].get(c, 0.0)})
    lvl = rng.choice((1, 2))
    lv = to_level(clc, lvl)
    pool = []
    for c, f in fr[lvl].items():
        if f >= LOCATION_MIN:
            q = quadrant_shares(lv, c)
            if q.max() >= LOCATION_PEAK:
                pool.append((c, int(q.argmax())))
    if pool:
        c, qi = rng.choice(pool)
        out.append({"qtype": "location", "level": lvl, "answer": QUADRANTS[qi], "classes": [c]})
    lvl = rng.choice((1, 2))
    present = sorted(fr[lvl])
    pairs = [(a, b) for i, a in enumerate(present) for b in present[i + 1:]
             if abs(fr[lvl][a] - fr[lvl][b]) >= COMPARE_GAP and max(fr[lvl][a], fr[lvl][b]) >= COMPARE_MIN]
    if pairs:
        a, b = rng.choice(pairs)
        if rng.random() < 0.5:
            a, b = b, a
        out.append({"qtype": "compare", "level": lvl, "answer": key(a if fr[lvl][a] > fr[lvl][b] else b), "classes": [a, b]})
    return out


def render(inst: dict, lang: str, template: str) -> tuple[dict, dict]:
    q, cls = inst["qtype"], inst["classes"]
    if q == "dominant":
        return {"type": "choice", "instructions": template, "criteria": class_choice(inst["level"], lang)}, {inst["answer"]: 1.0}
    if q == "presence":
        yes = inst["answer"] == "yes"
        return ({"type": "noul", "instructions": template.format(c=name(cls[0], lang))},
                {"yes": 1.0 if yes else 0.0, "no": 0.0 if yes else 1.0})
    if q == "share":
        return ({"type": "choice", "instructions": template.format(c=name(cls[0], lang)),
                 "criteria": dict(zip(SHARE_KEYS, SHARE_DESC[lang]))}, {inst["answer"]: 1.0})
    if q == "location":
        return ({"type": "choice", "instructions": template.format(c=name(cls[0], lang)),
                 "criteria": dict(zip(QUADRANTS, QUAD_DESC[lang]))}, {inst["answer"]: 1.0})
    a, b = cls
    return ({"type": "choice", "instructions": template.format(a=name(a, lang), b=name(b, lang)),
             "criteria": {key(a): (NAMES[a] if lang == "en" else NAMES_IT[a]) + ".",
                          key(b): (NAMES[b] if lang == "en" else NAMES_IT[b]) + "."}}, {inst["answer"]: 1.0})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--italian", type=float, default=0.3, help="share of questions rendered in Italian")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    stats = {}
    for split in ("train", "validation", "test"):
        clc = zarr.open_group(str(args.subset / f"{split}.zarr"), mode="r")["clc"]
        idx = pd.read_parquet(args.subset / f"{split}_index.parquet")
        rng = random.Random(f"{args.seed}-{split}")
        files = {"seen": open(args.out / f"{split}.jsonl", "w")}
        if split == "test":
            files["unseen"] = open(args.out / "test_unseen.jsonl", "w")
        n = {k: 0 for k in files}
        for i, row in idx.iterrows():
            pid = row["patch_id"]
            date = pid.split("_")[2][:8]
            state = {"sensor": "Sentinel-2 L2A, true-colour composite (bands B4, B3, B2)",
                     "ground_sample_distance": "10 m", "patch": "120 x 120 pixels (1.2 km x 1.2 km), north up",
                     "acquisition_date": f"{date[:4]}-{date[4:6]}-{date[6:]}", "country": row["country"],
                     "image": "<image:1>"}
            media = [{"type": "image", "path": f"rgb/{split}/{pid}.png", "w": 120, "h": 120}]
            for j, inst in enumerate(instances(np.asarray(clc[i]), rng)):
                lang = "it" if rng.random() < args.italian else "en"
                seen, unseen = T[inst["qtype"]][lang]
                choices = {"seen": seen, "unseen": unseen}
                tpl_pick = rng.random()
                for kind, fh in files.items():
                    tpls = choices[kind]
                    t_i = int(tpl_pick * len(tpls))
                    question, target = render(inst, lang, tpls[t_i])
                    meta = {"item_id": pid, "qtype": inst["qtype"], "level": inst["level"], "lang": lang,
                            "template": f"{kind}{t_i}", "classes": inst["classes"], "fraction": inst.get("fraction"),
                            "judge": "clc_reference_map"}
                    fh.write(json.dumps({"id": f"reben/{pid}/{j}", "source": "reben", "task": inst["qtype"], "split": split,
                                         "license": "cdla-permissive-1.0", "state": state, "question": question,
                                         "target": target, "media": media, "meta": meta}, ensure_ascii=False) + "\n")
                    n[kind] += 1
        for fh in files.values():
            fh.close()
        stats[split] = n
        print(split, len(idx), "patches", n, flush=True)
    (args.out / "summary.json").write_text(json.dumps({"stats": stats, "thresholds": {
        "present_min": PRESENT_MIN, "share_bins": SHARE_BINS, "share_margin": SHARE_MARGIN,
        "dominant_margin": DOMINANT_MARGIN, "location_min": LOCATION_MIN, "location_peak": LOCATION_PEAK,
        "compare_gap": COMPARE_GAP, "compare_min": COMPARE_MIN}, "italian_share": args.italian, "seed": args.seed}, indent=1))


if __name__ == "__main__":
    main()
