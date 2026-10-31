"""CORINE Land Cover nomenclature (3 levels: 5 / 15 / 44 classes).

Names from the official CLC nomenclature guidelines
(https://land.copernicus.eu/content/corine-land-cover-nomenclature-guidelines/html/, read 2026-10-22).
Level-3 codes are what the BigEarthNet v2.0 reference maps store; levels 1 and 2 are their prefixes.
"""
from __future__ import annotations

import numpy as np

NAMES: dict[int, str] = {
    1: "Artificial surfaces",
    11: "Urban fabric",
    111: "Continuous urban fabric",
    112: "Discontinuous urban fabric",
    12: "Industrial, commercial and transport units",
    121: "Industrial or commercial units",
    122: "Road and rail networks and associated land",
    123: "Port areas",
    124: "Airports",
    13: "Mine, dump and construction sites",
    131: "Mineral extraction sites",
    132: "Dump sites",
    133: "Construction sites",
    14: "Artificial, non-agricultural vegetated areas",
    141: "Green urban areas",
    142: "Sport and leisure facilities",
    2: "Agricultural areas",
    21: "Arable land",
    211: "Non-irrigated arable land",
    212: "Permanently irrigated land",
    213: "Rice fields",
    22: "Permanent crops",
    221: "Vineyards",
    222: "Fruit trees and berry plantations",
    223: "Olive groves",
    23: "Pastures",
    231: "Pastures",
    24: "Heterogeneous agricultural areas",
    241: "Annual crops associated with permanent crops",
    242: "Complex cultivation patterns",
    243: "Land principally occupied by agriculture, with significant areas of natural vegetation",
    244: "Agro-forestry areas",
    3: "Forest and seminatural areas",
    31: "Forest",
    311: "Broad-leaved forest",
    312: "Coniferous forest",
    313: "Mixed forest",
    32: "Shrub and/or herbaceous vegetation associations",
    321: "Natural grassland",
    322: "Moors and heathland",
    323: "Sclerophyllous vegetation",
    324: "Transitional woodland/shrub",
    33: "Open spaces with little or no vegetation",
    331: "Beaches, dunes, sands",
    332: "Bare rock",
    333: "Sparsely vegetated areas",
    334: "Burnt areas",
    335: "Glaciers and perpetual snow",
    4: "Wetlands",
    41: "Inland wetlands",
    411: "Inland marshes",
    412: "Peatbogs",
    42: "Coastal wetlands",
    421: "Salt marshes",
    422: "Salines",
    423: "Intertidal flats",
    5: "Water bodies",
    51: "Inland waters",
    511: "Water courses",
    512: "Water bodies",
    52: "Marine waters",
    521: "Coastal lagoons",
    522: "Estuaries",
    523: "Sea and ocean",
}

LEVEL_CODES: dict[int, list[int]] = {
    1: sorted(c for c in NAMES if c < 10),
    2: sorted(c for c in NAMES if 10 <= c < 100),
    3: sorted(c for c in NAMES if c >= 100),
}


def to_level(codes: np.ndarray | int, level: int) -> np.ndarray | int:
    """Map level-3 codes (e.g. 312) to their level-1 (3) or level-2 (31) prefix. Non-CLC values map to 0."""
    if level not in (1, 2, 3):
        raise ValueError(f"level must be 1, 2 or 3, got {level}")
    arr = np.asarray(codes)
    out = arr // 10 ** (3 - level)
    valid = np.isin(arr, LEVEL_CODES[3])
    out = np.where(valid, out, 0)
    return out if isinstance(codes, np.ndarray) else int(out)


def fractions(mask: np.ndarray, level: int) -> dict[int, float]:
    """Fraction of valid pixels per class at the given level (classes with zero pixels omitted)."""
    lv = to_level(mask, level)
    valid = lv > 0
    n = int(valid.sum())
    if n == 0:
        return {}
    codes, counts = np.unique(lv[valid], return_counts=True)
    return {int(c): float(k) / n for c, k in zip(codes, counts)}
