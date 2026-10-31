import numpy as np

from s1geo.clc import LEVEL_CODES, NAMES, fractions, to_level


def test_nomenclature_sizes():
    assert [len(LEVEL_CODES[l]) for l in (1, 2, 3)] == [5, 15, 44]
    for c in LEVEL_CODES[3]:
        assert c // 10 in NAMES and c // 100 in NAMES


def test_to_level_and_invalid():
    m = np.array([[312, 211], [0, 999]], dtype=np.uint16)
    assert to_level(m, 1).tolist() == [[3, 2], [0, 0]]
    assert to_level(m, 2).tolist() == [[31, 21], [0, 0]]
    assert to_level(312, 1) == 3


def test_fractions_ignore_invalid():
    m = np.array([311, 312, 211, 0], dtype=np.uint16)
    assert fractions(m, 1) == {2: 1 / 3, 3: 2 / 3}
    assert fractions(np.zeros(4, dtype=np.uint16), 1) == {}
