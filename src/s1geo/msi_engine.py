"""OneJev's production engine (qev.mm_engine.MMDecisionEngine) on 12-band Sentinel-2 patches (s1geo.msi).

The engine is used unchanged: prefix sharing across the questions of one image, CUDA graphs, fork-mode selection.
Two things are swapped in: the processor reads 12-band patches with the band statistics saved next to the model
(msi_norm.json), and media items that reference a Zarr row ({"type": "image", "path": "<split>.zarr", "row": i})
are read from the Zarr. The engine's own start-up self-test uses a synthetic RGB image; it is turned into a 12-band
patch (s1geo.msi.rgb_as_patch).

    from s1geo.msi_engine import MSIDecisionEngine
    eng = MSIDecisionEngine("runs/onejev/<run>/final", media_root="data/reben/subset_5pct", allow_local_paths=True)
"""
from __future__ import annotations

from pathlib import Path

from qev import media as qmedia
from qev.mm_engine import MMDecisionEngine

from s1geo import msi

_orig_load = qmedia.load
_cache: dict[tuple, object] = {}


def preload(root, media) -> None:
    """Read 12-band patches into memory, so a latency measurement starts from the decoded image like the other models'."""
    for m in media:
        if msi.is_msi(m):
            _cache[(str(root), m["path"], int(m["row"]))] = msi.load_patch(root, m)


def _patch(root, m):
    key = (str(root), m["path"], int(m["row"]))
    return _cache[key] if key in _cache else msi.load_patch(root, m)


def _load(media, root=None, allow_paths=True):
    if media and all(msi.is_msi(m) for m in media):
        return [_patch(root, m) for m in media], [], []
    if any(msi.is_msi(m) for m in media):
        raise qmedia.MediaError("a request mixes 12-band and other media")
    return _orig_load(media, root, allow_paths)


qmedia.load = _load


class MSIDecisionEngine(MMDecisionEngine):
    def __init__(self, model_path: str, msi_norm: str | None = None, **kwargs):
        self._msi_norm = msi.BandNorm.load(msi_norm or Path(model_path) / msi.NORM_FILE)
        super().__init__(model_path, **kwargs)
        self._use_msi_processor()

    def _use_msi_processor(self) -> None:
        if not isinstance(self.processor.image_processor, msi.MSIImageProcessor):
            self.processor = msi.msi_processor(self.processor, self._msi_norm)

    def _select_fork_mode(self) -> None:   # runs inside MMDecisionEngine.__init__, after the processor is loaded
        self._use_msi_processor()
        super()._select_fork_mode()
