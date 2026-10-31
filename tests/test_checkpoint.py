import torch

from s1geo.utils.checkpoint import LAST_CKPT, load_checkpoint, save_checkpoint


def test_roundtrip_and_no_tmp_left(tmp_path):
    save_checkpoint({"step": 3, "rng_cpu": torch.get_rng_state()}, tmp_path)
    state = load_checkpoint(tmp_path)
    assert state["step"] == 3
    assert not list((tmp_path / LAST_CKPT).parent.glob("*.tmp"))
    # Default CPU loading keeps the RNG state usable (regression: job 5885).
    torch.set_rng_state(state["rng_cpu"])


def test_missing_checkpoint_returns_none(tmp_path):
    assert load_checkpoint(tmp_path) is None
