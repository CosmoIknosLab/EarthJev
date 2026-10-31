from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    # Configs and schemas are referenced relative to the repository root.
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
