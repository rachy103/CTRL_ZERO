from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from sim import scope_guard


def test_changed_paths_collects_both_rename_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = b"R  sim/new_name.py\0sim/old_name.py\0"
    monkeypatch.setattr(
        scope_guard.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=output),
    )

    paths = scope_guard.changed_paths()

    assert paths == [
        (scope_guard.REPOSITORY_ROOT / "sim/new_name.py").resolve(),
        (scope_guard.REPOSITORY_ROOT / "sim/old_name.py").resolve(),
    ]


def test_changed_paths_collects_worktree_rename_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = b" R sim/new_name.py\0sim/old_name.py\0"
    monkeypatch.setattr(
        scope_guard.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=output),
    )

    paths = scope_guard.changed_paths()

    assert len(paths) == 2


def test_scope_guard_rejects_any_path_outside_sim() -> None:
    outside = (scope_guard.SIM_ROOT.parent / "main.py").resolve()

    with pytest.raises(scope_guard.ScopeViolation, match="outside sim"):
        scope_guard.assert_changes_confined_to_sim(
            [scope_guard.SIM_ROOT / "inside.py", outside]
        )
