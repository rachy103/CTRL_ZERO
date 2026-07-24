from __future__ import annotations

import subprocess
from pathlib import Path


SIM_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SIM_ROOT.parent


class ScopeViolation(RuntimeError):
    pass


def changed_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    entries = [entry for entry in result.stdout.split(b"\0") if entry]
    paths: list[Path] = []
    index = 0
    while index < len(entries):
        entry = entries[index].decode("utf-8", errors="surrogateescape")
        status = entry[:2]
        first_path = entry[3:]
        paths.append((REPOSITORY_ROOT / first_path).resolve())
        if "R" in status or "C" in status:
            index += 1
            if index >= len(entries):
                raise ScopeViolation("Malformed git porcelain output for rename/copy.")
            second_path = entries[index].decode(
                "utf-8", errors="surrogateescape"
            )
            paths.append((REPOSITORY_ROOT / second_path).resolve())
        index += 1
    return paths


def assert_changes_confined_to_sim(paths: list[Path] | None = None) -> list[Path]:
    inspected = changed_paths() if paths is None else [path.resolve() for path in paths]
    violations: list[Path] = []
    for path in inspected:
        try:
            path.relative_to(SIM_ROOT)
        except ValueError:
            violations.append(path)
    if violations:
        rendered = "\n".join(f"  - {path}" for path in violations)
        raise ScopeViolation(f"Changes outside sim are forbidden:\n{rendered}")
    return inspected


def main() -> int:
    inspected = assert_changes_confined_to_sim()
    print(f"scope ok: {len(inspected)} changed path(s), all inside {SIM_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
