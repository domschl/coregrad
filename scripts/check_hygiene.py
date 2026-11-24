#!/usr/bin/env python3
"""Repository hygiene guard for coregrad.

Ensures tracked and untracked-non-ignored files do not leak machine-specific
local paths, private directory names, or binary dataset/checkpoint artifacts.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Absolute Linux home path", re.compile(r"/home/[a-zA-Z0-9._-]+/")),
    ("Absolute macOS home path", re.compile(r"/Users/[a-zA-Z0-9._-]+/")),
    ("Tilde Storage/Docs path", re.compile(r"~/(Storage|Docs|Documents)/")),
]

FORBIDDEN_EXTENSIONS = {
    ".bin",
    ".idx",
    ".parquet",
    ".arrow",
    ".pt",
    ".pth",
    ".safetensors",
    ".ckpt",
    ".sqlite",
    ".db",
}

SELF_PATH = Path("scripts/check_hygiene.py")


def list_candidate_files(repo_root: Path) -> list[Path]:
    """Return tracked + untracked non-gitignored files."""
    cmd = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=True)
    return [Path(line) for line in res.stdout.splitlines() if line.strip()]


def check_repo(repo_root: Path) -> list[str]:
    violations: list[str] = []
    for rel_path in list_candidate_files(repo_root):
        full_path = repo_root / rel_path
        if not full_path.is_file():
            continue

        if rel_path.name.endswith(".local.yaml") or rel_path.name.endswith(".local.toml"):
            violations.append(f"{rel_path}: local config file must be gitignored")
            continue

        if rel_path.suffix.lower() in FORBIDDEN_EXTENSIONS:
            violations.append(f"{rel_path}: binary data/model artifact must not be tracked")
            continue

        if rel_path == SELF_PATH:
            continue

        try:
            text = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            for label, pattern in FORBIDDEN_PATTERNS:
                if pattern.search(line):
                    violations.append(f"{rel_path}:{lineno}: {label}")
    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    violations = check_repo(repo_root)
    if violations:
        print("Hygiene check FAILED:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1
    print("Hygiene check PASSED: no local paths or forbidden artifacts found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
