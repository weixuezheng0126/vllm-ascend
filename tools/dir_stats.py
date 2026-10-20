#!/usr/bin/env python3
"""Print a quick statistics summary for a directory tree.

Counts files by extension, reports total lines of text, and lists the
largest text files. Useful for a fast overview of an unfamiliar directory.

Examples
--------
    python tools/dir_stats.py .
    python tools/dir_stats.py vllm_ascend --top 20
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", "build", "dist"}
TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".cfg",
    ".ini",
    ".rst",
}


def iter_files(root: Path):
    """Yield all files under root, skipping well-known non-source directories."""
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def count_lines(path: Path) -> int:
    """Return the number of lines in a text file, or 0 if it cannot be read."""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=".", help="directory to scan")
    parser.add_argument("--top", type=int, default=10, help="how many large files to list")
    args = parser.parse_args()

    root = Path(args.path)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 1

    by_suffix: dict[str, int] = defaultdict(int)
    total_files = 0
    total_lines = 0
    largest: list[tuple[int, Path]] = []

    for path in iter_files(root):
        total_files += 1
        by_suffix[path.suffix or "<none>"] += 1
        if path.suffix in TEXT_SUFFIXES:
            lines = count_lines(path)
            total_lines += lines
            largest.append((lines, path))

    print(f"root       : {root.resolve()}")
    print(f"files      : {total_files}")
    print(f"text lines : {total_lines}")

    print("\nfiles by extension:")
    for suffix, count in sorted(by_suffix.items(), key=lambda item: (-item[1], item[0]))[:15]:
        print(f"  {suffix:<10} {count}")

    if largest:
        top = sorted(largest, key=lambda item: -item[0])[: args.top]
        print(f"\nlargest {len(top)} text files:")
        for lines, path in top:
            print(f"  {lines:>8}  {path.relative_to(root)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
