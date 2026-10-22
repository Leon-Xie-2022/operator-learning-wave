"""Create a SHA-256 manifest for generated paper artifacts.

The manifest is meant to be created from the frozen artifact bundle that
matches the submitted paper. It should be regenerated only when the paper
artifacts are intentionally updated.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


DEFAULT_PATTERNS = (
    "data/*_dataset.npz",
    "results/**/*.json",
    "results/**/*.csv",
    "results/**/*.txt",
    "results/**/*.png",
    "results/**/*.pth",
    "icml2026paper/*.png",
    "icml2026paper/*.pdf",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    files = []
    for pattern in patterns:
        files.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(set(files), key=lambda path: path.as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a SHA-256 artifact manifest")
    parser.add_argument("--root", default=".", help="Repository or artifact root")
    parser.add_argument("--output", default="artifact_manifest.sha256", help="Manifest output path")
    parser.add_argument("--pattern", action="append", help="Additional glob pattern to include")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    patterns = DEFAULT_PATTERNS + tuple(args.pattern or ())
    files = collect_files(root, patterns)

    lines = [
        "# SHA-256 manifest for the frozen paper artifact bundle",
        "# Format: <sha256> <relative-path>",
    ]
    for path in files:
        relative_path = path.relative_to(root).as_posix()
        lines.append(f"{file_sha256(path)} {relative_path}")

    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(files)} entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
