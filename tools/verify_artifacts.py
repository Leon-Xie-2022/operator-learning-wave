"""Verify hashes for frozen paper artifacts.

Use this script after downloading an artifact bundle that contains generated
datasets, checkpoints, evaluation summaries, and paper figures. It is read-only:
it never regenerates data or overwrites results.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict[str, str]:
    hashes = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid manifest line {line_number}: {line}")
        expected_hash, relative_path = parts
        hashes[relative_path.replace("\\", "/")] = expected_hash.lower()
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify frozen paper artifact hashes")
    parser.add_argument("manifest", help="Path to a sha256 manifest")
    parser.add_argument("--root", default=".", help="Repository or artifact root")
    args = parser.parse_args()

    root = Path(args.root)
    manifest = read_manifest(Path(args.manifest))
    missing = []
    mismatched = []

    for relative_path, expected_hash in manifest.items():
        path = root / relative_path
        if not path.exists():
            missing.append(relative_path)
            continue
        actual_hash = file_sha256(path)
        if actual_hash.lower() != expected_hash:
            mismatched.append((relative_path, expected_hash, actual_hash))

    if missing:
        print("Missing files:")
        for path in missing:
            print(f"  {path}")

    if mismatched:
        print("Hash mismatches:")
        for path, expected_hash, actual_hash in mismatched:
            print(f"  {path}")
            print(f"    expected: {expected_hash}")
            print(f"    actual:   {actual_hash}")

    if missing or mismatched:
        return 1

    print(f"Verified {len(manifest)} artifact files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
