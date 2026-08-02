#!/usr/bin/env python3
"""Reconstruct a Stage 2 tar payload from Base64 text chunks.

The historical payload chunks may represent either one split Base64 stream or
individually encoded binary chunks. Some archived gzip streams also have a
corrupt trailer even though the tar members are complete. This utility tries
bounded, deterministic decoding strategies and accepts a candidate only when
it is a safe tar archive containing every required file basename.
"""

from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import io
import tarfile
from pathlib import Path


def _add_candidate(
    candidates: list[tuple[str, bytes]], label: str, payload: bytes
) -> None:
    if payload:
        candidates.append((label, payload))


def _candidate_payloads(encoded_parts: list[bytes]) -> list[tuple[str, bytes]]:
    candidates: list[tuple[str, bytes]] = []
    orders = [("forward", encoded_parts)]
    if len(encoded_parts) > 1:
        orders.append(("reverse", list(reversed(encoded_parts))))

    for order_label, ordered in orders:
        joined = b"".join(ordered)
        try:
            _add_candidate(
                candidates,
                f"{order_label}:joined-base64",
                base64.b64decode(joined, validate=True),
            )
        except (ValueError, base64.binascii.Error):
            pass

        try:
            decoded_parts = [
                base64.b64decode(part, validate=True) for part in ordered
            ]
        except (ValueError, base64.binascii.Error):
            continue

        decoded_joined = b"".join(decoded_parts)
        _add_candidate(
            candidates, f"{order_label}:per-part-base64", decoded_joined
        )
        try:
            _add_candidate(
                candidates,
                f"{order_label}:per-part-double-base64",
                base64.b64decode(decoded_joined, validate=True),
            )
        except (ValueError, base64.binascii.Error):
            pass

    unique: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for label, payload in candidates:
        digest = hashlib.sha256(payload).hexdigest()
        if digest not in seen:
            unique.append((label, payload))
            seen.add(digest)
    return unique


def reconstruct_archive(
    part_paths: list[Path], destination: Path, required_basenames: set[str]
) -> str:
    if not part_paths:
        raise FileNotFoundError("No payload parts matched")

    encoded_parts = [b"".join(path.read_bytes().split()) for path in part_paths]
    print(
        "payload_parts=",
        [(path.name, len(data), data.count(b"=")) for path, data in zip(part_paths, encoded_parts)],
    )

    failures: list[str] = []
    for label, payload in _candidate_payloads(encoded_parts):
        digest = hashlib.sha256(payload).hexdigest()
        try:
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
                members = archive.getmembers()
                file_basenames = {
                    Path(member.name).name for member in members if member.isfile()
                }
                missing = required_basenames - file_basenames
                if missing:
                    raise ValueError(f"missing required files: {sorted(missing)}")
                for member in members:
                    member_path = Path(member.name)
                    if member_path.is_absolute() or ".." in member_path.parts:
                        raise ValueError(f"unsafe tar path: {member.name}")
                destination.mkdir(parents=True, exist_ok=True)
                archive.extractall(destination)
            print(
                f"selected_candidate={label} bytes={len(payload)} sha256={digest}"
            )
            return label
        except (tarfile.TarError, EOFError, OSError, ValueError) as exc:
            failures.append(f"{label}:{type(exc).__name__}:{exc}")

    raise RuntimeError("No valid archive candidate. " + " | ".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", required=True, help="Glob for ordered .b64 parts")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--required", action="append", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    part_paths = [Path(path) for path in sorted(glob.glob(args.parts))]
    reconstruct_archive(
        part_paths=part_paths,
        destination=Path(args.destination),
        required_basenames=set(args.required),
    )


if __name__ == "__main__":
    main()
