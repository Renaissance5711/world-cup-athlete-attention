#!/usr/bin/env python3
"""Reconstruct a Stage 2 tar payload from Base64 text chunks.

The historical payload chunks may represent either one split Base64 stream or
individually encoded binary chunks. Some archived gzip streams also have a
corrupt trailer even though the tar members are complete. A legacy panel
payload additionally contains one surplus Base64 character. This utility uses
bounded deterministic repairs and accepts a candidate only when it is a safe
tar archive containing every required file basename.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import glob
import hashlib
import io
import tarfile
import zlib
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
        except (ValueError, binascii.Error):
            pass

        try:
            decoded_parts = [
                base64.b64decode(part, validate=True) for part in ordered
            ]
        except (ValueError, binascii.Error):
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
        except (ValueError, binascii.Error):
            pass

    unique: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for label, payload in candidates:
        digest = hashlib.sha256(payload).hexdigest()
        if digest not in seen:
            unique.append((label, payload))
            seen.add(digest)
    return unique


def _gunzip_without_trailer(data: bytes) -> bytes:
    """Decode one gzip member while intentionally ignoring CRC/ISIZE trailer."""
    if len(data) < 18 or data[:2] != b"\x1f\x8b" or data[2] != 8:
        raise ValueError("not a supported gzip stream")

    flags = data[3]
    position = 10
    if flags & 0x04:
        if position + 2 > len(data):
            raise ValueError("truncated gzip extra header")
        extra_length = int.from_bytes(data[position : position + 2], "little")
        position += 2 + extra_length
    if flags & 0x08:
        position = data.index(b"\0", position) + 1
    if flags & 0x10:
        position = data.index(b"\0", position) + 1
    if flags & 0x02:
        position += 2
    if position >= len(data) - 8:
        raise ValueError("invalid gzip framing")

    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    output = decompressor.decompress(data[position:-8]) + decompressor.flush()
    if not decompressor.eof:
        raise ValueError("truncated or corrupt deflate stream")
    return output


def _extract_if_valid(
    label: str,
    payload: bytes,
    destination: Path,
    required_basenames: set[str],
) -> tuple[bool, str]:
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
        print(f"selected_candidate={label} bytes={len(payload)} sha256={digest}")
        return True, ""
    except (tarfile.TarError, EOFError, OSError, ValueError) as exc:
        return False, f"{label}:{type(exc).__name__}:{exc}"


def _single_surplus_character_repair(
    encoded_parts: list[bytes],
    destination: Path,
    required_basenames: set[str],
) -> str | None:
    joined = b"".join(encoded_parts)
    if len(joined) % 4 != 1:
        return None

    boundaries: list[int] = []
    running = 0
    for part in encoded_parts[:-1]:
        running += len(part)
        boundaries.append(running)

    targeted_positions: set[int] = set()
    for boundary in boundaries:
        targeted_positions.update(
            position
            for position in range(max(0, boundary - 32), min(len(joined), boundary + 33))
        )
    targeted_positions.update(range(min(32, len(joined))))
    targeted_positions.update(range(max(0, len(joined) - 32), len(joined)))

    def try_position(position: int) -> str | None:
        repaired = joined[:position] + joined[position + 1 :]
        try:
            compressed = base64.b64decode(repaired, validate=True)
            raw_tar = _gunzip_without_trailer(compressed)
        except (ValueError, binascii.Error, zlib.error):
            return None
        label = f"forward:delete-surplus-base64-char@{position}"
        valid, _ = _extract_if_valid(
            label, raw_tar, destination, required_basenames
        )
        return label if valid else None

    for position in sorted(targeted_positions):
        label = try_position(position)
        if label:
            print(f"repair_scope=targeted scanned={len(targeted_positions)}")
            return label

    scanned = 0
    for position in range(len(joined)):
        if position in targeted_positions:
            continue
        scanned += 1
        label = try_position(position)
        if label:
            print(f"repair_scope=full scanned={scanned}")
            return label
    return None


def reconstruct_archive(
    part_paths: list[Path], destination: Path, required_basenames: set[str]
) -> str:
    if not part_paths:
        raise FileNotFoundError("No payload parts matched")

    encoded_parts = [b"".join(path.read_bytes().split()) for path in part_paths]
    print(
        "payload_parts=",
        [
            (path.name, len(data), data.count(b"="))
            for path, data in zip(part_paths, encoded_parts)
        ],
    )

    failures: list[str] = []
    for label, payload in _candidate_payloads(encoded_parts):
        valid, failure = _extract_if_valid(
            label, payload, destination, required_basenames
        )
        if valid:
            return label
        failures.append(failure)

    repaired_label = _single_surplus_character_repair(
        encoded_parts, destination, required_basenames
    )
    if repaired_label:
        return repaired_label

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
