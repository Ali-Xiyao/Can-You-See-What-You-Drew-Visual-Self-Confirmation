"""CandidateManifest storage and paired-pool invariants."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from selfsight.schemas import CandidateRecord
from selfsight.utils.hashing import rgb_sha256, sha256_file
from selfsight.utils.jsonl import atomic_write_jsonl, read_jsonl


class CandidateManifest:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read(self) -> list[CandidateRecord]:
        if not self.path.exists():
            return []
        return [CandidateRecord.from_dict(record) for record in read_jsonl(self.path)]

    def write(self, records: Iterable[CandidateRecord], verify_rgb: bool = True) -> Path:
        records = list(records)
        ids = [record.candidate_id for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate candidate_id in manifest")
        for record in records:
            path = Path(record.image_path)
            if not path.is_absolute():
                raise ValueError(f"Candidate path must be absolute: {path}")
            if verify_rgb:
                if not path.is_file():
                    raise FileNotFoundError(path)
                actual = rgb_sha256(path)
                if actual != record.rgb_sha256:
                    raise ValueError(f"RGB hash mismatch for {record.candidate_id}")
        return atomic_write_jsonl(self.path, (record.to_dict() for record in records))

    def append(self, records: Iterable[CandidateRecord], verify_rgb: bool = True) -> Path:
        return self.write([*self.read(), *records], verify_rgb=verify_rgb)

    def file_digest(self) -> str:
        return sha256_file(self.path)


def pool_signature(records: Iterable[CandidateRecord]) -> dict[str, tuple[tuple[str, int, str], ...]]:
    grouped: dict[str, list[tuple[str, int, str]]] = {}
    for record in records:
        grouped.setdefault(record.prompt_id, []).append(
            (record.candidate_id, record.sampling_seed, record.rgb_sha256)
        )
    return {key: tuple(sorted(values)) for key, values in grouped.items()}


def assert_paired_candidate_pools(
    left: Iterable[CandidateRecord], right: Iterable[CandidateRecord], expected_k: int
) -> None:
    left_signature = pool_signature(left)
    right_signature = pool_signature(right)
    if left_signature != right_signature:
        missing_left = sorted(set(right_signature).difference(left_signature))
        missing_right = sorted(set(left_signature).difference(right_signature))
        mismatched = sorted(
            key for key in set(left_signature).intersection(right_signature)
            if left_signature[key] != right_signature[key]
        )
        raise ValueError(
            f"Unpaired candidate pools: missing_left={missing_left[:5]}, "
            f"missing_right={missing_right[:5]}, mismatched={mismatched[:5]}"
        )
    bad_k = {key: len(values) for key, values in left_signature.items() if len(values) != expected_k}
    if bad_k:
        raise ValueError(f"Candidate pools do not have K={expected_k}: {bad_k}")
