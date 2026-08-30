"""Check that a v3 manifest's visible text matches the vocabulary it claims.

The first `tier_a_probe` manifest carried `{"square_display_name": "box"}` on
every row while 183 of 256 prompts and questions still said "square". The claim
is what downstream consumers read, so the mismatch was silent.

Read-only. Writes a report next to nothing it audits, and never edits a manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfsight.utils.jsonl import atomic_write_json, read_jsonl
from selfsight.v3.vocabulary import audit_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", help="JSONL scene manifests to audit")
    parser.add_argument("--output", default=None, help="Write a JSON report here")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = {}
    clean = True
    for value in args.manifests:
        path = Path(value).resolve()
        report = audit_rows(read_jsonl(path))
        reports[str(path)] = report
        clean = clean and report["clean"]
        status = "CLEAN" if report["clean"] else "VIOLATION"
        print(
            f"[vocabulary] {status:<9} {path.name}: "
            f"{report['rows_with_internal_wording']}/{report['rows']} rows use "
            f"'{report['internal_name']}' in visible text; "
            f"{report['rows_claiming_but_violating']} of those claim "
            f"'{report['display_name']}'",
            flush=True,
        )
        for example in report["examples"]:
            print(f"[vocabulary]     {example['scene_id']}: {example['texts'][0]!r}")

    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            output,
            {
                "schema_version": 1,
                "benchmark_version": "3.0",
                "stage": "v3_vocabulary_audit",
                "clean": clean,
                "manifests": reports,
            },
        )
        print(f"[vocabulary] report -> {output}", flush=True)
    else:
        print(json.dumps({"clean": clean}, indent=2))
    return 0 if clean else 2


if __name__ == "__main__":
    raise SystemExit(main())
