"""Materialize immutable per-seed A800 YAML configs from the registered base profile."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base", type=Path, default=Path("configs/a800_80g_showo2.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in values["training"]["seeds"]]
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        config = deepcopy(values)
        config["profile"] = f"{values['profile']}_seed_{seed}"
        config["seed"] = seed
        config["training"]["seeds"] = [seed]
        path = args.output / f"a800_seed_{seed}.yaml"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite materialized seed config: {path}")
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
        print(path)


if __name__ == "__main__":
    main()
