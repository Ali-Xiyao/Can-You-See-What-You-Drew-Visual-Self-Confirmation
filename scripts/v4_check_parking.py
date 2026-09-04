"""Does parking the backbone actually give the card back?

The fix assumes it does. Reading the code does not settle it: the model is
loaded through a device_map, accelerate wraps `.to` and attaches
AlignDevicesHook to the blocks, and a hook that holds its own reference to the
weights would leave them on the card no matter what `.to("cpu")` reports. The
whole point of the fix is the free memory, so measure the free memory.

Five minutes here against an eleven-hour run that would otherwise fail the same
way it failed the first time.

    envs/showo2/python.exe scripts/v4_check_parking.py --device cuda:0
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


def used_mib(index: int) -> int:
    import torch

    free, total = torch.cuda.mem_get_info(index)
    return (total - free) // (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--need", type=int, default=17000,
                        help="MiB the ladder's adjudicator needs to load")
    args = parser.parse_args()

    import torch

    from selfsight.backbones.showo2 import Showo2Adapter

    index = torch.device(args.device).index or 0
    baseline = used_mib(index)
    print(f"before loading:      {baseline:6d} MiB used on {args.device}")

    backbone = Showo2Adapter(device=args.device, lazy=False)
    resident = used_mib(index)
    print(f"backbone resident:   {resident:6d} MiB used  (+{resident - baseline})")

    with backbone.parked():
        parked = used_mib(index)
        total = torch.cuda.mem_get_info(index)[1] // (1024 * 1024)
        headroom = total - parked
        print(f"while parked:        {parked:6d} MiB used  "
              f"(freed {resident - parked}, leaves {headroom} MiB)")

    restored = used_mib(index)
    print(f"after unparking:     {restored:6d} MiB used")

    # A forward after the round trip is the part that would fail silently: the
    # blocks carry accelerate hooks, and a parameter left behind on the CPU
    # would surface as a device mismatch here rather than at the next round.
    with tempfile.TemporaryDirectory() as scratch:
        drawn = backbone.generate_images(["a red apple on a table"], seeds=[0],
                                         output_dir=scratch, checkpoint_id="parking-check")
        drew = Path(drawn[0].image_path).exists()
    print(f"generation after the round trip: {'drew an image' if drew else 'NO IMAGE'}")

    ok = drew and headroom > args.need
    print()
    print("VERDICT:", "parking gives the card back and survives a forward" if ok
          else f"NOT GOOD -- freed {resident - parked} MiB, {headroom} MiB free "
               f"against {args.need} needed, generation after unparking: {drew}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
