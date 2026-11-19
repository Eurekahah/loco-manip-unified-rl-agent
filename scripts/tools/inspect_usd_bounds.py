#!/usr/bin/env python3
"""Print the world-space bounds of USD prims for scene-layout diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect USD metadata and prim bounds.")
parser.add_argument("usd_paths", nargs="+", type=Path, help="USD files to inspect.")
parser.add_argument(
    "--all-prims",
    action="store_true",
    help="Print every Xformable prim instead of only the default prim.",
)
parser.add_argument(
    "--apis",
    action="store_true",
    help="Print prim type and applied API schemas for every prim that declares any API.",
)
parser.add_argument(
    "--references",
    action="store_true",
    help="Print prims that carry references or payload composition arcs.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom  # noqa: E402


def _bounds(cache: UsdGeom.BBoxCache, prim) -> str:
    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if box.IsEmpty():
        return "<empty>"
    minimum, maximum = box.GetMin(), box.GetMax()
    return (
        f"min=({minimum[0]:.4f}, {minimum[1]:.4f}, {minimum[2]:.4f}) "
        f"max=({maximum[0]:.4f}, {maximum[1]:.4f}, {maximum[2]:.4f})"
    )


def main() -> None:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    for path in args_cli.usd_paths:
        path = path.resolve()
        stage = Usd.Stage.Open(str(path))
        if stage is None:
            print(f"ERROR open failed: {path}", flush=True)
            continue
        default_prim = stage.GetDefaultPrim()
        print(f"\nUSD: {path}", flush=True)
        print(
            f"defaultPrim={default_prim.GetPath() if default_prim else '<none>'} "
            f"upAxis={UsdGeom.GetStageUpAxis(stage)} "
            f"metersPerUnit={UsdGeom.GetStageMetersPerUnit(stage)}",
            flush=True,
        )
        if default_prim:
            print(f"default: {default_prim.GetName()} {_bounds(cache, default_prim)}", flush=True)
        if args_cli.apis:
            for prim in stage.Traverse():
                applied_schemas = list(prim.GetAppliedSchemas())
                if applied_schemas:
                    print(
                        f"API {prim.GetPath()} type={prim.GetTypeName()} schemas={applied_schemas}",
                        flush=True,
                    )
        if args_cli.references:
            for prim in stage.Traverse():
                references = prim.GetMetadata("references")
                payload = prim.GetMetadata("payload")
                if references or payload:
                    print(
                        f"ARC {prim.GetPath()} type={prim.GetTypeName()} "
                        f"references={references} payload={payload}",
                        flush=True,
                    )
        if args_cli.all_prims:
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Xformable):
                    print(f"{prim.GetPath()} {_bounds(cache, prim)}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
