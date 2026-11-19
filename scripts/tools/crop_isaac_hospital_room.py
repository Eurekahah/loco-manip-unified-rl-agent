#!/usr/bin/env python3
"""Crop a spatial region from Isaac's composed Hospital scene into a small USDA."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Crop a Hospital room by world-space bounds.")
parser.add_argument("--input", type=Path, required=True, help="Source hospital.usd.")
parser.add_argument("--output", type=Path, required=True, help="Destination .usda or .usd.")
parser.add_argument(
    "--bounds",
    type=float,
    nargs=6,
    required=True,
    metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    help="World-space crop bounds before recentering.",
)
parser.add_argument(
    "--translate",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("X", "Y", "Z"),
    help="Translation applied to all cropped prims.",
)
parser.add_argument(
    "--hospital-props-url",
    type=str,
    default=(
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/5.1/Isaac/Environments/Hospital/Props"
    ),
    help="Stable URL prefix replacing relative ./Props references.",
)
parser.add_argument(
    "--selection",
    choices=("contained", "intersects"),
    default="contained",
    help="Keep prims fully contained in bounds or merely intersecting bounds.",
)
parser.add_argument("--overwrite", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402


def _overlaps(box, bounds: tuple[float, ...]) -> bool:
    if box.IsEmpty():
        return False
    minimum, maximum = box.GetMin(), box.GetMax()
    return not (
        maximum[0] < bounds[0]
        or minimum[0] > bounds[1]
        or maximum[1] < bounds[2]
        or minimum[1] > bounds[3]
        or maximum[2] < bounds[4]
        or minimum[2] > bounds[5]
    )


def _contained(box, bounds: tuple[float, ...]) -> bool:
    if box.IsEmpty():
        return False
    minimum, maximum = box.GetMin(), box.GetMax()
    return (
        minimum[0] >= bounds[0]
        and maximum[0] <= bounds[1]
        and minimum[1] >= bounds[2]
        and maximum[1] <= bounds[3]
        and minimum[2] >= bounds[4]
        and maximum[2] <= bounds[5]
    )


def _repath_props_references(layer: Sdf.Layer, root_path: str, props_url: str) -> tuple[int, list[str]]:
    repathed = 0
    unresolved: list[str] = []
    paths: list[Sdf.Path] = []
    layer.Traverse(Sdf.Path(root_path), lambda path: paths.append(path))
    for path in paths:
        prim_spec = layer.GetPrimAtPath(path)
        if not prim_spec or not prim_spec.hasReferences:
            continue
        references = list(prim_spec.referenceList.GetAddedOrExplicitItems())
        replacements = []
        for reference in references:
            asset_path = reference.assetPath
            if not asset_path:
                unresolved.append(f"{path}: empty reference")
                continue
            if asset_path.startswith("./Props/"):
                replacements.append(Sdf.Reference(f"{props_url}/{Path(asset_path).name}"))
            else:
                unresolved.append(f"{path}: unsupported reference {asset_path}")
        if not replacements:
            continue
        prim_spec.ClearReferenceList()
        prim_spec.referenceList.prependedItems = replacements
        repathed += len(replacements)
    return repathed, unresolved


def main() -> None:
    source_path = args_cli.input.resolve()
    output_path = args_cli.output.resolve()
    if output_path.exists() and not args_cli.overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        os.remove(output_path)

    source_stage = Usd.Stage.Open(str(source_path))
    if source_stage is None:
        raise RuntimeError(f"could not open {source_path}")
    source_prim = source_stage.GetDefaultPrim()
    if not source_prim:
        raise RuntimeError(f"{source_path} has no default prim")

    destination_layer = Sdf.Layer.CreateNew(str(output_path))
    destination_layer.defaultPrim = "Root"
    destination_layer.pseudoRoot.SetInfo("upAxis", UsdGeom.GetStageUpAxis(source_stage))
    destination_layer.pseudoRoot.SetInfo(
        "metersPerUnit",
        UsdGeom.GetStageMetersPerUnit(source_stage),
    )

    root_spec = Sdf.CreatePrimInLayer(destination_layer, Sdf.Path("/Root"))
    root_spec.specifier = Sdf.SpecifierDef
    root_spec.typeName = "Xform"
    crop_spec = Sdf.CreatePrimInLayer(destination_layer, Sdf.Path("/Root/Crop"))
    crop_spec.specifier = Sdf.SpecifierDef
    crop_spec.typeName = "Xform"
    translate_attr = Sdf.AttributeSpec(
        crop_spec,
        "xformOp:translate",
        Sdf.ValueTypeNames.Double3,
    )
    translate_attr.default = Gf.Vec3d(*args_cli.translate)
    xform_order_attr = Sdf.AttributeSpec(
        crop_spec,
        "xformOpOrder",
        Sdf.ValueTypeNames.TokenArray,
    )
    xform_order_attr.default = ["xformOp:translate"]

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    copied = 0
    total = 0
    for source_child in source_prim.GetChildren():
        total += 1
        box = bbox_cache.ComputeWorldBound(source_child).ComputeAlignedRange()
        selection_test = _contained if args_cli.selection == "contained" else _overlaps
        if not selection_test(box, tuple(args_cli.bounds)):
            continue
        destination_path = f"/Root/Crop/{source_child.GetName()}"
        Sdf.CopySpec(
            source_stage.GetRootLayer(),
            source_child.GetPath(),
            destination_layer,
            destination_path,
        )
        copied += 1

    repathed, unresolved = _repath_props_references(
        destination_layer,
        "/Root/Crop",
        args_cli.hospital_props_url,
    )
    if unresolved:
        raise RuntimeError(
            "crop contains references that cannot be made portable: "
            + "; ".join(unresolved[:5])
        )
    destination_layer.Save()

    print(f"[crop] source={source_path}", flush=True)
    print(f"[crop] output={output_path}", flush=True)
    print(f"[crop] copied_top_level_prims={copied}/{total}", flush=True)
    print(f"[crop] repathed_prop_references={repathed}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
