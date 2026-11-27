#!/usr/bin/env python3
"""Convert a URDF assembly to a named STEP (AP214/AP242) model.

URDF is a kinematic description whose links reference triangle meshes (STL) or
Collada scenes (DAE).  STEP is a B-rep format, so this tool rebuilds every
triangle as a planar CAD face, sews the faces into shells/solids, and writes a
hierarchical XCAF assembly that keeps the URDF link and joint names.

Important conventions:

* URDF lengths are metres.  STEP is written in millimetres, which is what most
  CAD packages expect.  All mesh vertices and joint offsets are scaled by 1000.
* The model is exported in the URDF zero-joint configuration.
* Mesh decimation is applied by default.  ``--max-faces-per-link`` controls the
  trade-off between STEP file size and geometric fidelity.

Example:

    python scripts/tools/urdf_to_step.py \
        --urdf outputs/handoff/AS2W_PiperX_urdf_bundle/piper_x_description/urdf/piper_x_description.urdf \
        --output outputs/step_exports/PiperX.step \
        --max-faces-per-link 4000
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Small vector / transform helpers
# ---------------------------------------------------------------------------


def _floats(text: str | None, default: Sequence[float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=float)
    values = [float(x) for x in text.replace(",", " ").split()]
    if len(values) != len(default):
        raise ValueError(f"expected {len(default)} values, got {values!r}")
    return np.asarray(values, dtype=float)


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = (float(x) for x in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def origin_matrix(elem: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if elem is None:
        return transform
    xyz = _floats(elem.get("xyz"), (0.0, 0.0, 0.0))
    rpy = _floats(elem.get("rpy"), (0.0, 0.0, 0.0))
    transform[:3, :3] = rpy_to_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def apply_transform(transform: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    return vertices @ transform[:3, :3].T + transform[:3, 3]


def translate_mm(transform_m: np.ndarray) -> np.ndarray:
    """Copy a metre transform and convert its translation to millimetres."""
    out = np.array(transform_m, dtype=float, copy=True)
    out[:3, 3] *= 1000.0
    return out


def sanitize_name(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.\-]+", "_", name).strip("_")
    if not cleaned:
        cleaned = fallback
    return cleaned[:96]


# ---------------------------------------------------------------------------
# URDF model
# ---------------------------------------------------------------------------


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


@dataclass
class UrdfModel:
    path: Path
    robot_name: str
    links: dict[str, ET.Element]
    joints: list[Joint]
    package_roots: dict[str, Path]
    root: str = ""

    def __post_init__(self) -> None:
        children = {joint.child for joint in self.joints}
        roots = [name for name in self.links if name not in children]
        if len(roots) != 1:
            raise ValueError(
                f"{self.path}: expected one root link, found {roots!r}. "
                "A URDF with multiple disconnected trees is not supported."
            )
        self.root = roots[0]

    def child_joints(self) -> dict[str, list[Joint]]:
        result: dict[str, list[Joint]] = defaultdict(list)
        for joint in self.joints:
            result[joint.parent].append(joint)
        return result

    def world_transforms(self) -> dict[str, np.ndarray]:
        """Zero-configuration transform from every link frame to the root frame."""
        result = {self.root: np.eye(4)}
        by_parent = self.child_joints()
        stack = [self.root]
        while stack:
            parent = stack.pop()
            for joint in by_parent.get(parent, []):
                result[joint.child] = result[parent] @ joint.origin
                stack.append(joint.child)
        return result


def build_package_roots(urdf_path: Path, explicit: dict[str, Path]) -> dict[str, Path]:
    roots = {name: path.resolve() for name, path in explicit.items()}
    text = urdf_path.read_text(encoding="utf-8", errors="replace")
    package_names = set(re.findall(r"package://([^/\s\"']+)/", text))
    for package in package_names:
        if package in roots:
            continue
        for parent in urdf_path.parents:
            if parent.name == package:
                roots[package] = parent
                break
    return roots


def parse_urdf(
    urdf_path: Path, package_roots: dict[str, Path] | None = None
) -> UrdfModel:
    urdf_path = urdf_path.resolve()
    root = ET.parse(urdf_path).getroot()
    if root.tag != "robot":
        raise ValueError(f"{urdf_path}: expected <robot>, got <{root.tag}>")
    links = {link.get("name", ""): link for link in root.findall("link")}
    if "" in links:
        raise ValueError(f"{urdf_path}: found a link without a name")

    joints: list[Joint] = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            raise ValueError(f"{urdf_path}: joint {joint.get('name')!r} lacks parent/child")
        axis_elem = joint.find("axis")
        joints.append(
            Joint(
                name=joint.get("name", ""),
                kind=joint.get("type", "fixed"),
                parent=parent.get("link", ""),
                child=child.get("link", ""),
                origin=origin_matrix(joint.find("origin")),
                axis=_floats(axis_elem.get("xyz") if axis_elem is not None else None, (1.0, 0.0, 0.0)),
            )
        )

    for joint in joints:
        if joint.parent not in links:
            raise ValueError(f"{urdf_path}: joint {joint.name!r} references unknown parent {joint.parent!r}")
        if joint.child not in links:
            raise ValueError(f"{urdf_path}: joint {joint.name!r} references unknown child {joint.child!r}")

    roots = build_package_roots(urdf_path, package_roots or {})
    return UrdfModel(
        path=urdf_path,
        robot_name=root.get("name", urdf_path.stem),
        links=links,
        joints=joints,
        package_roots=roots,
    )


# ---------------------------------------------------------------------------
# Mesh loading / simplification
# ---------------------------------------------------------------------------


@dataclass
class MeshPart:
    name: str
    vertices: np.ndarray  # metres, in the link frame
    faces: np.ndarray
    color: tuple[float, float, float]
    source_faces: int = 0

    def __post_init__(self) -> None:
        if self.source_faces == 0:
            self.source_faces = len(self.faces)


def resolve_mesh_path(
    filename: str, urdf_dir: Path, package_roots: dict[str, Path]
) -> Path:
    if filename.startswith("package://"):
        rest = filename[len("package://") :]
        if "/" not in rest:
            raise ValueError(f"bad package URI: {filename}")
        package, relative = rest.split("/", 1)
        root = package_roots.get(package)
        if root is None:
            raise ValueError(
                f"cannot resolve {filename}: package {package!r} is unknown. "
                "Pass --package-root package=/path/to/package."
            )
        return (root / relative).resolve()
    path = Path(filename)
    if not path.is_absolute():
        path = urdf_dir / path
    return path.resolve()


def _color_from_material(material: ET.Element | None) -> tuple[float, float, float] | None:
    if material is None:
        return None
    color = material.find("color")
    if color is None:
        return None
    rgba = _floats(color.get("rgba"), (0.65, 0.65, 0.68, 1.0))
    return tuple(float(min(max(x, 0.0), 1.0)) for x in rgba[:3])


def _color_from_geometry(geom: trimesh.Trimesh, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    visual = getattr(geom, "visual", None)
    material = getattr(visual, "material", None)
    for attr in ("main_color", "baseColorFactor", "base_color"):
        value = getattr(material, attr, None)
        if value is None:
            continue
        arr = np.asarray(value).ravel()
        if arr.size < 3:
            continue
        if np.nanmax(arr) > 1.5:
            arr = arr / 255.0
        return tuple(float(min(max(x, 0.0), 1.0)) for x in arr[:3])
    face_colors = getattr(visual, "face_colors", None)
    if face_colors is not None and len(face_colors):
        arr = np.asarray(face_colors)[0][:3] / 255.0
        return tuple(float(x) for x in arr)
    return fallback


def _mesh_parts_from_scene(
    scene: trimesh.Scene,
    origin: np.ndarray,
    scale: np.ndarray,
    fallback_color: tuple[float, float, float],
    name_prefix: str,
) -> list[MeshPart]:
    parts: list[MeshPart] = []
    nodes = list(scene.graph.nodes_geometry)
    for index, node in enumerate(nodes):
        scene_transform, geometry_name = scene.graph.get(node)
        geom = scene.geometry.get(geometry_name)
        if geom is None or len(geom.faces) == 0:
            continue
        vertices = apply_transform(scene_transform, np.asarray(geom.vertices, dtype=float))
        vertices = vertices * scale
        vertices = apply_transform(origin, vertices)
        name = geometry_name
        if name.startswith("mesh-") or len(name) > 56:
            name = f"{name_prefix}_{index:03d}"
        parts.append(
            MeshPart(
                name=sanitize_name(name, f"{name_prefix}_{index:03d}"),
                vertices=vertices,
                faces=np.asarray(geom.faces, dtype=np.int64),
                color=_color_from_geometry(geom, fallback_color),
            )
        )
    return parts


def load_mesh_parts(
    filename: str,
    urdf_dir: Path,
    package_roots: dict[str, Path],
    origin: np.ndarray,
    scale: np.ndarray,
    fallback_color: tuple[float, float, float],
    name_prefix: str,
) -> list[MeshPart]:
    path = resolve_mesh_path(filename, urdf_dir, package_roots)
    if not path.exists():
        raise FileNotFoundError(f"{path} (referenced by URDF)")
    loaded = trimesh.load(str(path), force="scene")
    if isinstance(loaded, trimesh.Scene):
        return _mesh_parts_from_scene(
            loaded, origin, scale, fallback_color, name_prefix
        )
    geom = loaded
    vertices = np.asarray(geom.vertices, dtype=float) * scale
    vertices = apply_transform(origin, vertices)
    return [
        MeshPart(
            name=sanitize_name(name_prefix, "part"),
            vertices=vertices,
            faces=np.asarray(geom.faces, dtype=np.int64),
            color=_color_from_geometry(geom, fallback_color),
        )
    ]


def _primitive_part(geometry: ET.Element, origin: np.ndarray, color: tuple[float, float, float], name: str) -> MeshPart | None:
    if geometry.find("box") is not None:
        size = _floats(geometry.find("box").get("size"), (1.0, 1.0, 1.0))
        mesh = trimesh.creation.box(extents=size)
    elif geometry.find("cylinder") is not None:
        elem = geometry.find("cylinder")
        mesh = trimesh.creation.cylinder(
            radius=float(elem.get("radius", 0.1)),
            height=float(elem.get("length", 0.1)),
            sections=48,
        )
    elif geometry.find("sphere") is not None:
        elem = geometry.find("sphere")
        mesh = trimesh.creation.icosphere(
            radius=float(elem.get("radius", 0.1)), subdivisions=3
        )
    else:
        return None
    return MeshPart(
        name=name,
        vertices=apply_transform(origin, np.asarray(mesh.vertices, dtype=float)),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        color=color,
    )


def load_link_parts(
    model: UrdfModel, link_name: str
) -> list[MeshPart]:
    link = model.links[link_name]
    parts: list[MeshPart] = []
    urdf_dir = model.path.parent
    for visual_index, visual in enumerate(link.findall("visual")):
        origin = origin_matrix(visual.find("origin"))
        color = _color_from_material(visual.find("material")) or (0.65, 0.65, 0.68)
        geometry = visual.find("geometry")
        if geometry is None:
            continue
        mesh = geometry.find("mesh")
        if mesh is not None:
            filename = mesh.get("filename")
            if not filename:
                continue
            scale = _floats(mesh.get("scale"), (1.0, 1.0, 1.0))
            parts.extend(
                load_mesh_parts(
                    filename,
                    urdf_dir,
                    model.package_roots,
                    origin,
                    scale,
                    color,
                    f"{link_name}_v{visual_index}",
                )
            )
        else:
            part = _primitive_part(
                geometry, origin, color, f"{link_name}_v{visual_index}"
            )
            if part is not None:
                parts.append(part)
    return parts


def _decimate_with_pymeshlab(
    vertices: np.ndarray, faces: np.ndarray, target: int
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import pymeshlab  # type: ignore
    except Exception:
        return None
    try:
        mesh_set = pymeshlab.MeshSet()
        mesh_set.add_mesh(
            pymeshlab.Mesh(
                vertices.astype(np.float64), faces.astype(np.int32)
            )
        )
        mesh_set.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target),
            qualitythr=0.3,
            preserveboundary=True,
            preservenormal=True,
            optimalplacement=True,
            planarquadric=True,
        )
        current = mesh_set.current_mesh()
        new_vertices = np.asarray(current.vertex_matrix(), dtype=float)
        new_faces = np.asarray(current.face_matrix(), dtype=np.int64)
        if len(new_faces) == 0:
            return None
        return new_vertices, new_faces
    except Exception:
        return None


def _decimate_with_fast(
    vertices: np.ndarray, faces: np.ndarray, target: int
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import fast_simplification  # type: ignore
    except Exception:
        return None
    reduction = 1.0 - float(target) / float(len(faces))
    reduction = min(max(reduction, 0.01), 0.999)
    try:
        new_vertices, new_faces = fast_simplification.simplify(
            vertices.astype(np.float32),
            faces.astype(np.int32),
            target_reduction=reduction,
        )
        new_vertices = np.asarray(new_vertices, dtype=float)
        new_faces = np.asarray(new_faces, dtype=np.int64)
        if len(new_faces) == 0 or not np.isfinite(new_vertices).all():
            return None
        return new_vertices, new_faces
    except Exception:
        return None


def decimate_part(part: MeshPart, target: int) -> tuple[MeshPart, str]:
    count = len(part.faces)
    if target >= count or count <= 8:
        return part, "kept"
    target = max(20, int(target))
    result: tuple[np.ndarray, np.ndarray] | None = None
    method = "kept"
    if count >= 4000:
        result = _decimate_with_pymeshlab(part.vertices, part.faces, target)
        method = "pymeshlab"
        # Preserve-boundary decimation refuses to collapse open DAE sheets, so
        # fall back to the quadric simplifier when the budget was not reached.
        if result is not None and len(result[1]) > target * 1.5:
            result = None
    if result is None:
        result = _decimate_with_fast(part.vertices, part.faces, target)
        method = "fast_simplification"
    if result is None:
        return part, "kept"
    # Aggressive simplification of thin/open sheets can create a single triangle
    # spanning the whole part.  Retry with a larger budget before giving up.
    candidates: list[tuple[np.ndarray, np.ndarray, str]] = [
        (result[0], result[1], method)
    ]
    for factor in (2, 4):
        retry_target = min(count, target * factor)
        if retry_target >= count or retry_target == target:
            continue
        retry = _decimate_with_fast(part.vertices, part.faces, retry_target)
        if retry is not None:
            candidates.append((retry[0], retry[1], "fast_simplification"))
    for vertices, faces, candidate_method in candidates:
        if len(faces) < 4:
            continue
        candidate = MeshPart(
            name=part.name,
            vertices=vertices,
            faces=faces,
            color=part.color,
            source_faces=part.source_faces,
        )
        if _decimation_is_plausible(part, candidate):
            return candidate, candidate_method
    return part, "kept"


def _decimation_is_plausible(original: MeshPart, candidate: MeshPart) -> bool:
    """Reject decimations that shrink the part or create part-spanning triangles."""
    if len(candidate.faces) == 0 or len(candidate.vertices) == 0:
        return False
    if not np.isfinite(candidate.vertices).all():
        return False
    original_extent = np.ptp(original.vertices, axis=0)
    candidate_extent = np.ptp(candidate.vertices, axis=0)
    original_diag = float(np.linalg.norm(original_extent))
    if original_diag <= 0:
        return True
    for before, after in zip(original_extent, candidate_extent):
        if before > original_diag * 0.03 and after < before * 0.6:
            return False
    original_triangles = original.vertices[original.faces]
    candidate_triangles = candidate.vertices[candidate.faces]
    original_max_edge = float(
        np.linalg.norm(
            np.roll(original_triangles, 1, axis=1) - original_triangles, axis=2
        ).max()
    )
    candidate_max_edge = float(
        np.linalg.norm(
            np.roll(candidate_triangles, 1, axis=1) - candidate_triangles, axis=2
        ).max()
    )
    if candidate_max_edge > max(original_max_edge * 1.35, original_diag * 0.55):
        return False
    original_min = original.vertices.min(axis=0)
    original_max = original.vertices.max(axis=0)
    candidate_min = candidate.vertices.min(axis=0)
    candidate_max = candidate.vertices.max(axis=0)
    margin = original_diag * 0.02
    if (candidate_min < original_min - margin).any() or (
        candidate_max > original_max + margin
    ).any():
        return False
    if float(np.linalg.norm(candidate.vertices.mean(axis=0) - original.vertices.mean(axis=0))) > original_diag * 0.12:
        return False
    return True


def decimate_parts(
    parts: list[MeshPart], max_faces: int
) -> tuple[list[MeshPart], dict[str, object]]:
    total = sum(len(part.faces) for part in parts)
    if total <= max_faces or max_faces <= 0:
        return parts, {
            "source_faces": total,
            "decimated_faces": total,
            "methods": {},
        }
    targets: list[int] = []
    for part in parts:
        allocated = int(round(max_faces * len(part.faces) / total))
        targets.append(max(60, min(len(part.faces), allocated)))

    output: list[MeshPart] = []
    methods: dict[str, int] = defaultdict(int)
    deviations = []
    for part, target in zip(parts, targets):
        new_part, method = decimate_part(part, target)
        output.append(new_part)
        methods[method] += 1
        if len(part.faces) >= 5000 and len(new_part.faces) < len(part.faces):
            deviation = sampled_deviation_mm(part, new_part, samples=120)
            if deviation is not None:
                deviations.append(
                    {
                        "part": part.name,
                        "source_faces": len(part.faces),
                        "decimated_faces": len(new_part.faces),
                        "mean_mm": round(deviation[0], 4),
                        "p95_mm": round(deviation[1], 4),
                        "max_mm": round(deviation[2], 4),
                    }
                )
    return output, {
        "source_faces": total,
        "decimated_faces": sum(len(part.faces) for part in output),
        "methods": dict(methods),
        "deviations": deviations,
    }


def sampled_deviation_mm(
    original: MeshPart, simplified: MeshPart, samples: int = 300
) -> tuple[float, float, float] | None:
    if len(original.vertices) == 0 or len(simplified.faces) == 0:
        return None
    rng = np.random.default_rng(12345)
    count = min(samples, len(original.vertices))
    indices = rng.choice(len(original.vertices), count, replace=False)
    query = original.vertices[indices] * 1000.0
    target_mesh = trimesh.Trimesh(
        simplified.vertices * 1000.0, simplified.faces, process=False
    )
    try:
        _, distances, _ = trimesh.proximity.closest_point(target_mesh, query)
    except Exception:
        return None
    distances = np.asarray(distances, dtype=float)
    return (
        float(distances.mean()),
        float(np.percentile(distances, 95)),
        float(distances.max()),
    )


# ---------------------------------------------------------------------------
# OpenCASCADE mesh -> B-rep
# ---------------------------------------------------------------------------


from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Builder
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_Sewing,
)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.STEPCAFControl import STEPCAFControl_Reader, STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDF import TDF_Label
from OCP.TDocStd import TDocStd_Document
from OCP.TopAbs import TopAbs_COMPOUND, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Compound
from OCP.XCAFDoc import (
    XCAFDoc_ColorSurf,
    XCAFDoc_DocumentTool,
)
from OCP.collections import Sequence_TDF_Label
from OCP.gp import gp_Pnt, gp_Trsf


def _iter_subshapes(shape, shape_type) -> Iterable:
    explorer = TopExp_Explorer(shape, shape_type)
    while explorer.More():
        yield explorer.Current()
        explorer.Next()


def _count_subshapes(shape, shape_type) -> int:
    return sum(1 for _ in _iter_subshapes(shape, shape_type))


def _bbox_mm(shape) -> tuple[float, float, float, float, float, float]:
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    minimum = box.CornerMin()
    maximum = box.CornerMax()
    return (
        float(minimum.X()),
        float(minimum.Y()),
        float(minimum.Z()),
        float(maximum.X()),
        float(maximum.Y()),
        float(maximum.Z()),
    )


def _promote_closed_shells(shape):
    shells = [TopoDS.Shell(sub) for sub in _iter_subshapes(shape, TopAbs_SHELL)]
    if not shells or not all(shell.Closed() for shell in shells):
        return shape
    try:
        if len(shells) == 1:
            solid = BRepBuilderAPI_MakeSolid(shells[0]).Solid()
            if BRepCheck_Analyzer(solid).IsValid():
                return solid
            return shape
        compound = TopoDS_Compound()
        builder = BRep_Builder()
        builder.MakeCompound(compound)
        for shell in shells:
            solid = BRepBuilderAPI_MakeSolid(shell).Solid()
            builder.Add(compound, solid)
        if BRepCheck_Analyzer(compound).IsValid():
            return compound
    except Exception:
        pass
    return shape


def mesh_to_brep(
    part: MeshPart, sew_tolerance_mm: float = 0.01, unify: bool = True
):
    """Build a B-rep shell/solid from a decimated triangle mesh in millimetres."""
    vertices = np.asarray(part.vertices, dtype=float) * 1000.0
    faces = np.asarray(part.faces, dtype=np.int64)
    if len(faces) == 0:
        return None

    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]
    areas = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1) * 0.5
    valid = areas > 1e-10
    faces = faces[valid]
    if len(faces) == 0:
        return None

    brep_faces = []
    for triangle in faces:
        polygon = BRepBuilderAPI_MakePolygon()
        for vertex_index in triangle:
            point = vertices[vertex_index]
            polygon.Add(gp_Pnt(float(point[0]), float(point[1]), float(point[2])))
        polygon.Close()
        if not polygon.IsDone():
            continue
        face_maker = BRepBuilderAPI_MakeFace(polygon.Wire())
        if face_maker.IsDone():
            brep_faces.append(face_maker.Face())
    if not brep_faces:
        return None

    sewing = BRepBuilderAPI_Sewing(float(sew_tolerance_mm))
    for face in brep_faces:
        sewing.Add(face)
    sewing.Perform()
    shape = sewing.SewedShape()
    if shape.IsNull():
        return None

    if unify:
        try:
            unifier = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
            unifier.Build()
            shape = unifier.Shape()
        except Exception:
            pass

    shape = _promote_closed_shells(shape)
    return shape


def _gp_trsf(transform_mm: np.ndarray) -> gp_Trsf:
    trsf = gp_Trsf()
    trsf.SetValues(
        float(transform_mm[0, 0]),
        float(transform_mm[0, 1]),
        float(transform_mm[0, 2]),
        float(transform_mm[0, 3]),
        float(transform_mm[1, 0]),
        float(transform_mm[1, 1]),
        float(transform_mm[1, 2]),
        float(transform_mm[1, 3]),
        float(transform_mm[2, 0]),
        float(transform_mm[2, 1]),
        float(transform_mm[2, 2]),
        float(transform_mm[2, 3]),
    )
    return trsf


def _set_name(label: TDF_Label, name: str) -> None:
    TDataStd_Name.Set_s(label, TCollection_ExtendedString(name))


def build_step_document(
    model: UrdfModel,
    link_parts: dict[str, list[MeshPart]],
    sew_tolerance_mm: float,
    unify: bool,
):
    document = TDocStd_Document(TCollection_ExtendedString("urdf-to-step"))
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(document.Main())

    robot_label = shape_tool.NewShape()
    _set_name(robot_label, sanitize_name(model.robot_name, "robot"))

    link_labels: dict[str, TDF_Label] = {}
    stats: dict[str, object] = {}

    for link_name in model.links:
        link_label = shape_tool.NewShape()
        _set_name(link_label, sanitize_name(link_name, "link"))
        link_labels[link_name] = link_label
        part_stats = []
        for part in link_parts.get(link_name, []):
            shape = mesh_to_brep(part, sew_tolerance_mm, unify)
            if shape is None:
                part_stats.append(
                    {"name": part.name, "status": "no_brep", "faces": 0}
                )
                continue
            component_label = shape_tool.AddComponent(link_label, shape, False)
            _set_name(component_label, part.name)
            try:
                color_tool.SetColor(
                    component_label,
                    Quantity_Color(
                        float(part.color[0]),
                        float(part.color[1]),
                        float(part.color[2]),
                        Quantity_TOC_RGB,
                    ),
                    XCAFDoc_ColorSurf,
                )
            except Exception:
                pass
            part_stats.append(
                {
                    "name": part.name,
                    "status": "ok",
                    "faces": _count_subshapes(shape, TopAbs_FACE),
                    "solids": _count_subshapes(shape, TopAbs_SOLID),
                    "valid": bool(BRepCheck_Analyzer(shape).IsValid()),
                }
            )
        stats[link_name] = {
            "parts": part_stats,
            "faces": sum(int(p["faces"]) for p in part_stats),
            "solids": sum(int(p.get("solids", 0)) for p in part_stats),
        }

    # Nest the links according to the URDF joint tree.  Only the joint origin is
    # applied because the export is the zero-joint configuration.
    for joint in model.joints:
        parent_label = link_labels[joint.parent]
        child_label = link_labels[joint.child]
        transform_mm = translate_mm(joint.origin)
        shape_tool.AddComponent(
            parent_label, child_label, TopLoc_Location(_gp_trsf(transform_mm))
        )

    shape_tool.AddComponent(
        robot_label, link_labels[model.root], TopLoc_Location()
    )
    shape_tool.UpdateAssemblies()
    return document, stats


def write_step(document, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        Interface_Static.SetCVal_s("write.step.unit", "MM")
    except Exception:
        pass
    writer = STEPCAFControl_Writer()
    writer.SetColorMode(True)
    writer.SetNameMode(True)
    writer.SetLayerMode(True)
    if not writer.Transfer(document, STEPControl_AsIs):
        raise RuntimeError("STEPCAFControl_Writer.Transfer failed")
    status = writer.Write(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEPCAFControl_Writer.Write returned {status}")


def inspect_step(path: Path) -> dict[str, object]:
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"cannot read back {path}: {status}")
    reader.TransferRoots()
    shape = reader.OneShape()
    bbox = _bbox_mm(shape)
    return {
        "bbox_mm": bbox,
        "size_mm": (
            round(bbox[3] - bbox[0], 4),
            round(bbox[4] - bbox[1], 4),
            round(bbox[5] - bbox[2], 4),
        ),
        "faces": _count_subshapes(shape, TopAbs_FACE),
        "shells": _count_subshapes(shape, TopAbs_SHELL),
        "solids": _count_subshapes(shape, TopAbs_SOLID),
        "compounds": _count_subshapes(shape, TopAbs_COMPOUND),
        "valid": bool(BRepCheck_Analyzer(shape).IsValid()),
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
    }


def expected_bbox_mm(
    model: UrdfModel, link_parts: dict[str, list[MeshPart]]
) -> tuple[float, float, float, float, float, float] | None:
    world = model.world_transforms()
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    found = False
    for link_name, parts in link_parts.items():
        transform = translate_mm(world[link_name])
        for part in parts:
            if len(part.vertices) == 0:
                continue
            vertices = apply_transform(transform, part.vertices * 1000.0)
            minimum = np.minimum(minimum, vertices.min(axis=0))
            maximum = np.maximum(maximum, vertices.max(axis=0))
            found = True
    if not found:
        return None
    return (
        float(minimum[0]),
        float(minimum[1]),
        float(minimum[2]),
        float(maximum[0]),
        float(maximum[1]),
        float(maximum[2]),
    )


# ---------------------------------------------------------------------------
# Preview rendering
# ---------------------------------------------------------------------------


def render_preview(
    model: UrdfModel,
    link_parts: dict[str, list[MeshPart]],
    output: Path,
    max_faces_per_link: int = 0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    world = model.world_transforms()
    palette = plt.get_cmap("tab20")
    triangles: list[np.ndarray] = []
    colors: list[tuple[float, float, float, float]] = []
    for index, (link_name, parts) in enumerate(link_parts.items()):
        if not parts:
            continue
        transform = translate_mm(world[link_name])
        vertices = []
        faces = []
        offset = 0
        for part in parts:
            if len(part.faces) == 0:
                continue
            vertices.append(apply_transform(transform, part.vertices * 1000.0))
            faces.append(part.faces + offset)
            offset += len(part.vertices)
        if not vertices:
            continue
        merged = trimesh.Trimesh(
            np.vstack(vertices), np.vstack(faces), process=False
        )
        if max_faces_per_link and len(merged.faces) > max_faces_per_link:
            simplified = _decimate_with_fast(
                merged.vertices, merged.faces, max_faces_per_link
            )
            if simplified is not None:
                merged = trimesh.Trimesh(*simplified, process=False)
        color = palette(index % 20)
        for triangle in merged.vertices[merged.faces]:
            triangles.append(triangle)
            colors.append(color)

    if not triangles:
        return
    all_points = np.concatenate([tri.reshape(-1, 3) for tri in triangles], axis=0)
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    center = (lower + upper) * 0.5
    radius = float(np.max(upper - lower)) * 0.55

    views = [
        ("isometric", 24, -55),
        ("front", 0, -90),
        ("side", 0, 0),
    ]
    figure = plt.figure(figsize=(16, 12), dpi=130)
    for view_index, (title, elevation, azimuth) in enumerate(views, start=1):
        axis = figure.add_subplot(2, 2, view_index, projection="3d")
        collection = Poly3DCollection(
            triangles, facecolors=colors, edgecolors="none", linewidths=0.0
        )
        axis.add_collection3d(collection)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_title(f"{title}  (X forward, Z up)")
        axis.set_xlabel("X [mm]")
        axis.set_ylabel("Y [mm]")
        axis.set_zlabel("Z [mm]")
        axis.grid(True, alpha=0.25)
    figure.suptitle(model.robot_name, fontsize=16)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _parse_package_roots(values: Sequence[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--package-root expects NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        roots[name] = Path(path)
    return roots


def _parse_link_budgets(values: Sequence[str]) -> dict[str, int]:
    budgets: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--link-max-faces expects LINK=FACES, got {value!r}")
        link, faces = value.split("=", 1)
        budgets[link] = int(faces)
    return budgets


def convert(args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    urdf_path = Path(args.urdf)
    output = Path(args.output)
    package_roots = _parse_package_roots(args.package_root)
    link_budgets = _parse_link_budgets(args.link_max_faces)
    model = parse_urdf(urdf_path, package_roots)

    print(f"[urdf] {urdf_path}")
    print(f"[robot] {model.robot_name}: {len(model.links)} links, {len(model.joints)} joints, root={model.root}")
    if model.package_roots:
        print(f"[packages] {', '.join(f'{k}={v}' for k, v in sorted(model.package_roots.items()))}")

    link_parts: dict[str, list[MeshPart]] = {}
    source_faces = 0
    for link_name in model.links:
        parts = load_link_parts(model, link_name)
        link_parts[link_name] = parts
        source_faces += sum(len(part.faces) for part in parts)
        if parts:
            print(f"  {link_name:16s} {len(parts):3d} part(s) {sum(len(p.faces) for p in parts):8d} tris")

    source_bbox = expected_bbox_mm(model, link_parts)
    total_before = sum(len(part.faces) for parts in link_parts.values() for part in parts)
    total_after = 0
    decimation_records = []
    for link_name, parts in link_parts.items():
        budget = link_budgets.get(link_name, args.max_faces_per_link)
        decimated, info = decimate_parts(parts, budget)
        link_parts[link_name] = decimated
        total_after += int(info["decimated_faces"])
        if info["source_faces"] != info["decimated_faces"]:
            decimation_records.append({"link": link_name, "budget": budget, **info})

    print(f"[decimate] {total_before} -> {total_after} triangles (max {args.max_faces_per_link}/link)")

    document, link_stats = build_step_document(
        model, link_parts, args.sew_tolerance_mm, unify=not args.no_unify
    )
    write_step(document, output)
    print(f"[step] wrote {output}")

    inspection = inspect_step(output)
    expected = expected_bbox_mm(model, link_parts)
    reference = source_bbox or expected
    print(
        f"[verify] size={inspection['size_mm']} mm faces={inspection['faces']} "
        f"solids={inspection['solids']} valid={inspection['valid']} "
        f"file={inspection['file_size_mb']} MB"
    )

    preview = None
    if args.preview:
        preview = Path(args.preview)
        render_preview(model, link_parts, preview)
        print(f"[preview] {preview}")

    record = {
        "robot": model.robot_name,
        "urdf": str(urdf_path),
        "step_path": str(output),
        "root_link": model.root,
        "links": len(model.links),
        "joints": len(model.joints),
        "units": "millimetres",
        "config": "zero joint positions",
        "source_faces": total_before,
        "decimated_faces": total_after,
        "max_faces_per_link": args.max_faces_per_link,
        "link_max_faces": link_budgets,
        "decimation": decimation_records,
        "step": {
            **inspection,
            "source_bbox_mm": source_bbox,
            "decimated_bbox_mm": expected,
            "bbox_delta_mm": (
                tuple(
                    round(float(inspection["bbox_mm"][i] - reference[i]), 4)
                    for i in range(6)
                )
                if reference
                else None
            ),
        },
        "link_stats": link_stats,
        "elapsed_s": round(time.time() - started, 2),
        "preview": str(preview) if preview else None,
        "warnings": [],
    }
    if record["step"]["solids"] == 0:
        record["warnings"].append(
            "No closed solids were produced; the STEP contains sewn shells "
            "(expected for open or non-manifold source meshes)."
        )
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", required=True, help="input URDF")
    parser.add_argument("--output", required=True, help="output STEP file")
    parser.add_argument(
        "--package-root",
        action="append",
        default=[],
        help="package:// name=/path/to/package (repeatable)",
    )
    parser.add_argument(
        "--max-faces-per-link",
        type=int,
        default=4000,
        help="triangle budget per link after decimation (0 disables decimation)",
    )
    parser.add_argument(
        "--link-max-faces",
        action="append",
        default=[],
        help="per-link triangle budget, e.g. Link3=12000 (repeatable)",
    )
    parser.add_argument(
        "--sew-tolerance-mm",
        type=float,
        default=0.01,
        help="OpenCASCADE sewing tolerance in millimetres",
    )
    parser.add_argument(
        "--no-unify",
        action="store_true",
        help="skip ShapeUpgrade_UnifySameDomain",
    )
    parser.add_argument("--preview", help="optional PNG preview path")
    parser.add_argument(
        "--report",
        help="optional JSON report path (one record, or appended list)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        record = convert(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if report_path.exists():
            try:
                existing = json.loads(report_path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = [existing]
            except Exception:
                existing = []
        existing.append(record)
        report_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
