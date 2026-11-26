"""Compose a single AS2-W X + Piper X URDF.

The AS2-W base keeps its original leg/sensor tree. The Piper X arm is
renamed to the same subtree convention as ``M20_Piper_own.urdf``:

    arm_base_link -> arm_link1 ... arm_link6 -> gripper_base
                                           -> gripper_link1/2

Mount rationale
---------------
AS2-W's ``base_link`` has one continuous flat top deck: z = 0.076 for
x in [-0.12, +0.06] and y in [-0.06, +0.06].  Outside that patch the
surface drops away (side edges at |y| > 0.06) or climbs (rear bevel
x < -0.12, front sensor hood x > 0.065, which rises to z = 0.135).
Measured with downward ray casts on a 15 mm grid.

The arm therefore sits at the middle of the wheelbase (hip joints at
x = +-0.2119) rather than on the rear deck.  Piper X works towards +x, so
a centred base keeps its whole workspace in front of the robot, puts the
1.56 kg arm over the wheelbase centre instead of cantilevered behind the
rear axle, and halves the rear overhang of the folded elbow.

Piper X at its zero pose folds the elbow towards -x and points the gripper
towards +x (gripper base at x = -0.007, fingers at x = +0.127 in this
model).  Keeping ``rpy=(0, 0, 0)`` therefore keeps the hand facing the
same direction as the robot's sensors, which is what the low-shelf task
needs (the robot approaches the shelf front-first and pitches down).  A
180 deg yaw would put the hand and the folded elbow behind the robot.

The default ``base_to_arm`` offset places the 0.0985 m x 0.08 m plate at
x in [-0.0585, 0.04]: fully on the flat deck, 20 mm clear of the point
where the front hood starts to rise (x about 0.06), and 3.8 mm above the
deck.
"""

from __future__ import annotations

import argparse
import copy
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]

PIPERX_LINK_RENAME = {
    "base_link": "arm_base_link",
    "Link1": "arm_link1",
    "Link2": "arm_link2",
    "Link3": "arm_link3",
    "Link4": "arm_link4",
    "Link5": "arm_link5",
    "Link6": "arm_link6",
    "Link7": "gripper_link1",
    "Link8": "gripper_link2",
}

PIPERX_JOINT_RENAME = {
    "joint1": "arm_joint1",
    "joint2": "arm_joint2",
    "joint3": "arm_joint3",
    "joint4": "arm_joint4",
    "joint5": "arm_joint5",
    "joint6": "arm_joint6",
    "joint7": "gripper_joint1",
    "joint8": "gripper_joint2",
}

# New link name -> (source relative to Piper X package, output file name).
PIPERX_MESH_SOURCES = {
    "arm_base_link": ("meshes/base_link.STL", "arm_base_link.STL"),
    "arm_link1": ("meshes/Link1.STL", "link1.STL"),
    "arm_link2": ("meshes/Link2.STL", "link2.STL"),
    "arm_link3": ("meshes/Link3.STL", "link3.STL"),
    "arm_link4": ("meshes/Link4.STL", "link4.STL"),
    "arm_link5": ("meshes/Link5.STL", "link5.STL"),
    "gripper_base": ("meshes/dae/gripper_base_100_v2.dae", "gripper_base.STL"),
    "gripper_link1": ("meshes/dae/J7.dae", "gripper_link1.STL"),
    "gripper_link2": ("meshes/dae/J8.dae", "gripper_link2.STL"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--as2w-urdf", type=Path, required=True, help="Path to the official as2w.urdf")
    parser.add_argument("--piperx-urdf", type=Path, required=True, help="Path to piper_x_description.urdf")
    parser.add_argument("--as2w-mesh-root", type=Path, default=None, help="Directory containing as2w meshes/")
    parser.add_argument("--piperx-mesh-root", type=Path, default=None, help="Piper X package root")
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "deep_robotics_model" / "AS2W_PiperX" / "urdf"
    )
    parser.add_argument("--mount-x", type=float, default=0.0)
    parser.add_argument("--mount-y", type=float, default=0.0)
    parser.add_argument("--mount-z", type=float, default=0.0798)
    parser.add_argument("--mount-yaw", type=float, default=0.0)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output directory")
    args = parser.parse_args()
    if args.as2w_mesh_root is None:
        args.as2w_mesh_root = args.as2w_urdf.parent
    if args.piperx_mesh_root is None:
        args.piperx_mesh_root = args.piperx_urdf.parent.parent
    return args


def format_float(value: float) -> str:
    text = f"{value:.9f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def mesh_elements(link: ET.Element) -> list[ET.Element]:
    meshes: list[ET.Element] = []
    for tag in ("visual", "collision"):
        for element in link.findall(tag):
            geometry = element.find("geometry")
            mesh = None if geometry is None else geometry.find("mesh")
            if mesh is not None:
                meshes.append(mesh)
    return meshes


def copy_asset(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in (".dae", ".obj", ".ply"):
        trimesh.load(source, force="mesh").export(destination)
    else:
        shutil.copy2(source, destination)


def add_small_inertial(link: ET.Element, mass: float = 0.001) -> None:
    inertial = ET.Element("inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": format_float(mass)})
    ET.SubElement(
        inertial,
        "inertia",
        {"ixx": "1e-7", "ixy": "0", "ixz": "0", "iyy": "1e-7", "iyz": "0", "izz": "1e-7"},
    )
    link.insert(0, inertial)


def fix_piperx_finger_mass(link: ET.Element) -> None:
    """Piper X leaves Link7 with zero mass; give the finger a small physical mass."""
    inertial = link.find("inertial")
    if inertial is None:
        add_small_inertial(link, mass=0.01)
        return
    mass = inertial.find("mass")
    if mass is None:
        ET.SubElement(inertial, "mass", {"value": "0.01"})
    elif float(mass.get("value", "0")) == 0.0:
        mass.set("value", "0.01")
    inertia = inertial.find("inertia")
    if inertia is None:
        inertia = ET.SubElement(inertial, "inertia")
    inertia.attrib.update(
        {"ixx": "1e-5", "ixy": "0", "ixz": "0", "iyy": "6e-6", "iyz": "0", "izz": "1.4e-5"}
    )


def rewrite_link_meshes(link: ET.Element, output_name: str) -> None:
    for mesh in mesh_elements(link):
        mesh.set("filename", f"./meshes/{output_name}")


def build(args: argparse.Namespace) -> Path:
    as2w_root = ET.parse(args.as2w_urdf).getroot()
    piperx_root = ET.parse(args.piperx_urdf).getroot()

    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"{out_dir} exists; pass --force to overwrite")
        shutil.rmtree(out_dir)
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True)

    # --- AS2-W: copy meshes and rewrite mesh paths ---------------------
    for link in as2w_root.findall("link"):
        for mesh in mesh_elements(link):
            source = args.as2w_mesh_root / Path(mesh.get("filename"))
            if not source.is_file():
                raise FileNotFoundError(f"AS2-W mesh not found: {source}")
            copy_asset(source, mesh_dir / source.name)
            mesh.set("filename", f"./meshes/{source.name}")

    # --- Piper X: rename links/joints and rewrite mesh paths -----------
    piperx_links = {link.get("name"): link for link in piperx_root.findall("link")}
    for old_name, new_name in PIPERX_LINK_RENAME.items():
        link = piperx_links[old_name]
        link.set("name", new_name)
        if new_name in PIPERX_MESH_SOURCES:
            source_relative, output_name = PIPERX_MESH_SOURCES[new_name]
            source = args.piperx_mesh_root / source_relative
            if not source.is_file():
                raise FileNotFoundError(f"Piper X mesh not found: {source}")
            copy_asset(source, mesh_dir / output_name)
            rewrite_link_meshes(link, output_name)

    # Piper X puts the gripper-base mesh on Link6.  Move that visual and
    # collision to a dedicated gripper_base link so the EE body name matches
    # M20_Piper_own and the task code's ``gripper_base`` convention.
    arm_link6 = piperx_links["Link6"]
    gripper_base = ET.Element("link", {"name": "gripper_base"})
    add_small_inertial(gripper_base)
    gripper_base_source, gripper_base_output = PIPERX_MESH_SOURCES["gripper_base"]
    copy_asset(args.piperx_mesh_root / gripper_base_source, mesh_dir / gripper_base_output)
    for tag in ("visual", "collision"):
        for element in list(arm_link6.findall(tag)):
            arm_link6.remove(element)
            gripper_base.append(element)
    rewrite_link_meshes(gripper_base, gripper_base_output)

    fix_piperx_finger_mass(piperx_links["Link7"])

    for old_name, new_name in PIPERX_JOINT_RENAME.items():
        joint = next(joint for joint in piperx_root.findall("joint") if joint.get("name") == old_name)
        joint.set("name", new_name)
        for tag in ("parent", "child"):
            child = joint.find(tag)
            link_name = child.get("link")
            if link_name in PIPERX_LINK_RENAME:
                child.set("link", PIPERX_LINK_RENAME[link_name])
    piperx_joints = {joint.get("name"): joint for joint in piperx_root.findall("joint")}
    piperx_joints["gripper_joint1"].find("parent").set("link", "gripper_base")
    piperx_joints["gripper_joint2"].find("parent").set("link", "gripper_base")
    renamed_links = {link.get("name"): link for link in piperx_root.findall("link")}

    # --- assemble the combined tree -----------------------------------
    combined = ET.Element("robot", {"name": "AS2W_PiperX"})
    combined.append(
        ET.Comment(
            " AS2-W X + Piper X. base_to_arm xyz=("
            f"{format_float(args.mount_x)}, {format_float(args.mount_y)}, {format_float(args.mount_z)}"
            f"), rpy=(0, 0, {format_float(args.mount_yaw)}). Piper X +x faces robot +x. "
        )
    )
    for element in as2w_root:
        if element.tag in ("link", "joint"):
            combined.append(copy.deepcopy(element))

    base_to_arm = ET.SubElement(combined, "joint", {"name": "base_to_arm", "type": "fixed"})
    ET.SubElement(base_to_arm, "parent", {"link": "base_link"})
    ET.SubElement(base_to_arm, "child", {"link": "arm_base_link"})
    ET.SubElement(
        base_to_arm,
        "origin",
        {
            "xyz": " ".join(format_float(v) for v in (args.mount_x, args.mount_y, args.mount_z)),
            "rpy": f"0 0 {format_float(args.mount_yaw)}",
        },
    )

    combined.append(copy.deepcopy(renamed_links["arm_base_link"]))
    for index in range(1, 6):
        combined.append(copy.deepcopy(piperx_joints[f"arm_joint{index}"]))
        combined.append(copy.deepcopy(renamed_links[f"arm_link{index}"]))
    combined.append(copy.deepcopy(piperx_joints["arm_joint6"]))
    combined.append(copy.deepcopy(renamed_links["arm_link6"]))

    joint6_to_gripper_base = ET.Element("joint", {"name": "joint6_to_gripper_base", "type": "fixed"})
    ET.SubElement(joint6_to_gripper_base, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(joint6_to_gripper_base, "parent", {"link": "arm_link6"})
    ET.SubElement(joint6_to_gripper_base, "child", {"link": "gripper_base"})
    combined.append(joint6_to_gripper_base)
    combined.append(copy.deepcopy(gripper_base))
    combined.append(copy.deepcopy(piperx_joints["gripper_joint1"]))
    combined.append(copy.deepcopy(renamed_links["gripper_link1"]))
    combined.append(copy.deepcopy(piperx_joints["gripper_joint2"]))
    combined.append(copy.deepcopy(renamed_links["gripper_link2"]))

    output = out_dir / "AS2W_PiperX.urdf"
    tree = ET.ElementTree(combined)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


def validate(urdf: Path) -> None:
    root = ET.parse(urdf).getroot()
    link_elements = root.findall("link")
    links = {link.get("name") for link in link_elements}
    joints = root.findall("joint")
    if len(links) != len(link_elements):
        raise ValueError("duplicate link names")
    if len(joints) != len({joint.get("name") for joint in joints}):
        raise ValueError("duplicate joint names")
    parents: dict[str, str] = {}
    for joint in joints:
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        if parent not in links or child not in links:
            raise ValueError(f"joint {joint.get('name')} references an unknown link")
        if child in parents:
            raise ValueError(f"link {child} has more than one parent")
        parents[child] = parent
    roots = links - set(parents)
    if roots != {"base_link"}:
        raise ValueError(f"expected base_link as the only root, got {sorted(roots)}")
    mesh_count = 0
    for mesh in root.iter("mesh"):
        path = urdf.parent / mesh.get("filename")
        if not path.is_file():
            raise FileNotFoundError(f"missing mesh {path}")
        mesh_count += 1
    print(f"validated {urdf}: {len(links)} links, {len(joints)} joints, {mesh_count} meshes")


def main() -> None:
    args = parse_args()
    output = build(args)
    validate(output)
    print(output)


if __name__ == "__main__":
    main()
