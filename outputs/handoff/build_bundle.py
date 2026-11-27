# -*- coding: utf-8 -*-
"""Build a self-contained URDF bundle for AS2-W / Piper X / AS2W_PiperX."""

import os
import re
import shutil

REPO = r"D:\nvidia-isaac-sim\loco-manip-unified-rl-agent"
PIPER_ROOT = r"D:\nvidia-isaac-sim\own_assets\piper_isaac_sim"
OUT = os.path.join(REPO, r"outputs\handoff\AS2W_PiperX_urdf_bundle")
ZIP_BASE = os.path.join(REPO, r"outputs\handoff\AS2W_PiperX_urdf_bundle")
SRC_AS2W = os.path.join(REPO, r"outputs\handoff\_src\as2w_description")
ASM_DIR = os.path.join(REPO, r"deep_robotics_model\AS2W_PiperX")
README_TEMPLATE = os.path.join(REPO, r"outputs\handoff\README_bundle.md")
D405_SNIPPET = os.path.join(REPO, r"outputs\handoff\d405_camera_snippet.urdf")

if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT)


def cp(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def copytree_files(src, dst):
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        p = os.path.join(src, name)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(dst, name))


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_minimal_package(pkg_dir, package_name, version, description, license_name):
    pkg_xml = """<?xml version="1.0"?>
<package format="3">
  <name>{name}</name>
  <version>{version}</version>
  <description>{description}</description>
  <maintainer email="noreply@example.com">URDF bundle maintainer</maintainer>
  <license>{license_name}</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
""".format(name=package_name, version=version, description=description,
           license_name=license_name)
    cmake = """cmake_minimum_required(VERSION 3.5)
project({name})
find_package(ament_cmake REQUIRED)
install(DIRECTORY meshes DESTINATION share/${{PROJECT_NAME}})
ament_package()
""".format(name=package_name)
    write_text(os.path.join(pkg_dir, "package.xml"), pkg_xml)
    write_text(os.path.join(pkg_dir, "CMakeLists.txt"), cmake)


def make_d405_urdf(base_urdf_path, out_path):
    text = open(base_urdf_path, encoding="utf-8").read()
    text = re.sub(r"<robot\b[^>]*>", '<robot name="piper_x_description">', text,
                  count=1, flags=re.S)
    snippet = open(D405_SNIPPET, encoding="utf-8").read()
    marker = "</robot>"
    if marker not in text:
        raise RuntimeError("invalid base URDF: missing </robot>")
    text = text.replace(marker, snippet + marker, 1)
    write_text(out_path, text)


# 1) AS2-W (official unitree_ros package, meshes byte-identical to the repo copy)
as2w = os.path.join(OUT, "as2w_description")
cp(os.path.join(SRC_AS2W, "as2w.urdf"), os.path.join(as2w, "as2w.urdf"))
copytree_files(os.path.join(SRC_AS2W, "meshes"), os.path.join(as2w, "meshes"))

# 2) Piper X package, with the D405 camera variant
piper = os.path.join(OUT, "piper_x_description")
for name in ("package.xml", "CMakeLists.txt"):
    cp(os.path.join(PIPER_ROOT, "piper_x_description", name), os.path.join(piper, name))
for sub in ("config", "launch", "rviz"):
    s = os.path.join(PIPER_ROOT, "piper_x_description", sub)
    if os.path.isdir(s):
        copytree_files(s, os.path.join(piper, sub))
for name in ("piper_x_description.urdf", "PIPER-XURDF.csv"):
    s = os.path.join(PIPER_ROOT, "piper_x_description", "urdf", name)
    if os.path.isfile(s):
        cp(s, os.path.join(piper, "urdf", name))
copytree_files(os.path.join(PIPER_ROOT, "piper_x_description", "meshes"),
               os.path.join(piper, "meshes"))
copytree_files(os.path.join(PIPER_ROOT, "piper_x_description", "meshes", "dae"),
               os.path.join(piper, "meshes", "dae"))

# The upstream xacro entry point hard-codes a camera include that is not used by
# the plain URDF variants shipped here.
xacro_launch = os.path.join(piper, "launch", "display_xacro.launch.py")
if os.path.isfile(xacro_launch):
    os.remove(xacro_launch)

# D405 external dependencies
cp(os.path.join(PIPER_ROOT, "realsense2_description", "meshes", "d405.stl"),
   os.path.join(OUT, "realsense2_description", "meshes", "d405.stl"))
cp(os.path.join(PIPER_ROOT, "piper_description", "meshes", "dae", "realsense_mid_stand.dae"),
   os.path.join(OUT, "piper_description", "meshes", "dae", "realsense_mid_stand.dae"))
write_minimal_package(
    os.path.join(OUT, "realsense2_description"), "realsense2_description", "4.56.4",
    "Intel RealSense D405 mesh subset used by the Piper X URDF.", "Apache License 2.0")
write_minimal_package(
    os.path.join(OUT, "piper_description"), "piper_description", "0.0.0",
    "Piper camera bracket mesh subset used by the Piper X D405 URDF.", "Apache License 2.0")

make_d405_urdf(os.path.join(piper, "urdf", "piper_x_description.urdf"),
               os.path.join(piper, "urdf", "piper_x_description_d405.urdf"))

# Point the ROS 2 display launch at the D405 variant.
p = os.path.join(piper, "launch", "display_urdf.launch.py")
t = open(p, encoding="utf-8").read()
t = t.replace("piper_x_description_d435.urdf", "piper_x_description_d405.urdf")
write_text(p, t)

# 3) Assembled robot
asm = os.path.join(OUT, "as2w_piperx")
cp(os.path.join(ASM_DIR, "urdf", "AS2W_PiperX.urdf"), os.path.join(asm, "urdf", "AS2W_PiperX.urdf"))
copytree_files(os.path.join(ASM_DIR, "urdf", "meshes"), os.path.join(asm, "urdf", "meshes"))
cp(os.path.join(ASM_DIR, "README.md"), os.path.join(asm, "README.md"))

# 4) previews
prev = os.path.join(OUT, "previews")
os.makedirs(prev, exist_ok=True)
for name in ("as2w.png", "piper_x.png", "as2w_piperx.png", "piper_x_d405.png"):
    s = os.path.join(REPO, "outputs", "model_previews", "final", name)
    if os.path.isfile(s):
        cp(s, os.path.join(prev, name))

# README is kept as a separate template so rebuilding the bundle preserves it.
if os.path.isfile(README_TEMPLATE):
    cp(README_TEMPLATE, os.path.join(OUT, "README.md"))

# ---- fixes applied to the copies ----
fix_log = []

# 4a) wrong case: base_Link.dae vs base_link.dae
for rel in (r"piper_x_description\urdf\piper_x_description.urdf",
            r"piper_x_description\urdf\piper_x_description_d405.urdf"):
    p = os.path.join(OUT, rel)
    t = open(p, encoding="utf-8").read()
    n = t.count("base_Link.dae")
    if n:
        t = t.replace("base_Link.dae", "base_link.dae")
        write_text(p, t)
        fix_log.append("%s: base_Link.dae -> base_link.dae (%d)" % (rel, n))

# 4b) Rhino DAE unit metadata says millimeters but the coords are meters
p = os.path.join(OUT, r"piper_x_description\meshes\dae\base_link.dae")
t = open(p, encoding="utf-8").read()
if '<unit meter="0.001" name="Millimeters" />' in t:
    t = t.replace('<unit meter="0.001" name="Millimeters" />', '<unit meter="1" name="Meters" />')
    write_text(p, t)
    fix_log.append("meshes/dae/base_link.dae: unit meter 0.001 -> 1")

# ---- validate mesh references (case-sensitive) ----
def check_urdf(urdf_path, package_roots):
    text = open(urdf_path, encoding="utf-8").read()
    refs = re.findall(r'<mesh\s+filename="([^"]+)"', text)
    base = os.path.dirname(urdf_path)
    missing = []
    ok = 0
    for ref in refs:
        if ref.startswith("package://"):
            rest = ref[len("package://"):]
            pkg, _, sub = rest.partition("/")
            root = package_roots.get(pkg)
            if root is None:
                missing.append(ref + "  (package not shipped)")
                continue
            target = os.path.join(root, sub.replace("/", os.sep))
        else:
            target = os.path.normpath(os.path.join(base, ref.replace("/", os.sep)))
        if os.path.exists(target):
            ok += 1
        else:
            missing.append(ref)
    return len(refs), ok, missing


package_roots = {
    "piper_x_description": os.path.join(piper),
    "realsense2_description": os.path.join(OUT, "realsense2_description"),
    "piper_description": os.path.join(OUT, "piper_description"),
}

report = []
urdf_paths = []
for root, _dirs, files in os.walk(OUT):
    for name in files:
        if name.lower().endswith(".urdf"):
            urdf_paths.append(os.path.join(root, name))
urdf_paths.sort()
for path in urdf_paths:
    label = os.path.relpath(path, OUT).replace("\\", "/")
    total, ok, missing = check_urdf(path, package_roots)
    report.append("%-58s refs=%3d ok=%3d missing=%d" %
                  (label, total, ok, len(missing)))
    for m in missing:
        report.append("    MISSING: " + m)

forbidden = []
for root, _dirs, files in os.walk(OUT):
    for name in files:
        p = os.path.join(root, name)
        rel = os.path.relpath(p, OUT).replace("\\", "/")
        if "d435" in rel.lower():
            forbidden.append(rel)
            continue
        if name.lower().endswith((".urdf", ".xacro", ".md", ".py", ".xml", ".launch")):
            try:
                content = open(p, encoding="utf-8", errors="ignore").read().lower()
            except OSError:
                continue
            if "d435" in content:
                forbidden.append(rel + " (content)")

report.append("")
report.append("FORBIDDEN_D435=%d" % len(forbidden))
for item in forbidden:
    report.append("    " + item)

write_text(os.path.join(OUT, "_validate.txt"),
           "\n".join(report) + "\n\nFIXES:\n" + "\n".join(fix_log) + "\n")
print("\n".join(report))
print("\nFIXES:", fix_log)
print("OUT:", OUT)

# 6) keep the distributable zip in sync with the bundle
if os.path.isfile(ZIP_BASE + ".zip"):
    os.remove(ZIP_BASE + ".zip")
shutil.make_archive(ZIP_BASE, "zip", root_dir=os.path.dirname(OUT),
                    base_dir=os.path.basename(OUT))
print("ZIP:", ZIP_BASE + ".zip")
