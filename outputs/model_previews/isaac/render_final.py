"""Render AS2-W / Piper X / assembled robot with Isaac Sim RTX (headless)."""

import argparse
import math
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["piper_x", "as2w", "assembly"])
parser.add_argument("--out", required=True)
parser.add_argument("--pose", default="")
parser.add_argument("--poses", default="")
parser.add_argument("--piper-urdf", default="")
parser.add_argument("--piper-root", default="")
parser.add_argument("--views", default="iso,front,side")
parser.add_argument("--res", default="1600x1200")
args, _unknown = parser.parse_known_args()

REPO = r"D:\nvidia-isaac-sim\loco-manip-unified-rl-agent"
PIPER_PKG_ROOT = r"D:\nvidia-isaac-sim\own_assets\piper_isaac_sim"
PIPER_URDF = os.path.join(PIPER_PKG_ROOT, r"piper_x_description\urdf\piper_x_description.urdf")
ASM_URDF = os.path.join(REPO, r"deep_robotics_model\AS2W_PiperX\urdf\AS2W_PiperX.urdf")
ASM_DIR = os.path.dirname(ASM_URDF)
OUT_DIR = os.path.join(REPO, r"outputs\model_previews\isaac")
os.makedirs(OUT_DIR, exist_ok=True)

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

import numpy as np
import omni.kit.commands
import omni.timeline
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.sensors.camera import Camera
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
from scipy.spatial.transform import Rotation


def log(*a):
    print("[render]", *a, flush=True)


def make_material(stage, path, color, roughness=0.8, metallic=0.0):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def add_studio(stage, bmin, bmax, view_dir, floor_z):
    center = (bmin + bmax) * 0.5
    span = float(np.linalg.norm(bmax - bmin)) + 1.0
    floor_mat = make_material(stage, "/World/Studio/FloorMat", (0.50, 0.51, 0.53), 0.85)
    wall_mat = make_material(stage, "/World/Studio/WallMat", (0.26, 0.28, 0.31), 0.95)

    ground = UsdGeom.Mesh.Define(stage, "/World/Studio/Ground")
    gsize = span * 12.0
    pts = [(-gsize, -gsize, floor_z), (gsize, -gsize, floor_z),
           (gsize, gsize, floor_z), (-gsize, gsize, floor_z)]
    ground.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(floor_mat)

    d = np.asarray(view_dir, dtype=float)
    d /= np.linalg.norm(d)
    back = center - d * (span * 2.8)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, d)
    right /= np.linalg.norm(right)
    w = span * 2.6
    h = span * 2.6
    corners = [
        back - right * w - up * h * 0.30,
        back + right * w - up * h * 0.30,
        back + right * w + up * h * 1.7,
        back - right * w + up * h * 1.7,
    ]
    wall = UsdGeom.Mesh.Define(stage, "/World/Studio/Backdrop")
    wall.CreatePointsAttr([Gf.Vec3f(*c) for c in corners])
    wall.CreateFaceVertexCountsAttr([4])
    wall.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    wall.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(wall.GetPrim()).Bind(wall_mat)


def _direction_to_euler(direction):
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    z = -d
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(z, up))) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.stack([x, y, z], axis=1)
    e = Rotation.from_matrix(m).as_euler("xyz")
    return Gf.Vec3f(*[float(v) for v in e])


def add_lights(stage, view_dir):
    d = np.asarray(view_dir, dtype=float)
    d /= np.linalg.norm(d)

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(220.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    key = UsdLux.DistantLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(1500.0)
    key.CreateAngleAttr(9.0)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.94))
    kd = -d * 0.55 + np.array([0.0, 0.0, 1.0]) * 1.0
    kd /= np.linalg.norm(kd)
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(_direction_to_euler(kd))

    fill = UsdLux.DistantLight.Define(stage, "/World/Lights/Fill")
    fill.CreateIntensityAttr(380.0)
    fill.CreateAngleAttr(35.0)
    fill.CreateColorAttr(Gf.Vec3f(0.90, 0.94, 1.0))
    fd = -d * 0.1 + np.array([0.0, 0.0, 0.55])
    fd /= np.linalg.norm(fd)
    UsdGeom.Xformable(fill.GetPrim()).AddRotateXYZOp().Set(_direction_to_euler(fd))


def visible_bbox(robot_path):
    stage = omni.usd.get_context().get_stage()
    robot = stage.GetPrimAtPath(robot_path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    lo = np.array([1e9, 1e9, 1e9])
    hi = np.array([-1e9, -1e9, -1e9])
    found = False
    for link in robot.GetChildren():
        img = UsdGeom.Imageable(link)
        if img and img.GetVisibilityAttr().Get() == UsdGeom.Tokens.invisible:
            continue
        for p in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
            if not p.IsA(UsdGeom.Mesh):
                continue
            b = cache.ComputeWorldBound(p).ComputeAlignedRange()
            if b.IsEmpty():
                continue
            lo = np.minimum(lo, np.array(b.GetMin()))
            hi = np.maximum(hi, np.array(b.GetMax()))
            found = True
    if not found:
        raise RuntimeError("no visible mesh prims found")
    return lo, hi


def look_at(pos, target, up=np.array([0.0, 0.0, 1.0])):
    fwd = target - pos
    fwd /= np.linalg.norm(fwd)
    z = -fwd
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-8:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.stack([x, y, z], axis=1)
    q = Rotation.from_matrix(m).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def import_urdf(urdf_path, cwd):
    prev = os.getcwd()
    os.chdir(cwd)
    try:
        ok, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
        for name, val in [("merge_fixed_joints", False), ("convex_decomp", False),
                          ("import_inertia_tensor", True), ("fix_base", True),
                          ("distance_scale", 1.0)]:
            try:
                setattr(cfg, name, val)
            except Exception:
                pass
        ok, root = omni.kit.commands.execute(
            "URDFParseAndImportFile", urdf_path=urdf_path, import_config=cfg,
            get_articulation_root=True)
    finally:
        os.chdir(prev)
    log("import", ok, root)
    return root


def main():
    if args.model == "piper_x":
        art_root = import_urdf(args.piper_urdf or PIPER_URDF, args.piper_root or PIPER_PKG_ROOT)
        robot_path = "/piper_x_description"
    else:
        art_root = import_urdf(ASM_URDF, ASM_DIR)
        robot_path = "/AS2W_PiperX"

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = World(stage_units_in_meters=1.0)
    world.get_physics_context().set_gravity(0.0)

    if args.model == "as2w":
        for link in stage.GetPrimAtPath(robot_path).GetChildren():
            nm = link.GetName()
            if nm.startswith("arm_") or nm.startswith("gripper"):
                UsdGeom.Imageable(link).MakeInvisible()

    values = {}
    if args.model in ("piper_x", "assembly"):
        values = {"joint1": 0.0, "joint2": 0.95, "joint3": -1.25,
                  "joint4": 0.0, "joint5": 0.8, "joint6": 0.0,
                  "joint7": 0.035, "joint8": -0.035}
        values.update({"arm_joint1": 0.0, "arm_joint2": 0.95, "arm_joint3": -1.25,
                       "arm_joint4": 0.0, "arm_joint5": 0.8, "arm_joint6": 0.0,
                       "gripper_joint1": 0.035, "gripper_joint2": -0.035})
    if args.pose:
        for item in args.pose.split(","):
            k, v = item.split("=")
            values[k.strip()] = float(v)

    world.reset()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    world.step(render=False)

    try:
        art = Articulation(art_root)
        art.initialize()
        dofs = list(art.dof_names)
        log("dofs", dofs)
        targets = np.array([float(values.get(n, 0.0)) for n in dofs], dtype=np.float32)
        art.set_joint_positions(targets)
        for _ in range(3):
            world.step(render=False)
        log("applied", np.round(art.get_joint_positions(), 3))
    except Exception as exc:
        log("articulation skipped:", exc)

    bmin, bmax = visible_bbox(robot_path)
    log("bbox", np.round(bmin, 3), np.round(bmax, 3))

    views = {
        "iso": np.array([1.0, -1.25, 0.72]),
        "iso2": np.array([1.0, 1.15, 0.62]),
        "front": np.array([1.0, -0.08, 0.30]),
        "side": np.array([0.04, -1.0, 0.22]),
        "top": np.array([0.35, -0.45, 1.4]),
        "low": np.array([1.0, -0.55, 0.18]),
    }

    res_w, res_h = [int(v) for v in args.res.lower().split("x")]
    center = (bmin + bmax) * 0.5
    radius = 0.5 * float(np.linalg.norm(bmax - bmin))

    camera = Camera(prim_path="/World/Camera", resolution=(res_w, res_h), frequency=60)
    camera.initialize()
    camera.set_focal_length(58.0)
    camera.set_horizontal_aperture(36.0)

    pose_sets = []
    if args.poses:
        for chunk in args.poses.split(";"):
            item = {}
            for kv in chunk.split(","):
                k, v = kv.split("=")
                item[k.strip()] = float(v)
            pose_sets.append(item)

    if pose_sets:
        for idx, pose in enumerate(pose_sets):
            full = dict(values)
            full.update(pose)
            try:
                art = Articulation(art_root)
                art.initialize()
                dofs = list(art.dof_names)
                art.set_joint_positions(np.array([float(full.get(n, 0.0)) for n in dofs], dtype=np.float32))
                for _ in range(3):
                    world.step(render=False)
            except Exception as exc:
                log("sweep articulation skipped:", exc)
            bmin, bmax = visible_bbox(robot_path)
            center = (bmin + bmax) * 0.5
            radius = 0.5 * float(np.linalg.norm(bmax - bmin))
            direction = views["iso"] / np.linalg.norm(views["iso"])
            if stage.GetPrimAtPath("/World/Studio"):
                stage.RemovePrim("/World/Studio")
            if stage.GetPrimAtPath("/World/Lights"):
                stage.RemovePrim("/World/Lights")
            add_studio(stage, bmin, bmax, direction, floor_z=float(bmin[2]) - 0.0015)
            add_lights(stage, direction)
            fov_h = 2.0 * math.atan(0.5 * 36.0 / 58.0)
            fov_v = 2.0 * math.atan(math.tan(fov_h * 0.5) * res_h / res_w)
            fov = min(fov_h, fov_v)
            dist = radius / math.tan(fov * 0.5) * 1.0
            cam_pos = center + direction * dist
            cam_pos[2] = max(cam_pos[2], bmin[2] + radius * 0.22)
            camera.set_world_pose(position=cam_pos, orientation=look_at(cam_pos, center), camera_axes="usd")
            for _ in range(55):
                world.step(render=True)
            arr = np.asarray(camera.get_rgba())
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            path = os.path.join(OUT_DIR, "%s_p%d.png" % (args.out, idx))
            Image.fromarray(arr[:, :, :3], "RGB").save(path)
            log("saved", path)
        timeline.stop()
        app.close()
        return

    for name in [v.strip() for v in args.views.split(",") if v.strip()]:
        direction = views[name] / np.linalg.norm(views[name])
        if stage.GetPrimAtPath("/World/Studio"):
            stage.RemovePrim("/World/Studio")
        if stage.GetPrimAtPath("/World/Lights"):
            stage.RemovePrim("/World/Lights")
        add_studio(stage, bmin, bmax, direction, floor_z=float(bmin[2]) - 0.0015)
        add_lights(stage, direction)

        fov_h = 2.0 * math.atan(0.5 * 36.0 / 58.0)
        fov_v = 2.0 * math.atan(math.tan(fov_h * 0.5) * res_h / res_w)
        fov = min(fov_h, fov_v)
        dist = radius / math.tan(fov * 0.5) * 0.92
        cam_pos = center + direction * dist
        cam_pos[2] = max(cam_pos[2], bmin[2] + radius * 0.22)
        camera.set_world_pose(position=cam_pos, orientation=look_at(cam_pos, center),
                              camera_axes="usd")

        for _ in range(60):
            world.step(render=True)
        arr = np.asarray(camera.get_rgba())
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        path = os.path.join(OUT_DIR, "%s_%s.png" % (args.out, name))
        Image.fromarray(arr[:, :, :3], "RGB").save(path)
        log("saved", path)

    timeline.stop()
    app.close()


if __name__ == "__main__":
    main()
