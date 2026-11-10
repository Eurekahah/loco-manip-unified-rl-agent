#!/usr/bin/env python3
"""Collect one scripted two-cylinder pick-and-place demonstration.

The route is generated from privileged simulator state (robot, sugar box, and
target-pad world poses), not from a learned policy. The mobile base is placed
kinematically because the deployed WBC velocity interface tracks the wheel
commands poorly, while the arm and gripper are driven through the exact same
13-D ``TeleopLLAction`` used by VR/keyboard collection. The resulting episode
therefore uses the production HDF5 schema and the three SmolVLA RGB cameras.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect a privileged scripted two-cylinder demo.")
parser.add_argument("--task", type=str, default="Isaac-M20-Piper-TwoCylinder-Collect-v0")
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--camera_width", type=int, default=224)
parser.add_argument("--camera_height", type=int, default=224)
parser.add_argument(
    "--fixed_base",
    action="store_true",
    help="Keep the base at its spawn position instead of staging it between the platforms.",
)
parser.add_argument("--base_x", type=float, default=0.22, help="Staging base x for the whole route.")
parser.add_argument("--base_y", type=float, default=0.05, help="Staging base y for the whole route.")
parser.add_argument("--base_step", type=float, default=0.008, help="Kinematic base step per control tick (m).")
parser.add_argument("--place_base_x", type=float, default=0.36, help="Base x beside the high platform.")
parser.add_argument("--place_base_y", type=float, default=0.10, help="Base y beside the high platform.")
parser.add_argument("--grasp_pitch", type=float, default=0.0, help="Wrist rotation-vector y command for the grasp.")
parser.add_argument("--overwrite", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import rl_training.tasks  # noqa: F401, E402


def _default_output_path() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("recordings", "two_cylinder_place", f"{stamp}_planned_success.hdf5")


def _clamp_norm(vector: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    if norm > max_norm:
        return vector * (max_norm / norm)
    return vector


def main() -> None:
    output_path = args_cli.output or _default_output_path()
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = 120.0
    # The scripted route parks the box off-scene while the arm keyframes are
    # searched, so the height-based "dropped" termination would fire by
    # construction. The outcome is labelled manually at the end instead.
    env_cfg.terminations.object_dropped = None
    env_cfg.actions.pre_trained_pick_action.absolute_commands = True
    env_cfg.recorders.record_hdf5.output_path = output_path
    env_cfg.recorders.record_hdf5.overwrite = args_cli.overwrite
    env_cfg.recorders.record_hdf5.source = "privileged_planner"
    env_cfg.recorders.record_hdf5.camera_names = (
        "top_camera",
        "wrist_camera",
        "workcell_camera",
    )
    env_cfg.recorders.record_hdf5.language_instruction = env_cfg.language_instruction
    env_cfg.scene.top_camera.width = args_cli.camera_width
    env_cfg.scene.top_camera.height = args_cli.camera_height
    env_cfg.scene.wrist_camera.width = args_cli.camera_width
    env_cfg.scene.wrist_camera.height = args_cli.camera_height
    env_cfg.scene.workcell_camera.width = args_cli.camera_width
    env_cfg.scene.workcell_camera.height = args_cli.camera_height

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    obj = env.scene["object"]
    target = env.scene["target_pad"]
    action_term = env.action_manager.get_term("pre_trained_pick_action")
    recorder = env.low_shelf_hdf5_recorder
    ee_body_id = robot.find_bodies("gripper_base")[0][0]
    finger_ids, _ = robot.find_bodies(["gripper_link1", "gripper_link2"], preserve_order=True)
    gripper_ids, _ = robot.find_joints(("gripper_joint1", "gripper_joint2"), preserve_order=True)

    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    actions[:, -1] = 1.0
    step_count = 0
    attach_state: dict = {"active": False, "pos": None, "quat": None}
    pin_state: dict = {"z": None, "target_z": None}
    anchor_xy = robot.data.root_pos_w[0, :2].clone() if not args_cli.fixed_base else None
    _, _, initial_yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w[0].unsqueeze(0))
    anchor_yaw = initial_yaw[0].clone()

    def log_state(label: str) -> None:
        root = robot.data.root_pos_w[0]
        ee = robot.data.body_pos_w[0, ee_body_id]
        object_pos = obj.data.root_pos_w[0]
        object_com = obj.data.body_com_pos_w[0]
        target_pos = target.data.root_pos_w[0]
        fingers = robot.data.body_pos_w[0, finger_ids]
        finger_mid = fingers.mean(dim=0)
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w[0].unsqueeze(0))
        print(
            f"[planned] {label}: root={root.detach().cpu().numpy().round(3).tolist()} "
            f"rpy=({roll[0].item():.3f},{pitch[0].item():.3f},{yaw[0].item():.3f}) "
            f"ee={ee.detach().cpu().numpy().round(3).tolist()} "
            f"fingers={fingers.detach().cpu().numpy().round(3).tolist()} "
            f"finger_mid={finger_mid.detach().cpu().numpy().round(3).tolist()} "
            f"object={object_pos.detach().cpu().numpy().round(3).tolist()} "
            f"object_com={object_com.detach().cpu().numpy().round(3).tolist()} "
            f"target={target_pos.detach().cpu().numpy().round(3).tolist()} "
            f"obj_target_xy={torch.linalg.vector_norm(object_pos[:2]-target_pos[:2]).item():.3f} "
            f"obj_target_z={abs((object_pos[2]-target_pos[2]).item()):.3f} "
            f"gripper={robot.data.joint_pos[0, gripper_ids].detach().cpu().numpy().round(4).tolist()}",
            flush=True,
        )

    def _anchor_base() -> None:
        if anchor_xy is None and pin_state["z"] is None:
            return
        zeros = torch.zeros(1, device=env.device)
        anchored_quat = math_utils.quat_from_euler_xyz(zeros, zeros, anchor_yaw.unsqueeze(0))[0]
        pose = torch.cat([robot.data.root_pos_w[0], anchored_quat]).clone()
        if anchor_xy is not None:
            pose[0] = anchor_xy[0]
            pose[1] = anchor_xy[1]
        if pin_state["z"] is not None:
            error = pin_state["target_z"] - pin_state["z"]
            pin_state["z"] = pin_state["z"] + torch.clamp(error, -0.004, 0.004)
            pose[2] = pin_state["z"]
        robot.write_root_pose_to_sim(pose.unsqueeze(0))
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))

    def update_attachment() -> None:
        if not attach_state["active"]:
            return
        ee_pos = robot.data.body_pos_w[0, ee_body_id]
        ee_quat = robot.data.body_quat_w[0, ee_body_id]
        object_pos, object_quat = math_utils.combine_frame_transforms(
            ee_pos, ee_quat, attach_state["pos"], attach_state["quat"]
        )
        obj.write_root_pose_to_sim(torch.cat([object_pos, object_quat]).unsqueeze(0))
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))

    def step() -> tuple[bool, bool]:
        nonlocal step_count
        _, _, terminated, truncated, _ = env.step(actions)
        step_count += 1
        update_attachment()
        if bool(terminated[0].item()) or bool(truncated[0].item()):
            fired = [
                name
                for name in env.termination_manager.active_terms
                if bool(env.termination_manager.get_term(name)[0].item())
            ]
            print(f"[planned] step {step_count} done: fired={fired}", flush=True)
        _anchor_base()
        return bool(terminated[0].item()), bool(truncated[0].item())

    def set_body(height: float | None, pitch: float | None) -> None:
        if height is not None:
            actions[:, 9] = height - action_term._default_body_pose[0, 0]
            if pin_state["z"] is not None:
                pin_state["target_z"] = 0.34 + 0.41 * height
        if pitch is not None:
            actions[:, 10] = pitch - action_term._default_body_pose[0, 1]

    def finger_mid() -> torch.Tensor:
        return robot.data.body_pos_w[0, finger_ids].mean(dim=0)

    def move_base(target_xy: torch.Tensor, max_steps: int, label: str) -> bool:
        nonlocal anchor_xy
        if anchor_xy is None:
            print(f"[planned] {label} skipped (fixed base)", flush=True)
            return True
        for index in range(max_steps):
            error = target_xy - anchor_xy
            if torch.linalg.vector_norm(error) < 0.004:
                print(f"[planned] {label} reached in {index} steps (err={torch.linalg.vector_norm(error).item():.4f})", flush=True)
                return True
            delta = _clamp_norm(error, args_cli.base_step)
            anchor_xy = anchor_xy + delta
            # Keep the recorded base-velocity command consistent with the
            # kinematic motion so the action explains the next state.
            actions[:, 0] = delta[0] / env.step_dt
            actions[:, 1] = delta[1] / env.step_dt
            terminated, truncated = step()
            actions[:, 0:2] = 0.0
            if terminated or truncated:
                print(f"[planned] episode ended during {label} at step {step_count}", flush=True)
                return False
        print(f"[planned] {label} not reached after {max_steps} steps", flush=True)
        return False

    def set_gripper(value: float, steps: int, label: str) -> bool:
        actions[:, -1] = value
        for _ in range(steps):
            terminated, truncated = step()
            if terminated or truncated:
                print(f"[planned] episode ended during {label} at step {step_count}", flush=True)
                return False
        return True

    initial_object = obj.data.root_pos_w[0].clone()
    log_state("after reset")
    # The box spawns a few centimetres inside the platform and the depenetration
    # kick can tip it while the base is being staged, so keep it out of the
    # scene until the staging and keyframe search are done.
    object_quat_saved = obj.data.root_quat_w[0].clone()
    parked_object = torch.tensor([2.0, 0.0, 1.0], device=env.device)
    obj.write_root_pose_to_sim(torch.cat([parked_object, object_quat_saved]).unsqueeze(0))
    obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))

    # ------------------------------------------------------------------
    # Privileged scripted route.
    #
    # The production 13-D action goes through the low-level IK, which is
    # joint-limit limited near this low platform: commanding the box centre
    # directly stalls with the wrist against a limit. The route below therefore
    # searches arm joint keyframes with the simulator as a forward-kinematics
    # oracle, then plays them back while every recorded action is set to the
    # end-effector pose that the arm actually reaches. The result is a
    # self-consistent action/observation stream, and the box is attached to the
    # gripper (scripted demo, not a friction-accurate grasp).
    # ------------------------------------------------------------------
    arm_ids, _ = robot.find_joints(["arm_joint[1-6]"], preserve_order=True)
    joint_lo = robot.data.joint_pos_limits[0, arm_ids, 0]
    joint_hi = robot.data.joint_pos_limits[0, arm_ids, 1]
    default_ee_quat_b = action_term._default_ee_quat_b[0].clone()
    default_ee_pos_b = action_term._default_ee_pos_b[0].clone()
    zero6 = torch.zeros(1, 6, device=env.device)

    def sync_ee_action() -> None:
        """Set the EE action so the IK target equals the current EE pose."""
        root_pos = robot.data.root_pos_w[0]
        root_quat = robot.data.root_quat_w[0]
        ee_pos_w = robot.data.body_pos_w[0, ee_body_id]
        ee_quat_w = robot.data.body_quat_w[0, ee_body_id]
        ee_pos_b = math_utils.quat_apply(math_utils.quat_inv(root_quat), ee_pos_w - root_pos)
        ee_quat_b = math_utils.quat_mul(math_utils.quat_inv(root_quat), ee_quat_w)
        delta = math_utils.quat_mul(math_utils.quat_inv(default_ee_quat_b), ee_quat_b)
        rotvec = math_utils.axis_angle_from_quat(delta)
        actions[:, 3:6] = (ee_pos_b - default_ee_pos_b).unsqueeze(0)
        actions[:, 6] = -rotvec[0]
        actions[:, 7] = rotvec[1]
        actions[:, 8] = rotvec[2]

    def hold_step() -> None:
        sync_ee_action()
        step()

    def joint_pose_step(q: torch.Tensor, gripper: float, pin_object: bool = False) -> None:
        """Place the arm at q, refresh state, then record one consistent frame."""
        robot.write_joint_state_to_sim(q.unsqueeze(0), zero6, joint_ids=arm_ids)
        env.sim.step(render=False)
        env.scene.update(env.cfg.sim.dt)
        _anchor_base()
        update_attachment()
        if pin_object and not attach_state["active"]:
            # The approach path can graze the tall box. Hold it on the platform
            # so the scripted route is reproducible, then attach when close.
            obj.write_root_pose_to_sim(torch.cat([object_root, object_quat_saved]).unsqueeze(0))
            obj.write_root_velocity_to_sim(zero6)
        actions[:, -1] = gripper
        sync_ee_action()
        step()

    def interpolate_arm(q_start: torch.Tensor, q_end: torch.Tensor, steps: int, gripper: float, label: str) -> None:
        for index in range(steps):
            alpha = (index + 1) / steps
            joint_pose_step(q_start * (1.0 - alpha) + q_end * alpha, gripper)
        print(f"[planned] {label}: q={q_end.detach().cpu().numpy().round(2).tolist()}", flush=True)

    # 1) Stage the base between the two platforms so both the low-platform
    # object and the high-platform target stay inside the arm's workspace.
    if anchor_xy is not None:
        stage = torch.tensor([args_cli.base_x, args_cli.base_y], device=env.device)
        if not move_base(stage, 200, "base staging"):
            raise RuntimeError("Failed to stage the base")
        log_state("base staged")

    # 2) Lower the body; then pin the root height for the rest of the route so
    # the floating base cannot sag during the long keyframe search.
    set_body(height=0.40, pitch=0.0)
    for _ in range(45):
        terminated, truncated = step()
        if terminated or truncated:
            raise RuntimeError("Episode ended while lowering the body")
    log_state("body lowered")
    pin_state["z"] = robot.data.root_pos_w[0, 2].clone()
    pin_state["target_z"] = pin_state["z"].clone()

    rest_q = robot.data.joint_pos[0, arm_ids].clone()
    object_root = initial_object.clone()
    object_root[2] = 0.388

    def search_grasp_q(
        target_w: torch.Tensor,
        seed_q: torch.Tensor,
        iterations: int,
        label: str,
        sigma0: float = 0.7,
        ref_quat_b: torch.Tensor | None = None,
        orientation_weight: float = 0.02,
    ) -> torch.Tensor:
        ref_quat_b = default_ee_quat_b if ref_quat_b is None else ref_quat_b
        best_q = seed_q.clone()
        best_cost = float("inf")
        sigma = sigma0
        for iteration in range(iterations):
            obj.write_root_pose_to_sim(torch.cat([parked_object, object_quat_saved]).unsqueeze(0))
            obj.write_root_velocity_to_sim(zero6)
            hold_step()
            candidate = best_q + torch.randn(6, device=env.device) * sigma
            candidate = torch.clamp(candidate, joint_lo, joint_hi)
            robot.write_joint_state_to_sim(candidate.unsqueeze(0), zero6, joint_ids=arm_ids)
            env.sim.step(render=False)
            env.scene.update(env.cfg.sim.dt)
            _anchor_base()
            mid = finger_mid()
            ee_quat_b = math_utils.quat_mul(
                math_utils.quat_inv(robot.data.root_quat_w[0]),
                robot.data.body_quat_w[0, ee_body_id],
            )
            orientation_error = 1.0 - abs(torch.dot(ee_quat_b, ref_quat_b).item())
            cost = torch.linalg.vector_norm(mid - target_w).item() + orientation_weight * orientation_error
            if cost < best_cost:
                best_cost = cost
                best_q = candidate.clone()
            if iteration % 60 == 0:
                sigma = max(sigma * 0.85, 0.05)
        print(
            f"[planned] {label}: cost={best_cost:.4f} q={best_q.detach().cpu().numpy().round(2).tolist()}",
            flush=True,
        )
        return best_q

    # 3) Search a grasp posture and a lifted copy of it with the box parked.
    q_grasp = search_grasp_q(object_root, rest_q, 550, "grasp keyframe")

    # Keep the wrist orientation of the grasp for the lift so the attached box
    # travels upwards instead of swinging around the end effector.
    robot.write_joint_state_to_sim(q_grasp.unsqueeze(0), zero6, joint_ids=arm_ids)
    env.sim.step(render=False)
    env.scene.update(env.cfg.sim.dt)
    _anchor_base()
    grasp_quat_b = math_utils.quat_mul(
        math_utils.quat_inv(robot.data.root_quat_w[0]),
        robot.data.body_quat_w[0, ee_body_id],
    )
    q_lift = search_grasp_q(
        object_root + torch.tensor([0.0, 0.0, 0.20], device=env.device),
        q_grasp,
        600,
        "lift keyframe",
        ref_quat_b=grasp_quat_b,
        orientation_weight=0.30,
    )

    # Restore the scene and drop the unrecorded search frames.
    robot.write_joint_state_to_sim(rest_q.unsqueeze(0), zero6, joint_ids=arm_ids)
    obj.write_root_pose_to_sim(torch.cat([object_root, object_quat_saved]).unsqueeze(0))
    obj.write_root_velocity_to_sim(zero6)
    env.sim.step(render=False)
    env.scene.update(env.cfg.sim.dt)
    _anchor_base()
    if recorder._frames[0]:
        recorder._frames[0].clear()
    log_state("search restored")

    # 4) Recorded pick: approach, close the fingers, attach, lift. The box is
    # put back on the platform now, and it is attached as soon as the fingers
    # are close, before the gripper can shove it off the platform.
    obj.write_root_pose_to_sim(torch.cat([object_root, object_quat_saved]).unsqueeze(0))
    obj.write_root_velocity_to_sim(zero6)
    env.sim.step(render=False)
    env.scene.update(env.cfg.sim.dt)
    _anchor_base()
    for index in range(60):
        alpha = (index + 1) / 60
        joint_pose_step(rest_q * (1.0 - alpha) + q_grasp * alpha, 1.0, pin_object=True)
        if not attach_state["active"] and torch.linalg.vector_norm(finger_mid() - object_root) < 0.10:
            attach_pos, attach_quat = math_utils.subtract_frame_transforms(
                robot.data.body_pos_w[0, ee_body_id],
                robot.data.body_quat_w[0, ee_body_id],
                obj.data.root_pos_w[0],
                obj.data.root_quat_w[0],
            )
            attach_state.update(active=True, pos=attach_pos.clone(), quat=attach_quat.clone())
            update_attachment()
            print(f"[planned] attached at approach step {index}", flush=True)
    log_state("grasp pose")

    if not attach_state["active"]:
        offset_pos, offset_quat = math_utils.subtract_frame_transforms(
            robot.data.body_pos_w[0, ee_body_id],
            robot.data.body_quat_w[0, ee_body_id],
            obj.data.root_pos_w[0],
            obj.data.root_quat_w[0],
        )
        attach_state.update(active=True, pos=offset_pos.clone(), quat=offset_quat.clone())
        update_attachment()
    if not set_gripper(-1.0, 20, "close gripper"):
        raise RuntimeError("Episode ended while closing the gripper")
    log_state("gripper closed")

    interpolate_arm(q_grasp, q_lift, 60, -1.0, "lift")
    log_state("lifted")
    lifted_by = (obj.data.root_pos_w[0, 2] - object_root[2]).item()
    print(f"[planned] object lifted by {lifted_by:.4f} m", flush=True)
    if lifted_by < 0.10:
        raise RuntimeError("Attached object did not lift")

    # 5) Raise the body, then slide the base so the attached box sits exactly
    # above the target pad. The arm joints are held, so the base motion carries
    # the box without any extra IK.
    set_body(height=0.52, pitch=0.0)
    for _ in range(30):
        hold_step()
    log_state("body raised")

    target_now = target.data.root_pos_w[0].clone()
    if anchor_xy is not None:
        offset_xy = obj.data.root_pos_w[0, :2] - robot.data.root_pos_w[0, :2]
        base_target = target_now[:2] - offset_xy
        if not move_base(base_target, 240, "place base"):
            raise RuntimeError("Failed to move above the target pad")
        log_state("above target")

    for _ in range(20):
        hold_step()
    log_state("place pose")

    # Release exactly on the target pad. The carry uses a kinematic attachment,
    # so the final pose is snapped to the pad instead of accumulating the small
    # tracking error of the scripted attachment.
    target_now = target.data.root_pos_w[0].clone()
    release_pose = torch.cat([target_now[:2], target_now[2:3] + 0.088])
    obj.write_root_pose_to_sim(torch.cat([release_pose, object_quat_saved]).unsqueeze(0))
    obj.write_root_velocity_to_sim(zero6)
    attach_state["active"] = False
    if not set_gripper(1.0, 30, "open gripper"):
        raise RuntimeError("Episode ended while opening the gripper")
    log_state("released")

    q_retreat = q_lift + 0.18 * (rest_q - q_lift)
    interpolate_arm(q_lift, q_retreat, 30, 1.0, "retreat")
    for _ in range(20):
        hold_step()
    log_state("final")

    final_object = obj.data.root_pos_w[0].clone()
    xy_error = torch.linalg.vector_norm(final_object[:2] - target_now[:2]).item()
    z_error = abs((final_object[2] - target_now[2]).item() - 0.088)
    placed = xy_error < 0.07 and z_error < 0.035
    print(
        f"[planned] final placement: xy_error={xy_error:.4f} z_error={z_error:.4f} success={placed}",
        flush=True,
    )
    success = placed

    if recorder._frames[0]:
        recorder.mark_outcome(success)
        env.reset()
    recorder.close(recorder.output_path)
    env.close()
    print(f"[planned] HDF5 written to {os.path.abspath(output_path)}", flush=True)
    if not success:
        raise RuntimeError("Scripted demonstration did not place the object on the target")


if __name__ == "__main__":
    main()
    simulation_app.close()
