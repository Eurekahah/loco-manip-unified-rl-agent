#!/usr/bin/env python3
"""Probe object bounds and arm reach for the two-cylinder grasp."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Probe object geometry and grasp reach.")
parser.add_argument("--task", type=str, default="Isaac-M20-Piper-TwoCylinder-Teleop-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from pxr import Usd, UsdGeom  # noqa: E402
import omni.usd  # noqa: E402

import rl_training.tasks  # noqa: F401, E402


def _clamp_norm(vector: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    if norm > max_norm:
        return vector * (max_norm / norm)
    return vector


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 7
    env_cfg.episode_length_s = 120.0
    env_cfg.actions.pre_trained_pick_action.absolute_commands = True
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    obj = env.scene["object"]
    action_term = env.action_manager.get_term("pre_trained_pick_action")
    ee_body_id = robot.find_bodies("gripper_base")[0][0]
    finger_ids, _ = robot.find_bodies(["gripper_link1", "gripper_link2"], preserve_order=True)
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    actions[:, -1] = 1.0
    actions[:, 9] = 0.40 - action_term._default_body_pose[0, 0]
    actions[:, 10] = -0.12 - action_term._default_body_pose[0, 1]

    anchor = robot.data.root_pos_w[0, :2].clone()
    _, _, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w[0].unsqueeze(0))
    anchor_yaw = yaw[0].clone()

    def anchored_step() -> None:
        env.step(actions)
        pos = robot.data.root_pos_w[0].clone()
        roll, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w[0].unsqueeze(0))
        quat = math_utils.quat_from_euler_xyz(roll, pitch, anchor_yaw.unsqueeze(0))[0]
        pos[0], pos[1] = anchor
        robot.write_root_pose_to_sim(torch.cat([pos, quat]).unsqueeze(0))

    for _ in range(60):
        anchored_step()

    stage = omni.usd.get_context().get_stage()
    prim_path = "/World/envs/env_0/Object"
    prim = stage.GetPrimAtPath(prim_path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    lo = torch.tensor(list(rng.GetMin()), dtype=torch.float32)
    hi = torch.tensor(list(rng.GetMax()), dtype=torch.float32)
    print(f"[geom] prim={prim_path} valid={bool(prim.IsValid())}", flush=True)
    print(f"[geom] aabb_min={lo.tolist()} aabb_max={hi.tolist()} size={(hi-lo).tolist()}", flush=True)
    print(f"[geom] root={obj.data.root_pos_w[0].detach().cpu().tolist()} com={obj.data.body_com_pos_w[0].detach().cpu().tolist()}", flush=True)
    print(f"[geom] body_root={robot.data.root_pos_w[0].detach().cpu().tolist()} ee={robot.data.body_pos_w[0, ee_body_id].detach().cpu().tolist()}", flush=True)
    print(f"[geom] fingers={robot.data.body_pos_w[0, finger_ids].detach().cpu().tolist()}", flush=True)

    def servo(target_w: torch.Tensor, pitch: float, steps: int, label: str) -> None:
        actions[:, 7] = pitch
        for index in range(steps):
            ee_w = robot.data.body_pos_w[0, ee_body_id]
            error = target_w - ee_w
            if torch.linalg.vector_norm(error) < 0.012:
                break
            root_pos = robot.data.root_pos_w[0]
            root_quat = robot.data.root_quat_w[0]
            desired_b = math_utils.quat_apply(math_utils.quat_inv(root_quat), ee_w + _clamp_norm(error, 0.015) - root_pos)
            actions[:, 3:6] = (desired_b - action_term._default_ee_pos_b[0]).unsqueeze(0)
            anchored_step()
        ee = robot.data.body_pos_w[0, ee_body_id]
        fingers = robot.data.body_pos_w[0, finger_ids]
        err = torch.linalg.vector_norm(target_w - ee).item()
        print(
            f"[geom] {label} pitch={pitch:+.2f} steps={index} err={err:.4f} "
            f"ee={ee.detach().cpu().numpy().round(3).tolist()} "
            f"finger_mid={fingers.mean(dim=0).detach().cpu().numpy().round(3).tolist()}",
            flush=True,
        )

    object_com = obj.data.body_com_pos_w[0].clone()
    servo(object_com + torch.tensor([-0.16, 0.0, 0.10], device=env.device), 0.0, 70, "pre-grasp-p0")
    servo(object_com, 0.0, 70, "grasp-p0")
    servo(object_com, -0.5, 70, "grasp-pm05")
    servo(object_com, -1.0, 70, "grasp-pm10")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
