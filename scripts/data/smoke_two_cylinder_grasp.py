#!/usr/bin/env python3
"""Closed-loop smoke test for approaching, grasping, and lifting the YCB can."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke-test the two-cylinder YCB grasp.")
parser.add_argument("--task", type=str, default="Isaac-M20-Piper-TwoCylinder-Teleop-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_move_steps", type=int, default=90)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import rl_training.tasks  # noqa: F401, E402


def _clamp_norm(vector: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    if norm > max_norm:
        return vector * (max_norm / norm)
    return vector


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 11
    env_cfg.episode_length_s = 120.0
    # Use calibrated absolute commands for this scripted probe. This avoids
    # integrating a closed-loop world target in the action term's body frame.
    env_cfg.actions.pre_trained_pick_action.absolute_commands = True
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    obj = env.scene["object"]
    action_term = env.action_manager.get_term("pre_trained_pick_action")
    ee_body_id = robot.find_bodies("gripper_base")[0][0]
    actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    actions[:, -1] = 1.0
    step_count = 0

    def step() -> None:
        nonlocal step_count
        _, _, terminated, truncated, _ = env.step(actions)
        step_count += 1
        if bool(terminated[0].item()) or bool(truncated[0].item()):
            fired = [
                name
                for name in env.termination_manager.active_terms
                if bool(env.termination_manager.get_term(name)[0].item())
            ]
            print(
                f"[grasp] episode reset at step={step_count} "
                f"terminated={bool(terminated[0].item())} "
                f"truncated={bool(truncated[0].item())} terms={fired}",
                flush=True,
            )

    def log(label: str) -> None:
        ee_pos = robot.data.body_pos_w[0, ee_body_id]
        obj_pos = obj.data.root_pos_w[0]
        print(
            f"[grasp] {label}: ee={ee_pos.detach().cpu().tolist()} "
            f"object={obj_pos.detach().cpu().tolist()}",
            flush=True,
        )

    # Lower and pitch the body before reaching into the low platform.
    body_height_offset = 0.40 - action_term._default_body_pose[0, 0].item()
    body_pitch_offset = -0.12 - action_term._default_body_pose[0, 1].item()
    actions[:, 9] = body_height_offset
    actions[:, 10] = body_pitch_offset
    for _ in range(12):
        step()
    for _ in range(20):
        step()
    log("body lowered")

    object_pos = obj.data.root_pos_w[0].clone()
    pre_grasp = object_pos + torch.tensor([-0.15, 0.02, 0.07], device=env.device)
    grasp_pos = object_pos + torch.tensor([-0.09, 0.02, 0.0], device=env.device)

    def move_ee(target_w: torch.Tensor, max_steps: int, label: str) -> None:
        for step_index in range(max_steps):
            root_pos = robot.data.root_pos_w[0]
            root_quat = robot.data.root_quat_w[0]
            target_b = math_utils.quat_apply(
                math_utils.quat_inv(root_quat), target_w - root_pos
            )
            actual_b, actual_quat_b = math_utils.subtract_frame_transforms(
                root_pos,
                root_quat,
                robot.data.body_pos_w[0, ee_body_id],
                robot.data.body_quat_w[0, ee_body_id],
            )
            actual_error = torch.linalg.vector_norm(target_b - actual_b)
            if actual_error < 0.012:
                print(f"[grasp] {label} target reached in {step_index} steps", flush=True)
                break
            target_b = actual_b + _clamp_norm(target_b - actual_b, 0.018)
            delta_b = target_b - action_term._default_ee_pos_b[0]
            actions[:, 3:6] = delta_b.unsqueeze(0)
            action_term._default_ee_quat_b[0] = actual_quat_b
            actions[:, 6:9] = 0.0
            step()
            if step_index % 10 == 0:
                command = env.command_manager.get_command("ee_pose")[0]
                actual_b, _ = math_utils.subtract_frame_transforms(
                    robot.data.root_pos_w[0],
                    robot.data.root_quat_w[0],
                    robot.data.body_pos_w[0, ee_body_id],
                    robot.data.body_quat_w[0, ee_body_id],
                )
                arm_ids, arm_names = robot.find_joints(["arm_joint[1-6]"], preserve_order=True)
                print(
                    f"[grasp] {label} step={step_index} cmd_b={command[:3].detach().cpu().tolist()} "
                    f"actual_b={actual_b.detach().cpu().tolist()}",
                    flush=True,
                )
                print(
                    f"[grasp] arm_target={robot.data.joint_pos_target[0, arm_ids].detach().cpu().tolist()} "
                    f"arm_pos={robot.data.joint_pos[0, arm_ids].detach().cpu().tolist()}",
                    flush=True,
                )

    move_ee(pre_grasp, args_cli.max_move_steps, "pre-grasp")
    log("pre-grasp")
    move_ee(grasp_pos, args_cli.max_move_steps, "grasp")
    log("grasp pose")
    finger_ids, _ = robot.find_bodies(["gripper_link1", "gripper_link2"], preserve_order=True)
    gripper_ids, _ = robot.find_joints(["gripper_joint1", "gripper_joint2"], preserve_order=True)
    print(
        f"[grasp] finger_pos={robot.data.body_pos_w[0, finger_ids].detach().cpu().tolist()}",
        flush=True,
    )

    # Let the low-level controller settle at the object before closing.
    for _ in range(20):
        step()
    log("before close")

    actions[:, -1] = -1.0
    for _ in range(30):
        step()
    log("after close")
    print(
        f"[grasp] gripper_joint_pos={robot.data.joint_pos[0, gripper_ids].detach().cpu().tolist()} "
        f"finger_pos={robot.data.body_pos_w[0, finger_ids].detach().cpu().tolist()}",
        flush=True,
    )

    lifted_pos = grasp_pos + torch.tensor([0.0, 0.0, 0.20], device=env.device)
    move_ee(lifted_pos, args_cli.max_move_steps, "lift")
    for _ in range(20):
        step()
    log("after lift")

    initial_height = object_pos[2].item()
    final_height = obj.data.root_pos_w[0, 2].item()
    print(
        f"[grasp] object height delta={final_height - initial_height:.4f} m "
        f"(initial={initial_height:.4f}, final={final_height:.4f})",
        flush=True,
    )
    print(
        f"[grasp] RESULT={'PASS' if final_height - initial_height > 0.08 else 'FAIL'}",
        flush=True,
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
