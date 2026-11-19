from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch


@dataclass(frozen=True)
class ProjectedIKOptions:
    """Runtime options for the batched projected DLS step."""

    position_tolerance: float = 0.01
    orientation_tolerance: float = 0.08
    position_weight: float = 1.0
    orientation_weight: float = 0.35
    step_gain: float = 0.45
    max_joint_step: float = 0.08
    adaptive_damping: bool = True
    damping_min: float = 0.002
    damping_max: float = 0.15
    singular_value_threshold: float = 0.06


class ProjectedIKStep(NamedTuple):
    """Result of one projected DLS step."""

    joint_pos: torch.Tensor
    delta_joint_pos: torch.Tensor
    position_error: torch.Tensor
    orientation_error: torch.Tensor
    damping: torch.Tensor


class MultiStartIKResult(NamedTuple):
    """Best result returned by the batched multi-start solver."""

    joint_pos: torch.Tensor
    position_error: torch.Tensor
    orientation_error: torch.Tensor
    success: torch.Tensor
    quality: torch.Tensor


def normalize_quat(quat: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    """Normalize ``wxyz`` quaternions."""

    return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(eps)


def quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Multiply two batches of ``wxyz`` quaternions."""

    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    """Return the conjugate of ``wxyz`` quaternions."""

    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def orientation_error(current_quat: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
    """Return the shortest-path axis-angle error that maps current to target."""

    current_quat = normalize_quat(current_quat)
    target_quat = normalize_quat(target_quat)
    error_quat = quat_mul(target_quat, quat_conjugate(current_quat))
    error_quat = torch.where(error_quat[..., :1] < 0.0, -error_quat, error_quat)
    error_quat = normalize_quat(error_quat)

    vector = error_quat[..., 1:]
    sin_half = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, error_quat[..., :1].clamp_min(0.0))
    axis_scale = torch.where(
        sin_half > 1.0e-7,
        angle / sin_half.clamp_min(1.0e-7),
        2.0,
    )
    return vector * axis_scale


def quat_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Convert a batch of rotation matrices to normalized ``wxyz`` quaternions."""

    m00, m01, m02 = matrix[:, 0, 0], matrix[:, 0, 1], matrix[:, 0, 2]
    m10, m11, m12 = matrix[:, 1, 0], matrix[:, 1, 1], matrix[:, 1, 2]
    m20, m21, m22 = matrix[:, 2, 0], matrix[:, 2, 1], matrix[:, 2, 2]

    trace = m00 + m11 + m22
    quat = torch.zeros((matrix.shape[0], 4), dtype=matrix.dtype, device=matrix.device)

    cond1 = trace > 0.0
    cond2 = (~cond1) & (m00 > m11) & (m00 > m22)
    cond3 = (~cond1) & (~cond2) & (m11 > m22)
    cond4 = (~cond1) & (~cond2) & (~cond3)

    if cond1.any():
        scale = torch.sqrt(trace[cond1] + 1.0) * 2.0
        quat[cond1, 0] = 0.25 * scale
        quat[cond1, 1] = (m21[cond1] - m12[cond1]) / scale
        quat[cond1, 2] = (m02[cond1] - m20[cond1]) / scale
        quat[cond1, 3] = (m10[cond1] - m01[cond1]) / scale
    if cond2.any():
        scale = torch.sqrt(1.0 + m00[cond2] - m11[cond2] - m22[cond2]) * 2.0
        quat[cond2, 0] = (m21[cond2] - m12[cond2]) / scale
        quat[cond2, 1] = 0.25 * scale
        quat[cond2, 2] = (m01[cond2] + m10[cond2]) / scale
        quat[cond2, 3] = (m02[cond2] + m20[cond2]) / scale
    if cond3.any():
        scale = torch.sqrt(1.0 + m11[cond3] - m00[cond3] - m22[cond3]) * 2.0
        quat[cond3, 0] = (m02[cond3] - m20[cond3]) / scale
        quat[cond3, 1] = (m01[cond3] + m10[cond3]) / scale
        quat[cond3, 2] = 0.25 * scale
        quat[cond3, 3] = (m12[cond3] + m21[cond3]) / scale
    if cond4.any():
        scale = torch.sqrt(1.0 + m22[cond4] - m00[cond4] - m11[cond4]) * 2.0
        quat[cond4, 0] = (m10[cond4] - m01[cond4]) / scale
        quat[cond4, 1] = (m02[cond4] + m20[cond4]) / scale
        quat[cond4, 2] = (m12[cond4] + m21[cond4]) / scale
        quat[cond4, 3] = 0.25 * scale
    return normalize_quat(quat)


def _task_weighted_step(
    current_pos: torch.Tensor,
    current_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    jacobian: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
    options: ProjectedIKOptions,
    *,
    position_only: bool = False,
) -> ProjectedIKStep:
    """Compute one weighted, damped and projected joint-space step."""

    position_error = target_pos - current_pos
    ori_error = orientation_error(current_quat, target_quat)
    if position_only:
        pose_error = position_error
        task_jacobian = jacobian[:, :3]
        row_weight = torch.full(
            (position_error.shape[0], 3),
            math.sqrt(options.position_weight),
            dtype=position_error.dtype,
            device=position_error.device,
        )
    else:
        pose_error = torch.cat((position_error, ori_error), dim=-1)
        task_jacobian = jacobian
        row_weight = torch.cat(
            (
                torch.full_like(position_error, math.sqrt(options.position_weight)),
                torch.full_like(ori_error, math.sqrt(options.orientation_weight)),
            ),
            dim=-1,
        )

    weighted_error = pose_error * row_weight
    weighted_jacobian = task_jacobian * row_weight.unsqueeze(-1)

    jacobian_t = weighted_jacobian.transpose(1, 2)
    normal = weighted_jacobian @ jacobian_t
    if options.adaptive_damping:
        # The determinant of JJ^T is the squared manipulability. It is much cheaper
        # than a batched SVD at high control rates and still gives a smooth singularity
        # signal for the small 3x3/6x6 task blocks used here.
        manipulability = torch.linalg.det(normal).clamp_min(0.0).sqrt()
        task_rank = min(task_jacobian.shape[1], task_jacobian.shape[2])
        threshold = max(options.singular_value_threshold, 1.0e-6) ** task_rank
        normalized = (manipulability / threshold).clamp(0.0, 1.0)
        damping_sq = (1.0 - normalized.square()) * options.damping_max**2
        damping_sq = damping_sq.clamp_min(options.damping_min**2)
    else:
        damping_sq = torch.full(
            (weighted_jacobian.shape[0],),
            options.damping_max**2,
            dtype=weighted_jacobian.dtype,
            device=weighted_jacobian.device,
        )

    normal += damping_sq[:, None, None] * torch.eye(
        normal.shape[-1], dtype=normal.dtype, device=normal.device
    ).unsqueeze(0)
    delta = jacobian_t @ torch.linalg.solve(normal, weighted_error.unsqueeze(-1))
    delta = delta.squeeze(-1) * options.step_gain
    delta = delta.clamp(-options.max_joint_step, options.max_joint_step)
    next_joint_pos = (joint_pos + delta).clamp(joint_lower, joint_upper)
    effective_delta = next_joint_pos - joint_pos
    return ProjectedIKStep(
        joint_pos=next_joint_pos,
        delta_joint_pos=effective_delta,
        position_error=position_error,
        orientation_error=ori_error,
        damping=damping_sq.sqrt(),
    )


def projected_ik_step(
    current_pos: torch.Tensor,
    current_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    jacobian: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
    options: ProjectedIKOptions,
    *,
    position_only: bool = False,
) -> ProjectedIKStep:
    """Public wrapper used by the action term."""

    return _task_weighted_step(
        current_pos,
        current_quat,
        target_pos,
        target_quat,
        jacobian,
        joint_pos,
        joint_lower,
        joint_upper,
        options,
        position_only=position_only,
    )


def solve_multi_start(
    chain,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    seeds: torch.Tensor,
    seed_reference: torch.Tensor,
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
    q_rest: torch.Tensor,
    options: ProjectedIKOptions,
    *,
    max_iterations: int,
    full_step_gain: float,
    full_max_joint_step: float,
    seed_travel_weight: float,
    position_only: bool = False,
) -> MultiStartIKResult:
    """Run batched multi-start projected DLS and return the best candidate per environment.

    Args:
        chain: A ``pytorch_kinematics.SerialChain`` from arm root to end effector.
        target_pos/target_quat: ``(N, 3)`` and ``(N, 4)`` targets in the chain root frame.
        seeds: ``(S, N, J)`` candidate joint configurations.
        seed_reference: ``(N, J)`` configuration used to prefer nearby solutions.
    """

    num_seeds, num_envs, num_joints = seeds.shape
    flat_seeds = seeds.reshape(num_seeds * num_envs, num_joints).clone()
    if joint_lower.dim() == 1:
        joint_lower = joint_lower.unsqueeze(0).expand(num_envs, -1)
    if joint_upper.dim() == 1:
        joint_upper = joint_upper.unsqueeze(0).expand(num_envs, -1)
    lower = joint_lower.unsqueeze(0).expand(num_seeds, -1, -1).reshape(-1, num_joints)
    upper = joint_upper.unsqueeze(0).expand(num_seeds, -1, -1).reshape(-1, num_joints)
    if q_rest.dim() == 1:
        q_rest = q_rest.unsqueeze(0).expand(num_envs, -1)
    rest = q_rest.unsqueeze(0).expand(num_seeds, -1, -1).reshape(-1, num_joints)
    target_position = target_pos.unsqueeze(0).expand(num_seeds, -1, -1).reshape(-1, 3)
    target_orientation = target_quat.unsqueeze(0).expand(num_seeds, -1, -1).reshape(-1, 4)

    joint_pos = flat_seeds.clamp(lower, upper)
    best_joint_pos = joint_pos.clone()
    best_position_error = torch.full(
        (num_seeds * num_envs,), float("inf"), dtype=joint_pos.dtype, device=joint_pos.device
    )
    best_orientation_error = best_position_error.clone()
    best_quality = best_position_error.clone()
    seed_distance = torch.linalg.vector_norm(joint_pos - seed_reference.repeat(num_seeds, 1), dim=-1)
    rest_distance = torch.linalg.vector_norm(joint_pos - rest, dim=-1)

    solve_options = ProjectedIKOptions(
        position_tolerance=options.position_tolerance,
        orientation_tolerance=options.orientation_tolerance,
        position_weight=options.position_weight,
        orientation_weight=options.orientation_weight,
        step_gain=full_step_gain,
        max_joint_step=full_max_joint_step,
        adaptive_damping=options.adaptive_damping,
        damping_min=options.damping_min,
        damping_max=options.damping_max,
        singular_value_threshold=options.singular_value_threshold,
    )

    for _ in range(max_iterations):
        jacobian, transform = chain.jacobian(joint_pos, ret_eef_pose=True)
        current_position = transform[:, :3, 3]
        current_orientation = quat_from_matrix(transform[:, :3, :3])
        position_error = target_position - current_position
        ori_error = orientation_error(current_orientation, target_orientation)

        pos_norm = torch.linalg.vector_norm(position_error, dim=-1)
        ori_norm = torch.linalg.vector_norm(ori_error, dim=-1)
        normalized_ori = 0.0 if position_only else options.orientation_weight * ori_norm
        quality = (
            pos_norm / max(options.position_tolerance, 1.0e-6)
            + normalized_ori / max(options.orientation_tolerance, 1.0e-6)
            + seed_travel_weight * seed_distance
            + 0.01 * rest_distance
        )
        improvement = quality < best_quality
        best_quality = torch.where(improvement, quality, best_quality)
        best_position_error = torch.where(improvement, pos_norm, best_position_error)
        best_orientation_error = torch.where(improvement, ori_norm, best_orientation_error)
        best_joint_pos[improvement] = joint_pos[improvement]

        step = _task_weighted_step(
            current_position,
            current_orientation,
            target_position,
            target_orientation,
            jacobian,
            joint_pos,
            lower,
            upper,
            solve_options,
            position_only=position_only,
        )
        joint_pos = step.joint_pos

    best_joint_pos = best_joint_pos.reshape(num_seeds, num_envs, num_joints)
    best_quality = best_quality.reshape(num_seeds, num_envs)
    best_position_error = best_position_error.reshape(num_seeds, num_envs)
    best_orientation_error = best_orientation_error.reshape(num_seeds, num_envs)
    best_seed_index = torch.argmin(best_quality, dim=0)
    env_index = torch.arange(num_envs, device=best_joint_pos.device)
    selected_joint_pos = best_joint_pos[best_seed_index, env_index]
    selected_position_error = best_position_error[best_seed_index, env_index]
    selected_orientation_error = best_orientation_error[best_seed_index, env_index]
    selected_quality = best_quality[best_seed_index, env_index]
    if position_only:
        success = selected_position_error <= options.position_tolerance
    else:
        success = (selected_position_error <= options.position_tolerance) & (
            selected_orientation_error <= options.orientation_tolerance
        )
    return MultiStartIKResult(
        joint_pos=selected_joint_pos,
        position_error=selected_position_error,
        orientation_error=selected_orientation_error,
        success=success,
        quality=selected_quality,
    )


def make_seed_candidates(
    current_joint_pos: torch.Tensor,
    previous_solution: torch.Tensor,
    q_rest: torch.Tensor,
    joint_lower: torch.Tensor,
    joint_upper: torch.Tensor,
    num_random: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Create ``(S, N, J)`` seeds from continuity, rest and deterministic random starts."""

    lower = joint_lower.expand_as(current_joint_pos)
    upper = joint_upper.expand_as(current_joint_pos)
    if q_rest.dim() == 1:
        q_rest = q_rest.unsqueeze(0).expand_as(current_joint_pos)
    seeds = [
        current_joint_pos,
        previous_solution,
        q_rest,
        lower + 0.25 * (upper - lower),
        lower + 0.75 * (upper - lower),
    ]
    if num_random > 0:
        random_unit = torch.rand(
            (num_random, *current_joint_pos.shape),
            dtype=current_joint_pos.dtype,
            device=current_joint_pos.device,
            generator=generator,
        )
        random_seeds = lower.unsqueeze(0) + random_unit * (upper - lower).unsqueeze(0)
        seeds.extend(random_seeds.unbind(dim=0))
    return torch.stack(seeds, dim=0).clamp(lower.unsqueeze(0), upper.unsqueeze(0))
