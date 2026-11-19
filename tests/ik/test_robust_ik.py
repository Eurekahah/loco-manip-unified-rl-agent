from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
IK_MODULE_PATH = (
    ROOT
    / "source"
    / "rl_training"
    / "rl_training"
    / "tasks"
    / "manager_based"
    / "locomotion"
    / "velocity"
    / "mdp"
    / "ik.py"
)


@pytest.fixture(scope="module")
def ik_module():
    spec = importlib.util.spec_from_file_location("robust_ik_test_module", IK_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_projected_step_respects_limits_and_step_budget(ik_module):
    current_pos = torch.zeros(1, 3)
    current_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    target_pos = torch.tensor([[1.0, 1.0, 1.0]])
    target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    jacobian = torch.eye(6).unsqueeze(0)
    joint_pos = torch.zeros(1, 6)
    lower = torch.full((6,), -0.1)
    upper = torch.full((6,), 0.1)
    options = ik_module.ProjectedIKOptions(
        step_gain=1.0,
        max_joint_step=0.03,
        damping_min=0.001,
        damping_max=0.01,
        singular_value_threshold=0.1,
    )

    result = ik_module.projected_ik_step(
        current_pos,
        current_quat,
        target_pos,
        target_quat,
        jacobian,
        joint_pos,
        lower,
        upper,
        options,
    )

    assert torch.allclose(
        result.delta_joint_pos[:, :3].abs(), torch.full((1, 3), 0.03)
    )
    assert torch.allclose(result.delta_joint_pos[:, 3:], torch.zeros(1, 3), atol=1.0e-5)
    assert torch.all(result.joint_pos <= upper)
    assert torch.all(result.joint_pos >= lower)


def test_orientation_error_uses_shortest_quaternion_path(ik_module):
    current = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    equivalent_target = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])

    error = ik_module.orientation_error(current, equivalent_target)

    assert torch.allclose(error, torch.zeros_like(error), atol=1.0e-6)


def test_multi_start_ik_finds_reachable_two_link_pose(ik_module):
    pk = pytest.importorskip("pytorch_kinematics")
    urdf = """
    <robot name="two_link">
      <link name="base"/>
      <link name="link1"/>
      <link name="tool"/>
      <joint name="joint1" type="revolute">
        <parent link="base"/>
        <child link="link1"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14159" upper="3.14159" effort="100" velocity="10"/>
      </joint>
      <joint name="joint2" type="revolute">
        <parent link="link1"/>
        <child link="tool"/>
        <origin xyz="0.5 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14159" upper="3.14159" effort="100" velocity="10"/>
      </joint>
    </robot>
    """
    chain = pk.build_serial_chain_from_urdf(urdf, "tool", root_link_name="base")
    target_q = torch.tensor([[0.6, -0.4], [-0.3, 0.7]])
    target_transform = chain.forward_kinematics(target_q).get_matrix()
    target_pos = target_transform[:, :3, 3]
    target_quat = ik_module.quat_from_matrix(target_transform[:, :3, :3])
    lower = torch.tensor([-3.14159, -3.14159])
    upper = torch.tensor([3.14159, 3.14159])
    options = ik_module.ProjectedIKOptions(
        position_tolerance=1.0e-3,
        orientation_tolerance=1.0e-3,
        orientation_weight=1.0,
        adaptive_damping=True,
        damping_min=1.0e-4,
        damping_max=0.1,
        singular_value_threshold=0.05,
    )
    generator = torch.Generator()
    generator.manual_seed(7)
    seeds = ik_module.make_seed_candidates(
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        lower,
        upper,
        num_random=4,
        generator=generator,
    )

    result = ik_module.solve_multi_start(
        chain,
        target_pos,
        target_quat,
        seeds,
        torch.zeros(2, 2),
        lower,
        upper,
        torch.zeros(2, 2),
        options,
        max_iterations=80,
        full_step_gain=0.8,
        full_max_joint_step=0.4,
        seed_travel_weight=0.0,
    )

    assert bool(result.success.all())
    assert result.position_error.max() <= options.position_tolerance
    assert result.orientation_error.max() <= options.orientation_tolerance


def test_multi_start_ik_returns_best_effort_for_unreachable_pose(ik_module):
    pk = pytest.importorskip("pytorch_kinematics")
    urdf = """
    <robot name="two_link_unreachable">
      <link name="base"/>
      <link name="tool"/>
      <joint name="joint1" type="revolute">
        <parent link="base"/>
        <child link="tool"/>
        <origin xyz="0.5 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14159" upper="3.14159" effort="100" velocity="10"/>
      </joint>
    </robot>
    """
    chain = pk.build_serial_chain_from_urdf(urdf, "tool", root_link_name="base")
    lower = torch.tensor([-3.14159])
    upper = torch.tensor([3.14159])
    options = ik_module.ProjectedIKOptions(position_tolerance=1.0e-3, orientation_tolerance=1.0e-3)
    seeds = torch.zeros(1, 1, 1)
    target_pos = torch.tensor([[10.0, 10.0, 0.0]])
    target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    result = ik_module.solve_multi_start(
        chain,
        target_pos,
        target_quat,
        seeds,
        torch.zeros(1, 1),
        lower,
        upper,
        torch.zeros(1, 1),
        options,
        max_iterations=20,
        full_step_gain=0.7,
        full_max_joint_step=0.3,
        seed_travel_weight=0.0,
    )

    assert not bool(result.success[0])
    assert torch.isfinite(result.joint_pos).all()
    assert torch.isfinite(result.position_error).all()
