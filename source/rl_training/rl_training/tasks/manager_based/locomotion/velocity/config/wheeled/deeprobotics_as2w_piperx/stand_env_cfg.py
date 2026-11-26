# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""AS2-W X + Piper X 静止站立可视化环境（配合 ``scripts/tools/zero_agent.py``）。

用法::

    python scripts/tools/zero_agent.py --task Visualize-AS2W-PiperX-v0

设计要点（只求"看得清、站得住"，不是训练环境）：

1. 平地、单机器人；无命令 / 无奖励 / 无课程 / 无随机化。
2. 动作只有一路全关节位置目标（``use_default_offset=True``）⇒ 全零动作就是保持
   ``AS2W_PIPERX_STAND_JOINT_POS`` 的站立姿态。
3. 终止只保留 timeout，且 episode 很长，避免观察过程中被 reset 传送回原点。
4. 物理步长 5 ms、决策 20 ms（50 Hz 渲染），与 M20 环境一致。
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp
from rl_training.assets.deeprobotics import DEEPROBOTICS_AS2W_PIPERX_CFG


@configclass
class AS2WPiperXSceneCfg(InteractiveSceneCfg):
    """平地 + 机器人 + 灯光。接触报告由 spawn 的 ``activate_contact_sensors`` 打开。"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = DEEPROBOTICS_AS2W_PIPERX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/Light/Dome",
        spawn=sim_utils.DomeLightCfg(intensity=900.0, color=(1.0, 1.0, 1.0)),
    )
    sun = AssetBaseCfg(
        prim_path="/World/Light/Sun",
        spawn=sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 0.96, 0.9), angle=0.6),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.653, 0.271, 0.271, 0.653)),
    )


@configclass
class ObservationsCfg:
    """最小的 policy 观测组（只为满足 ManagerBasedRLEnv，不参与控制）。"""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """全关节位置目标：零动作 + use_default_offset ⇒ 保持 init_state 的站立角。"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        # IsaacLab 5.1 的 JointActionCfg 要求显式给出名字/正则，None 会在 find_joints 里报错
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip=None,
        preserve_order=True,
    )


@configclass
class RewardsCfg:
    """无奖励：本环境只做可视化检查。"""


@configclass
class TerminationsCfg:
    """只保留超时，方便长时间观察。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class AS2WPiperXStandEnvCfg(ManagerBasedRLEnvCfg):
    scene: AS2WPiperXSceneCfg = AS2WPiperXSceneCfg(num_envs=1, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 3600.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01

        # 默认视角：右前上方看整车 + 机械臂
        self.viewer.eye = (1.70, 1.35, 1.05)
        self.viewer.lookat = (0.03, 0.0, 0.50)
