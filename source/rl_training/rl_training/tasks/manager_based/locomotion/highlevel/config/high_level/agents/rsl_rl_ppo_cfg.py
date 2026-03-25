# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class HighLevelRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "high_level_rough"
    empirical_normalization = False
    clip_actions = 100
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HighLevelFlatPPORunnerCfg(HighLevelRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "high_level_flat"

@configclass
class HighLevelNavFlatPPORunnerCfg(HighLevelRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "high_level_nav_flat"

@configclass
class HighLevelNavFlatTeacherPPORunnerCfg(HighLevelRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 3000
        self.experiment_name = "high_level_nav_flat_teacher"

        self.obs_groups = {
            "policy": ["teacher"],
            "critic": ["critic"],
        }


# agents/rsl_rl_distillation_cfg.py  ← 学生蒸馏配置
from isaaclab_rl.rsl_rl import (
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlDistillationAlgorithmCfg,
)

@configclass
class HighLevelNavDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations    = 1500
    experiment_name   = "high_level_nav_flat_student"

    # ★ 核心：学生 policy 组 = 本体 + 图像；teacher 组 = 教师特权观测
    obs_groups = {
        "policy":  ["policy"],            # 学生 Actor 的输入
        "teacher": ["teacher"],           # 教师 Actor 的输入（加载冻结权重）
        "critic":  ["critic"],
    }

    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std          = 0.5,
        student_hidden_dims     = [512, 256, 128],
        teacher_hidden_dims     = [512, 256, 128],  # 需与教师训练时一致
        activation              = "elu",
        student_obs_normalization = True,
        teacher_obs_normalization = True,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs = 5,
        learning_rate       = 1.0e-3,
        gradient_length     = 1,      # BPTT 长度（若用 RNN 可调大）
        loss_type           = "mse",  # BC loss: MSE(student_action, teacher_action)
    )