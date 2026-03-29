# agents/rsl_rl_distillation_cfg.py  ← 学生蒸馏配置
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlDistillationAlgorithmCfg,
)

@configclass
class HighLevelNavFlatDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations    = 1500
    save_interval = 100

    experiment_name   = "high_level_nav_flat_teacher"

    # ★ 核心：学生 policy 组 = 本体 + 图像；teacher 组 = 教师特权观测
    obs_groups = {
        "policy":  ["policy"],            # 学生 Actor 的输入
        "teacher": ["teacher"],           # 教师 Actor 的输入（加载冻结权重）
        "critic":  ["critic"],
    }

    policy = RslRlDistillationStudentTeacherCfg(
        class_name="StudentTeacher",
        init_noise_std          = 0.5,
        student_hidden_dims     = [512, 256, 128],
        teacher_hidden_dims     = [512, 256, 128],  # 需与教师训练时一致
        activation              = "elu",
        student_obs_normalization = False,
        teacher_obs_normalization = False,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs = 5,
        learning_rate       = 1.0e-3,
        gradient_length     = 1,      # BPTT 长度（若用 RNN 可调大）
        loss_type           = "mse",  # BC loss: MSE(student_action, teacher_action)
    )

@configclass
class HighLevelPickFlatDistillationRunnerCfg(HighLevelNavFlatDistillationRunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "high_level_pick_flat_student"

        # ★ 核心：学生 policy 组 = 本体 + 图像；teacher 组 = 教师特权观测
        self.obs_groups = {
            "policy":  ["policy"],            # 学生 Actor 的输入
            "teacher": ["teacher"],           # 教师 Actor 的输入（加载冻结权重）
            "critic":  ["critic"],
        }