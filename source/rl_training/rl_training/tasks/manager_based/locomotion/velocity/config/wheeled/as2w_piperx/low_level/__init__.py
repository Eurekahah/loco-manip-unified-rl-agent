# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause
"""AS2W-PiperX 低层 RL 的 manager 组配置。

每个 manager 组一个文件（交接单 §6.3）：

* :mod:`.actions`       —— 12 腿位置 + 4 轮速度 + IK 臂（0 维），总动作 16；
* :mod:`.commands`      —— ``base_velocity`` / ``ee_pose`` / ``body_pose``；
* :mod:`.observations`  —— policy 83 / critic 86 / history 700 / privileged 54；
* :mod:`.rewards`       —— 只定义启用的 23 项；
* :mod:`.events`        —— startup / reset / interval 三档（含 ``check_policy_layout``）；
* :mod:`.terminations`  —— 4 项；
* :mod:`.curriculum`    —— 14 项；
* :mod:`.env_cfg`       —— 组合以上 + 环境级 ``sim``/``decimation`` 设置。

本文件不放任何逻辑，避免 ``import_packages`` 递归导入时的副作用。
"""