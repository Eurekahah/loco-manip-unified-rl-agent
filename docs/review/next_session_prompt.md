# 下个 session 的开工 prompt（可直接整段复制）

```text
仓库：D:\nvidia-isaac-sim\loco-manip-unified-rl-agent（Isaac Lab 5.1 + rsl_rl 的轮腿+Piper 机械臂 RL 训练库）
python：C:\Users\autolab\miniconda3\envs\env_isaac_lab\python.exe（跑 Isaac 脚本前设 $env:PYTHONIOENCODING='utf-8'，
        都必须 --headless；沙箱里 .git 只读，git 命令要 -c safe.directory=D:/nvidia-isaac-sim/loco-manip-unified-rl-agent
        并需要 escalate 才能 checkout/commit/push）

【当前状态】main = a855e8f（未被我改过）。所有修复都在这些分支上，都还没合并：
  codex/hl-replay-layout        dc45d0e   # 低层 replay 布局单一来源 + 观测/动作布局断言（清单 ④⑤⑥⑨⑯ + 清单外 A/B）
  codex/hl-replay-l2            1f56b6e   # L2 布局推导（ee_action_dim=-1 默认）+ 按 policy_layout.json 匹配 obs 维度
  codex/ll-keep-ee-goal         1253bba   # WBC 低层恢复保留 ee_goal 观测
  codex/export-deploy-policy    4276970   # export_deploy_policy.py（含 history encoder 的部署态导出）
  codex/hl-fix-ll-command       e064bc6   # ① PreTrainedPickAction 补 ll_command/ll_command_w + 奖励项改 ll_command_world
  codex/hl-fix-ee-command       7a22759   # ② IK 目标写 pose_command_b（+start/end_b）、③ flat 的 ee_goal 用 root 系
  codex/ll-history-flat-eegoal  45f9e74   # 训练用配置（bad_orientation_2=0.8rad、保留 ee_goal）+ 说明 + 导出脚本（已推 GitHub）
  codex/docs-review             1fb76e4   # 两份清单的来源分支，不要合并
高层 P0（①②③）已全部修完并实测通过；高层三个任务 train 2 iter 均 exit 0。
文档在 docs/review/：high_level_todo.md（含修复记录）、known_issues.md、
  history_low_level_policy_todo.md、bad_orientation_analysis_zh.md、progress_summary_zh.md、本文件。

【这次要做的事，按优先级】

1) bad_orientation_2 的课程方案落地（先读 docs/review/bad_orientation_analysis_zh.md）
   - 背景：6~7 月的 run 里该终止只有 0.6~3.3%（episode 满 1000 步），9 月起变成 62~78%。
     归因（已实测）：不是 IK 坏了，而是"臂扰动底盘的能力"变了 ——
     ① 机器人资产 M20_adjusted.usd → M20_Piper_own.usd
     ② piper_arm 执行器 Implicit(40/8) → DelayedPD(300/20)（实测同策略下臂速 1.74 vs 0.95 rad/s）
     ③ EE 参考系 arm_link6+0.135 → gripper_base+0，目标半径收到 0.3~0.52
     ④ 新增 root_height_below_minimum=0.3，而 body_pose.height_range 下界是 0.33（只剩 3cm）
   - 要做：
     a. 先测"默认位姿在 height-invariant 坐标系下的半径"（HeightInvariantEECommandCfg.ranges.p_l 的
        s0 起点必须 ≤ 它），可以写个 probe 在 reset 后打印 pose_end_cart/半径。
     b. 在 WBCCurriculumCfg 里加 EE 目标课程 s0→s3（用现成的 mdp.modify_term_cfg + mdp.override_value，
        照 body_pose_height_range_s2 的写法）：s0 = EE 目标锁在默认位姿 → s1 放开位置半径 →
        s2 放开姿态 ±10° → s3 全范围。
     c. 顺手修 HeightInvariantEECommand._update_command：位置有 T_traj 插值但姿态直接跳终点，
        改成像样的 slerp（math_utils.quat_slerp）。
     d. 可选：把 piper_arm 刚度做成课程（终值 300/20，前段 40/8），并核对 assets/deeprobotics.py 里
        你自己注释的目标区间（"DelayedPD 在 60~100 / 0~20"，300 偏高）。
     e. 验收：Episode_Termination/bad_orientation_2 在 s0 阶段应接近 0；Metrics/base_velocity/error_vel_xy
        要随迭代下降（7500 那个 run 是反向上升的）；Policy/mean_noise_std 不应无限上涨。
   - 可选对照实验（约 30 min/组）：同代码跑"EE 课程开 / EE 目标固定默认位姿"两组各 300 iter，对比该终止曲线。

2) history 低层策略的回放支持（读 docs/review/history_low_level_policy_todo.md）
   - 导出侧已完成（policy.pt 已是 forward(policy_obs, history_flat)），差回放侧：
     10 步环形缓冲（reset 时清零）、按 policy_layout.json 决定单/双输入调用、
     history 单步向量里的 last_action 必须用**低层 16 维动作**而不是高层 action_manager。
   - 坑：checkpoint_dims() 读 actor.0.weight 对 history 策略会得到 108（含 latent），必须以
     policy_layout.json 的 policy_obs_dim 为准。

3) P1 收尾（都不大）
   - ⑦ checkpoint 路径参数化：目前高层 cfg 里是 _LOW_LEVEL_WBC_POLICY 一处常量，
     改成环境变量/命令行，并在加载失败时给明确报错。
   - ⑧ 去掉 high_level_env_cfg.py:30 的模块级 LOW_LEVEL_ENV_CFG 实例化，并修 render_interval
     (=低层 decimation=4 < 高层 decimation=40) 触发的多次渲染警告。
   - ⑤ 的 R1 步：抽 LowLevelPolicyActionBase，把 nav / openvla / pre_trained_policy 三个
     还没迁移的 action term 纳入（顺带修：nav 的 lateral/angular_velocity_penalty 读了
     action_term.ll_command，但 PreTrainedNavAction 根本没有这个属性）。
   - 低层 known_issues ⑤⑥⑦：body_names="" 占位符、disable_zero_weight_rewards 对 None 不健壮、
     feet_distance_y_exp 传了类型对象。
   - ⑯ 剩下：低层 env 自己也加布局打印/断言；velocity/mdp/observations.py::joint_pos_rel_without_wheel
     补上"列序 == 原生序"的断言。

【工作方式（沿用上次约定）】
  - 一个问题一个分支（codex/ 前缀），改完在该分支上测试并贴**实测数据**，
    然后停下来等我确认再合并；不要自动合并 main。
  - 测试至少包含：scripts/reinforcement_learning/rsl_rl/train.py --task <相关任务> --headless
    --num_envs 64 --max_iterations 2（exit 0 + 打印关键维度）；坐标系类修复要有"与独立复算一致"的对比。
  - 每修完一条，把 docs/review 里对应条目改成 [已修] 并注明 commit（文档也走分支）。
  - 如果某条有多种合理改法，先给我 2~3 个选项 + 你的推荐再动手。
  - 环境备注：一个进程只建一个 Isaac env（第二个会卡死）；probe 脚本里注意别用 exit()；
    .git 只读要 escalate。
