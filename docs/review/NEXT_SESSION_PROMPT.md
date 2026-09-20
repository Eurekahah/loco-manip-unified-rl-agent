# 下个 session 的开工 prompt（可直接整段复制）

> 说明：这是给"下一个 AI session"用的交接 prompt。历史版本在
> `codex/docs-next-session @ bf9c5d3`。写完新版本就把这一节整体替换掉。

```text
仓库：D:\nvidia-isaac-sim\loco-manip-unified-rl-agent（Isaac Lab 5.1 + rsl_rl 的轮腿+Piper 机械臂 RL 训练库）
python：C:\Users\autolab\miniconda3\envs\env_isaac_lab\python.exe（跑 Isaac 脚本前设 $env:PYTHONIOENCODING='utf-8'，
        都必须 --headless；沙箱里 .git 只读，git 命令要 -c safe.directory=D:/nvidia-isaac-sim/loco-manip-unified-rl-agent
        并需要 escalate 才能 checkout/commit/push；`git checkout` 可能被"stat-dirty 的假 M"挡住，
        确认内容一致（git hash-object == rev-parse HEAD:path）后用 `git checkout -f`）

【当前状态】main = 7ff5b86（**低层内容已并入 main**）：
  26584e9 merge: 低层训练配置（bad_orientation_2=0.8rad + 保留 ee_goal）+ 训练说明 + 导出脚本
  469fbd4 feat(lowlevel): EE 目标课程 s0-s3（低位锚点 + 区间阶梯）+ 姿态 slerp + 两个探针
  fef34a7 fix(lowlevel): known_issues ⑤⑥⑦ + ⑯（占位符→None / 奖励清理 / 类型参数 / 布局断言）
  7ff5b86 fix(lowlevel): root_height 专项（低位锚点 / height_range 0.55 / 扰动课程 / 稳态指标）
  ⇒ 现在 main 上 4 个低层任务 2 iter 全部 EXIT=0，可以直接训练。

【已验证结果】4096 envs、iter=14000 同口径对比：
  旧 2026-09-19_09-02-50（无课程）: root_height 0.361、bad_orientation 0.010、ep_len 805、reward 15.4
  新 2026-09-20_00-50-31（本代码）  : root_height 0.122、bad_orientation 0.010、ep_len 883、reward 22.7
  部署态策略已导出：logs/rsl_rl/history_adaptation/2026-09-20_00-50-31/exported_deploy/
  （自检 0.000e+00；注意同目录 exported/policy.pt 是 actor-only，别用）

【文档约定（新）】docs/review/ 只保留：
  TODO_zh.md             # 唯一未完成清单（P0→P3，带日期）
  DONE_zh.md             # 已完成（带日期 + commit + 验收数字）
  DEFECT_LOG_zh.md       # 每个缺陷/特性：现象→根因→修正→结果（DEF-0xx）
  NEXT_SESSION_PROMPT.md # 本文件
  templates/DOC_TEMPLATE_zh.md、templates/DEFECT_ENTRY_TEMPLATE_zh.md
  每次改动**必须**：更新 TODO/DONE 的"更新记录"（加日期）、给新缺陷/特性在 DEFECT_LOG 里加一条。
  旧的分主题文档（high_level_todo / known_issues / history_low_level_policy_todo /
  bad_orientation_analysis_zh / progress_summary_zh / todo_master）已并入上面三份，
  原文在分支上（见 TODO_zh.md 末尾"旧文档去哪了"）。

【本 session 建议按 TODO_zh.md 的 P0 开始】
  P0-1 高层链合并（main 上的高层任务仍是旧代码，会崩）：
       hl-replay-layout(dc45d0e) → hl-replay-l2(1f56b6e) → hl-fix-ll-command(e064bc6)
       → hl-fix-ee-command(0389253) → hl-replay-history(47519b7) → hl-ckpt-params(e9edc34)
       → hl-replay-base-class(7ec8b4c)；已知冲突点与解法写在 TODO_zh.md P0-1；
       合并时删掉它们带来的旧 docs/review/*.md（会与新文档重复）。
       验收：4 个高层任务 --headless --num_envs 64 --max_iterations 2 全 exit 0。
  P0-2 把"训练完必须用 export_deploy_policy.py 导出部署态策略"写进训练流程/说明。
  之后进 P1（训练稳定性：noise_std/error_vel_xy）、P2（高层迁移与工程债）、P3（工具与文档）。

【命令备忘】
  # 训练（低层主线）
  python scripts/reinforcement_learning/rsl_rl/train.py \
      --task History-Adaptation-Deeprobotics-M20-v0 --headless --num_envs 4096
  # 课程阶段对应迭代数（num_steps_per_env=24）：s0 <1042、s1 <2083、s2 <3125、s3 之后
  # 冒烟回归（每个改动都要跑；退出码用 `cmd *> log; $LASTEXITCODE`）
  python scripts/reinforcement_learning/rsl_rl/train.py --task <task> --headless --num_envs 64 --max_iterations 2
  # 导出部署态策略
  python scripts/reinforcement_learning/rsl_rl/export_deploy_policy.py \
      --run logs/rsl_rl/history_adaptation/<run> --checkpoint model_15000.pt \
      --out_dir logs/rsl_rl/history_adaptation/<run>/exported_deploy
  # 诊断探针
  python scripts/reinforcement_learning/rsl_rl/probe_root_height_termination.py \
      --task History-Adaptation-Deeprobotics-M20-v0 --headless --num_envs 512 --steps 1000 \
      --keep_push --action_noise_std 1.0 --freeze_ee_preset {none|default|low} --policy <policy.pt>
  python scripts/reinforcement_learning/rsl_rl/probe_ee_curriculum.py \
      --task Flat-Deeprobotics-M20-Piper-WBC-v0 --headless --num_envs 16 --steps 150

【踩坑备忘（累计）】
  * 一个进程只建一个 Isaac env；probe 用 os._exit(0) 退出。
  * 课程只在 **env reset** 时推进（curriculum_manager.compute）；若关掉终止/超时做实验，
    课程不会动 —— 要手动把终态区间写进 cfg。
  * `env.event_manager.get_term_cfg(name)` 对已置 None 的事件抛 ValueError（先查 active_terms）。
  * `episode_length_buf == 0` 在复位后的整步内都为真；判"刚复位"要用"相比上一次 tick 变小"。
  * IsaacLab `CircularBuffer.buffer` 是 `[最旧..最新]`；reset 后第一次 push 会填满整窗。
  * 终止项要看**合计**（bad_orientation_2 + root_height_below_minimum），否则调一个阈值只是搬家。
  * `Metrics/body_pose/height_error_bias` 会被塌陷瞬间拉偏；稳态看 `height_error_bias_steady`。
  * tensorboard events 文件很大（57 MB），EventAccumulator 读一次 30~60 s；
    CommandTerm 的 metrics 是"复位那一刻"的均值（抖动大）。
  * 文档里指向旧文档的路径还没全部更新（见 TODO_zh.md P3"文档收尾"）。
```
