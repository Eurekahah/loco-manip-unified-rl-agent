# 下个 session 的开工 prompt（可直接整段复制）

> 说明：这是给"下一个 AI session"用的交接 prompt。历史版本在
> `codex/docs-next-session @ bf9c5d3`。写完新版本就把这一节整体替换掉。

```text
仓库：D:\nvidia-isaac-sim\loco-manip-unified-rl-agent（Isaac Lab 5.1 + rsl_rl 的轮腿+Piper 机械臂 RL 训练库）
python：C:\Users\autolab\miniconda3\envs\env_isaac_lab\python.exe（跑 Isaac 脚本前设 $env:PYTHONIOENCODING='utf-8'，
        都必须 --headless；沙箱里 .git 只读，git 命令要 -c safe.directory=D:/nvidia-isaac-sim/loco-manip-unified-rl-agent
        并需要 escalate 才能 checkout/commit/push；`git checkout` 可能被"stat-dirty 的假 M"挡住，
        确认内容一致（git hash-object == rev-parse HEAD:path）后用 `git checkout -f`）

【当前状态】main = e78d479（**低层 + 高层都已并入 main，P0 已清空**）：
  7ff5b86 fix(lowlevel): root_height 专项（低位锚点 / height_range 0.55 / 扰动课程 / 稳态指标）
  129848e merge(highlevel): P0-1 第一步 —— replay 布局 / L2 / ll_command / ee_command / history
  af4602d merge(highlevel): P0-1 第二步 —— ⑦ checkpoint 路径参数化 + ⑧ 懒加载低层 cfg
  07601e9 merge(highlevel): P0-1 第三步 —— ⑤ 的 R1（抽 LowLevelPolicyActionBase + 迁移 nav）
  30d5411 / 0772757 docs: 删掉分支带来的旧 docs/review/*.md（含 Windows 大小写冲突那个）
  498e847 feat(tool): P0-2 部署态导出固化（默认 exported_deploy/ + train 收尾提示 + 训练说明）
  e78d479 docs: P0 收尾（TODO 清空 P0 / DONE 第五节 / DEFECT_LOG DEF-018·019）
  ⇒ main 上 4 个低层任务 + 4 个高层任务 2 iter 全部 EXIT=0（高层 reward 1.11/1.28/0.15/10.25）。
  ⇒ 合并冲突的两处解法见 DEFECT_LOG_zh.md DEF-018（大小写路径）/ DEF-019（⑦⑧ × R1）。

【已验证结果】4096 envs、iter=14000 同口径对比：
  旧 2026-09-19_09-02-50（无课程）: root_height 0.361、bad_orientation 0.010、ep_len 805、reward 15.4
  新 2026-09-20_00-50-31（本代码）  : root_height 0.122、bad_orientation 0.010、ep_len 883、reward 22.7
  部署态策略已导出（最新 checkpoint model_19999.pt）：
  logs/rsl_rl/history_adaptation/2026-09-20_00-50-31/exported_deploy/{policy.pt, policy.onnx, policy_layout.json}
  （TorchScript 自检 0.000e+00；ONNX 相对误差 1.9e-07；注意 <run>/exported/policy.pt 是 actor-only，别用）

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

【本 session 已完成 P0；下个 session 建议进 P1（训练质量）】
  P1-1 训练稳定性：mean_noise_std 1.0→~1.49、error_vel_xy 0.38→~0.89 的长期退化（要 A/B）。
  P1-2 s3 阶段臂扰动鲁棒性：root_height_below_minimum 在 s3 后 0.09~0.15（s0~s2 只有 0.01~0.04）。
  P1-3 低层 known_issues 剩余条目（①⑧⑩⑪⑫⑬⑭⑮⑱，清单见 TODO_zh.md P1）。
  之后：P2（高层迁移到基类 + 工程债）、P3（回归矩阵脚本化 / summarize_run.py / 文档收尾）。

【命令备忘】
  # 训练（低层主线）
  python scripts/reinforcement_learning/rsl_rl/train.py \
      --task History-Adaptation-Deeprobotics-M20-v0 --headless --num_envs 4096
  # 课程阶段对应迭代数（num_steps_per_env=24）：s0 <1042、s1 <2083、s2 <3125、s3 之后
  # 冒烟回归（每个改动都要跑；退出码用 `cmd *> log; $LASTEXITCODE`）
  python scripts/reinforcement_learning/rsl_rl/train.py --task <task> --headless --num_envs 64 --max_iterations 2
  # 导出部署态策略（默认写 <run>/exported_deploy/：policy.pt + policy.onnx + policy_layout.json）
  python scripts/reinforcement_learning/rsl_rl/export_deploy_policy.py \
      --run logs/rsl_rl/history_adaptation/<run> --checkpoint model_19999.pt   # --no-onnx / --opset 可选
  # 诊断探针
  python scripts/reinforcement_learning/rsl_rl/probe_root_height_termination.py \
      --task History-Adaptation-Deeprobotics-M20-v0 --headless --num_envs 512 --steps 1000 \
      --keep_push --action_noise_std 1.0 --freeze_ee_preset {none|default|low} --policy <policy.pt>
  python scripts/reinforcement_learning/rsl_rl/probe_ee_curriculum.py \
      --task Flat-Deeprobotics-M20-Piper-WBC-v0 --headless --num_envs 16 --steps 150

【踩坑备忘（累计）】
  * **Windows 大小写不敏感**：合并"两边各自新增、只差大小写"的文件（本轮是
    next_session_prompt.md vs NEXT_SESSION_PROMPT.md）时，git 会把它当两个路径 ——
    别用 `git commit -- <path>`（会解析到另一个文件、提交成"内容替换"），
    先 `git rm --cached` 清掉多余那份、`git add -A` 再提交。
  * 合并 ⑦⑧（横切所有 action term 的载入段）与 R1（纵切这批 `__init__` 的骨架）必然冲突：
    解法是"并集 + 保留基类写法 + 基类里调 load_low_level_policy"，见 DEF-019。
  * `export_deploy_policy.py` 默认输出目录已改成 `<run>/exported_deploy`；
    `play.py` 的 `<run>/exported/policy.pt` 是 **actor-only**（115 = 83 + 32 latent，latent 无来源），
    两者别混用（脚本会打印警告/提示）。
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
