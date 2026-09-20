# 下个 session 的开工 prompt（可直接整段复制）

> 说明：这是给"下一个 AI session"用的交接 prompt。历史版本在
> `codex/docs-next-session @ bf9c5d3`。写完新版本就把这一节整体替换掉。

```text
仓库：D:\nvidia-isaac-sim\loco-manip-unified-rl-agent（Isaac Lab 5.1 + rsl_rl 的轮腿+Piper 机械臂 RL 训练库）
python：C:\Users\autolab\miniconda3\envs\env_isaac_lab\python.exe（跑 Isaac 脚本前设 $env:PYTHONIOENCODING='utf-8'，
        都必须 --headless；沙箱里 .git 只读，git 命令要 -c safe.directory=D:/nvidia-isaac-sim/loco-manip-unified-rl-agent
        并需要 escalate 才能 checkout/commit/push；`git checkout` 可能被"stat-dirty 的假 M"挡住，
        确认内容一致（git hash-object == rev-parse HEAD:path）后用 `git checkout -f`）

【当前状态】**部署基线 = main @ 2d49f47**（低层 + 高层都已并入 main，P0 已清空）：
  这个 commit 就是"现在拿去部署"的代码，对应 run
  logs/rsl_rl/history_adaptation/2026-09-20_00-50-31 的 exported_deploy/*
  （产物 sha256 与"训练代码 vs main"的核对见 DONE_zh.md 第六节 / DEF-022）。
  main 之后还会继续往前走 ⇒ **部署时 checkout 2d49f47，不要用"当时的 main"**。
  之后的提交（按时间）：
  79b1626 docs 清悬空引用 / 708ca53 ONNX 导出（DEF-020）/ 7458672 部署文档 + probe_deploy_layout（DEF-021）
  / 2d49f47 部署交接（DEF-021 收尾）
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
  P1-1 **已归因（2026-09-20，DEF-023）**：mean_noise_std 不是发散而是**有界平台**（新 run s0~s3
      0.973/1.009/1.132/1.473，旧 run 1.236→1.505，Δ末 -0.035）⇒ 机制是"log_std 无上界 +
      entropy_coef=0.01 的熵奖励 + adaptive 调度把 LR 压到 1e-5"；error_vel_xy 的上升**与命令
      课程同形**（旧 run 0.15→0.77 也升）⇒ 是口径产物，不是退化（新 run 末 1000 reward 23.6 vs 17.8、
      ep_len 917 vs 768、合计摔倒 0.122 vs 0.331）。**待做**：配置侧 A/B（log_std 上界 / entropy_coef
      降到 0.005~0.002，~5k iter 即可看出平台），验收口径已改成"平台 ≤1.2 + 固定命令 eval"。
  P1-2 s3 阶段臂扰动鲁棒性：root_height_below_minimum s0~s2 = 0.021~0.027 → **s3 = 0.118**，
      同期 ee_pose/orientation_error 0.32 → 0.85、height_error_bias_steady 全程仅 1.3~1.8 cm
      ⇒ 剩余摔倒是"臂摆动时倾覆"（bad_orientation_2 在 s3 反而降到 0.014，所以要看合计 0.1325）。
  P1-3 低层 known_issues 剩余条目（①⑧⑩⑪⑫⑬⑭⑮⑱，清单见 TODO_zh.md P1）。
  之后：P2（高层迁移到基类 + 工程债）、P3（回归矩阵脚本化 / summarize_run.py / 文档收尾）。
  ※ P3 的 summarize_run.py **已做完**（本轮）：阶段均值 + 采样网格 + 两 run 对比 + `--derive`。

【命令备忘】
  # 部署（sim2sim/MuJoCo → sim2real）先看 docs/deploy_sim2sim_sim2real_zh.md（DEF-021）
  # 在部署机上先跑一次拿"权威布局"（关节序/默认角/增益/观测 scale）：
  python scripts/reinforcement_learning/rsl_rl/probe_deploy_layout.py \
      --task History-Adaptation-Deeprobotics-M20-v0 --headless --num_envs 2
  # 训练（低层主线）
  python scripts/reinforcement_learning/rsl_rl/train.py \
      --task History-Adaptation-Deeprobotics-M20-v0 --headless --num_envs 4096
  # 课程阶段对应迭代数（num_steps_per_env=24）：s0 <1042、s1 <2083、s2 <3125、s3 之后
  # 冒烟回归（每个改动都要跑；退出码用 `cmd *> log; $LASTEXITCODE`）
  python scripts/reinforcement_learning/rsl_rl/train.py --task <task> --headless --num_envs 64 --max_iterations 2
  # 导出部署态策略（默认写 <run>/exported_deploy/：policy.pt + policy.onnx + policy_layout.json）
  python scripts/reinforcement_learning/rsl_rl/export_deploy_policy.py \
      --run logs/rsl_rl/history_adaptation/<run> --checkpoint model_19999.pt   # --no-onnx / --opset 可选
  # 训练曲线分析（不启动 Isaac；首次解析 30~50 s，之后走 <run>/.summary_cache.npz，<1 s）
  python scripts/reinforcement_learning/rsl_rl/summarize_run.py \
      --run logs/rsl_rl/history_adaptation/2026-09-20_00-50-31 --list-tags      # 先看有哪些 tag
  python scripts/reinforcement_learning/rsl_rl/summarize_run.py \
      --run logs/rsl_rl/history_adaptation/2026-09-20_00-50-31 \
      --baseline logs/rsl_rl/history_adaptation/2026-09-19_09-02-50 \
      --derive "合计摔倒=Episode_Termination/bad_orientation_2+Episode_Termination/root_height_below_minimum" \
      --tags mean_noise_std --tags error_vel_xy --tags 合计摔倒 --tags Train/mean_reward
  # 诊断探针
  python scripts/reinforcement_learning/rsl_rl/probe_root_height_termination.py \
      --task History-Adaptation-Deeprobotics-M20-v0 --headless --num_envs 512 --steps 1000 \
      --keep_push --action_noise_std 1.0 --freeze_ee_preset {none|default|low} --policy <policy.pt>
  python scripts/reinforcement_learning/rsl_rl/probe_ee_curriculum.py \
      --task Flat-Deeprobotics-M20-Piper-WBC-v0 --headless --num_envs 16 --steps 150

【踩坑备忘（累计）】
  * **本文件里的 main 哈希会滞后**（人肉回填，main 一动就旧）：要"拿去部署的那份代码"一律看
    docs/review/DONE_zh.md 第六节「部署基线」（现在 = 2d49f47 + run 2026-09-20_00-50-31 的导出物指纹）。
  * run 目录里有 rsl_rl 自动 dump 的 `<run>/git/loco-manip-unified-rl-agent.diff`（训练启动时的
    branch + 工作区改动）——这是"这个 checkpoint 是哪份代码训的"的**唯一证据**，迁移/重训前先看它。
    实测 2026-09-20_00-50-31 = 分支 codex/ll-height-stability @ 96e1b66 + 注释掉死代码
    FKReachableEECommand；与 main 的差异只有占位符清理/启动自检（无动力学变化），见 DEF-022。
  * `Metrics/*` 全是"复位那一刻"的均值，单点抖动很大（同一阶段逐点能差 0.2）——判趋势用
    summarize_run.py 的**阶段均值**，不要看单迭代数字。
  * **关节顺序有三套**（部署最容易错）：① 动作序 = 12 腿(fl,fr,hl,hr) + 4 轮；
    ② articulation 原生序（观测 joint_pos/joint_vel 的 24 维）= 四个 hipx → arm1 → 四个 hipy
    → arm2 → 四个 knee → arm3 → 四个 wheel(15..18) → arm4-6 → 夹爪；③ MuJoCo MJCF 序 =
    每腿 hipx/hipy/knee/wheel 连续。一律按关节名映射。
  * history 的 70 维用**原始值**（不乘 scale），policy_obs 的同名项乘了 scale —— 两者不能复用同一个向量。
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
