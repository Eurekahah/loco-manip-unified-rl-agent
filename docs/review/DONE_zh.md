# 已完成清单（DONE）—— 按主题

**文档职责**：记录"已经做完并且有实测验收"的事情（含 commit 与关键数字）。
未完成的在 `TODO_zh.md`；每条缺陷/特性的现象→原因→修正→结果在 `DEFECT_LOG_zh.md`。

**维护约定**：见 `templates/DOC_TEMPLATE_zh.md`。完成任务时从 TODO 迁到这里，
**保留日期与 commit**；只写结论与验收数字，过程细节写进 DEFECT_LOG。

## 更新记录

| 日期 | 更新内容 | 相关 commit / 分支 |
|---|---|---|
| 2026-09-20 | 初版：合并 6 份旧文档里的"已修"条目；记录低层内容并入 main | `main @ 7ff5b86` |
| 2026-09-20 | 新增第五节：高层链并入 `main`（4 个高层任务 2 iter 全 EXIT=0）+ 导出部署态策略固化成流程；第二节标题去掉"未合并 main" | `codex/hl-merge-p0`（`129848e`/`af4602d`/`07601e9`/`0772757`） |
| 2026-09-20 | 新增第六节 **部署基线**：`main @ 2d49f47` = 部署口径代码，对应 run `2026-09-20_00-50-31` 的 `exported_deploy/*`（含 sha256 与"训练代码 vs main"的差异核对） | `2d49f47` |

---

## 一、低层训练（本轮主线）

| 日期 | 内容 | 关键实测 | commit |
|---|---|---|---|
| 2026-09-20 | 低层内容并入 `main`（训练配置 + EE 课程 + slerp + root_height 专项 + known_issues ⑤⑥⑦⑯ + 导出脚本 + 训练说明） | 4 个低层任务 `--num_envs 64 --max_iterations 2` 全 EXIT=0；`env.yaml` 里 `ee_goal_stages`/`disturbance_ramp`/`steady_error_clip=0.15`/`limit_angle=0.8` 均生效 | `26584e9`(merge) `469fbd4` `fef34a7` `7ff5b86` |
| 2026-09-20 | `bad_orientation_2` 改成旋转不变量 + 阈值 0.8 rad；**保留** `ee_goal`（policy 76→83） | History-Adaptation 2 iter exit 0；`policy 83 / history 700 / privileged 89` | `45f9e74` / `26584e9` |
| 2026-09-19 | **EE 目标课程**：s0 锚点 + s1/s2/s3 区间阶梯（`mdp.apply_range_stages`）；EE 姿态命令改 slerp | 课程 4 阶段自检全过；slerp 与官方单样本最大分量偏差 1.19e-07；姿态单步跳变 155.7°→3.12° | `469fbd4`（原 `9ccb8ec`/`96e1b66`） |
| 2026-09-20 | **root_height 专项**：s0 锚点从"默认（举起）位姿"改成**低位锚点**、`body_pose.height_range` 上界 0.60→0.55、扰动课程（push/外力 30%→100%，25k 步）、新增 `height_error_bias_steady` | 机制对照：锁低位 **1.0%** vs 锁默认位姿 **55.5%** vs 无课程 **25.8%**（20s 高度终止率）；阈值反事实证明"单独降阈值无效"（0.30→0.26 只 25.8%→24.4%） | `7ff5b86`（原 `96e1b66`） |
| 2026-09-20 | **训练结果**：run `2026-09-20_00-50-31`（4096 envs，15k iter，本分支代码） | `root_height_below_minimum` 0.361→**0.122**、合计摔倒 0.371→**0.132**、ep_len 805→**883**、reward 15.4→**22.7**、`height_error_bias_steady` 1.1 cm（iter=14000 对比旧 run） | 代码 `96e1b66` |
| 2026-09-20 | 部署态导出（`model_15000.pt`）—— actor-only 陷阱已避开 | `exported_deploy/{policy.pt,policy_layout.json}`，自检 scripted↔eager 与 `act_inference` 均 **0.000e+00**；layout `history/83/10×70/latent32/action16` | 脚本 `4276970` |
| 2026-09-19 | 低层 `known_issues` ⑤⑥⑦ + ⑯ 剩余：占位符 → `None`、`disable_zero_weight_rewards` 对 None 容错 + `term_names`、`stance_width` 改数值、启动期布局打印/断言（`mdp.check_policy_layout`）、`joint_pos_rel_without_wheel` 列序断言 | Arm/WBC/History 三任务 2 iter EXIT=0；启动打印 `policy 86 = … + actions22`、`history 700` | `fef34a7`（原 `6d22006`） |
| 2026-09-18 | 手臂奖励坐标系修复（root 系统一）+ EE body 索引缓存；privileged 观测缓存 reset 感知；手臂隔离 env | 见 `DEFECT_LOG_zh.md` DEF-017 | `b3496a5` / `905c2df` |

## 二、高层 replay（2026-09-20 起**已并入 main**，合并过程见第五节）

| 日期 | 内容 | 关键实测 | commit |
|---|---|---|---|
| 2026-09-19 | **①** `PreTrainedPickAction` 补 `ll_command`/`ll_command_w`；8 处世界系奖励项改用 `ll_command_world()` | 三任务 2 iter EXIT=0、reward 与改前逐位一致；坐标系与独立复算 0.000e+00 | `e064bc6` |
| 2026-09-19 | **②** IK 目标写 `pose_command_b`（并同步 `pose_start_b/pose_end_b`）；**③** flat 的 `ee_goal` 改 root 系 | `\|pose_command_b − 高层目标\| = 0.000e+00`；三任务 2 iter EXIT=0 | `7a22759` |
| 2026-09-19 | **清单外 A**：`actions` 观测少 7 维（IK 改 CommandManager 驱动后）→ 按布局拼接；**清单外 B**：轮关节掩码按**列**置零 | flat 83=83、WBC/teleop 86=86；正确列下标 `[12,13,14,15]` | `dc45d0e` |
| 2026-09-19 | **④** 低层动作 scale 不再硬编码（实测原写法是**死代码**：`JointAction.__init__` 已把 scale 编译成内部张量）；**⑥** 观测组模板 deepcopy，不再就地改 cfg；**⑨** `ee_pose_commands` → `ee_goal` | 三任务 2 iter EXIT=0 | `dc45d0e` |
| 2026-09-19 | **L2 布局推导**（`ee_action_dim=-1` 默认从低层 cfg 推导）+ 按 `policy_layout.json` 匹配 obs 维度（`ee_goal` 自动取舍） | teleop 83=83、WBC 83=83、flat(L1) 83=83 | `1f56b6e` |
| 2026-09-19 | **history 低层策略回放支持**：10 步窗口（复用 IsaacLab `CircularBuffer`）+ 单/双输入调用 + `last_action` 用低层 16 维动作 + 复位判定改"跳变检测" | 单步 vs 训练函数独立复算 `0.000e+00`；整窗顺序 40 次 tick `0.000e+00`；teleop/Pick-WBC 用 history 策略 2 iter EXIT=0 | `0d37c99` |
| 2026-09-19 | **⑦** checkpoint 路径参数化（环境变量 `RL_TRAINING_LOW_LEVEL_POLICY_<KEY>` / hydra）+ 统一加载报错；**⑧** `LOW_LEVEL_ENV_CFG` 改懒加载 + `render_interval` 修成不重复渲染 | 无效路径给出三种修法、有效覆盖 83=83；`Rendering step-size` 0.02→0.2、警告消失、reward 不回归 | `1b6d5c8` |
| 2026-09-19 | **⑤ 的 R1**：抽 `LowLevelPolicyActionBase`（载入/布局/观测/history/tick/`ll_command` 单一来源）+ 迁移 nav；顺带修 nav 奖励项读不存在的 `ll_command` | nav 2 iter EXIT=0、`低层 obs 69 = checkpoint 69`；`lateral_velocity_penalty` 从 AttributeError 变成 -0.0750/-0.0275 | `2c5a85a` |
| 2026-09-19 | 探针工具：`probe_ee_default_pose.py`、`probe_ee_curriculum.py`、`probe_history_window.py`、`probe_root_height_termination.py` | 见各自 commit | `9ccb8ec` / `0d37c99` / `96e1b66` |

## 三、更早的修复（来自原 `known_issues.md` 一、1-4/16/17 与三、）

| 日期 | 内容 | 关键实测 | commit |
|---|---|---|---|
| 2026-09-18 | `joint_pos_rel_without_wheel` 索引空间混用（清错关节）→ policy 的 `joint_pos/joint_vel` 回 `[".*"]` 原生序（24 维） | 被清掉的关节从 `hr_wheel + arm_joint1/2/3` 纠正为 `fl/fr/hl/hr_wheel` | `b3496a5`（另见 replay 侧 `dc45d0e`、断言 `6d22006`） |
| 2026-09-18 | privileged 观测缓存对 reset 模式随机化不敏感 → 抽出 `_PrivilegedCachedTerm`（`update_on_reset` 开关）；顺带发现 `ObsTerm.params` 会原样透传 | @4096 envs 每次 reset 净增 1.92 ms（摊销 0.0019 ms/step），缓存路径不变 | `905c2df` |
| 2026-09-18 | 手臂奖励坐标系不一致（root vs world）→ 统一 root 系 + EE body 索引缓存 | 旧算法把奖励算成 0.0000，正确值 0.0010 | `b3496a5` |
| 2026-09-18 | legacy 手臂奖励权重不可用（零动作稳态 `Στ²≈2.4e4`）→ `torque=-1e-5 / vel=-1e-3 / acc=-1e-8` | 总奖励从 -86.2 回到 +0.53 | `b3496a5` |
| 2026-09-18 | `[已确认：不是 bug]`"臂/夹爪持续接触 90 N"是归因错误（sensor 与 articulation body 顺序不同） | 按 `sensor.body_names` 归因：轮子 85~115 N（地面支撑），臂/夹爪 1.3~4.1 N | `1fb76e4` |

## 四、基础设施

* `export_deploy_policy.py`：把带 history encoder 的策略导出成部署态
  `forward(policy_obs, history_flat)` + `policy_layout.json`（纯 torch，不起 Isaac）。
* `low_level_replay.py`：低层 replay 的单一来源（关节分组/布局/观测组装/启动期断言/
  history 窗口/`ll_command` 辅助）。
* `mdp.check_policy_layout`（低层 startup 事件）：打印观测/动作布局并断言
  "`actions` 槽位宽度 == 动作总维度"。
* `summarize_run.py`（2026-09-20 新增）：把 run 的 tensorboard 标量压成
  "关键指标 × 课程阶段 / × 迭代采样"表，支持两 run 对齐对比；`--derive` 可把终止项
  求和（"合计摔倒"）；首次解析 70 MB 事件文件 30~50 s，之后走 `<run>/.summary_cache.npz`。
  用它做完了 P1-1 归因与 P1-2 证据（`DEFECT_LOG_zh.md` DEF-023）。

## 五、高层链并入 `main`（P0-1）+ 导出流程固化（P0-2）

| 日期 | 内容 | 关键实测 | commit |
|---|---|---|---|
| 2026-09-20 | **P0-1 高层 replay 链合并进 `main`**（replay 布局：`actions` 观测少 7 维 + 轮关节掩码 → L2 布局推导 → ① `ll_command` → ② IK 目标写 `pose_command_b` + ③ `ee_goal` 用 root 系 → history 回放 → ⑦ checkpoint 参数化 + ⑧ 懒加载低层 cfg → ⑤ 的 R1 抽 `LowLevelPolicyActionBase` + 迁移 nav） | 4 个高层任务 `--headless --num_envs 64 --max_iterations 2` **全 EXIT=0**：Pick-Flat-Teacher reward **1.11**、Pick-WBC-Flat **1.28**、Teleop **0.15**、Nav-Flat-Teacher **10.25**；启动打印 "低层 obs 维度 == checkpoint 期望"（83/76/76/69）、动作分块 `leg12+wheel4+ee_ik{0,7}`；4 份日志均无 Traceback | `129848e`（第一步）、`af4602d`（⑦⑧）、`07601e9`（R1） |
| 2026-09-20 | 合并冲突按 TODO 约定解决：R1 保留基类写法，基类里的内联加载换成 ⑦ 的 `load_low_level_policy(...)`，并给 `verify_low_level_layout` 补传 `policy_layout_json` | 见 `DEFECT_LOG_zh.md` DEF-018 | `07601e9` |
| 2026-09-20 | 删掉高层分支带来的旧 `docs/review/*.md`（其中 `next_session_prompt.md` 与 main 的 `NEXT_SESSION_PROMPT.md` 只差大小写） | `docs/review/` 现在只剩 TODO/DONE/DEFECT_LOG/NEXT_SESSION_PROMPT + `templates/`；`git status` 干净 | `30d5411`、`0772757` |
| 2026-09-20 | **P0-2 导出流程固化**：`export_deploy_policy.py` 默认输出目录 `<run>/exported` → **`<run>/exported_deploy`**（并把 actor-only 陷阱写成显式警告）；`train.py` 训练结束打印可直接复制的导出命令；`docs/train_history_flat_zh.md` 补"训练完成后"章节 + 修正"高层 replay 还不支持 history"的过时说明 | `2026-09-20_00-50-31/model_15000.pt` 实测导出 `exported_deploy/{policy.pt 1001636 B, policy_layout.json}`：scripted vs eager **0.000e+00**、与 `ActorCriticHistory.act_inference` 交叉校验 **0.000e+00**、`kind=history / policy_obs_dim=83 / 10×70 / latent32 / action16` | `498e847` |
| 2026-09-20 | **ONNX 导出**：`--onnx/--no-onnx`（默认开）+ `--opset`（默认 17）；自检 = `onnx.checker` + onnxruntime↔TorchScript **相对**误差 + batch 1/5 动态维验证；`policy_layout.json` 增补 `onnx` 段与 `history_order/history_note`（见 DEF-020） | run `2026-09-20_00-50-31` **重新导出最新 checkpoint `model_19999.pt`**：`exported_deploy/{policy.pt, policy.onnx 981 KB, policy_layout.json}`；相对误差 **1.87e-07**（绝对 3.43e-05 / 幅值 183.6）；独立复核 B=1/3/8 相对误差 1.5e-07~2.7e-07；图 = opset17、`policy_obs['batch',83]`+`history_flat['batch',700]` → `action['batch',16]` | `708ca53` |
| 2026-09-20 | **部署交接**：`probe_deploy_layout.py`（一次性打印关节序/默认角/限位/动作增益/观测逐项 scale·clip·noise/关键 body/默认姿态几何）+ `docs/deploy_sim2sim_sim2real_zh.md`（接口契约、IK 复刻、MuJoCo 参数、七步上线顺序、失败模式表） | 探针实测：**动作增益 hipx 0.125 / 其余腿 0.25 / 轮速 5.0**；原生关节序 wheel=15..18；观测 policy **83**/critic 86/history **700**/privileged 89；MuJoCo 默认姿态 `gripper_base` 相对 base `(0.3492,0,0.4326)` vs Isaac `(0.3492,0,0.4327)` | `7458672` |

## 六、部署基线（deploy baseline）

**当前拿去部署的基线 = `main @ 2d49f47`**（`git rev-list --left-right --count origin/main...main` = `0 0`，
即与 `origin/main` 完全一致、已推送）。基线一旦记录就不再"漂"：后续代码/文档提交只在
`main` 上往前走，**要部署就 checkout 这个 commit（或用 tag）**，不要用"当时的 main"。

| 项 | 值 / 位置 |
|---|---|
| 部署口径代码 | `main @ 2d49f47`（含低层 root_height 专项 + 高层链 + 导出/部署工具链） |
| 对应训练 run | `logs/rsl_rl/history_adaptation/2026-09-20_00-50-31`（4096 envs，iter 0 → 19999） |
| checkpoint | `<run>/model_19999.pt` |
| 部署态策略 | `<run>/exported_deploy/{policy.pt, policy.onnx, policy_layout.json}` |
| 接口契约 | `policy_obs 83` + `history_flat 700`（10×70，**最旧→最新**）→ `action 16`；ONNX opset 17，与 TorchScript 相对误差 **1.87e-07** |

导出物指纹（`Get-FileHash -Algorithm SHA256`，2026-09-20 记录；换 checkpoint/改环境
重导出后**必须**重新记录）：

| 文件 | 字节 | sha256 |
|---|---|---|
| `exported_deploy/policy.pt` | 1001636 | `43C63D19…4522F6510` |
| `exported_deploy/policy.onnx` | 981002 | `77757542…B2FF53B8D1` |
| `exported_deploy/policy_layout.json` | 1640 | `7E3B11EB…B9E6A7947` |
| `model_19999.pt`（源头） | — | `592A50D6…89A87AC14` |

### 训练代码 vs 部署代码（这条必须能回溯）

run 目录里的 `<run>/git/loco-manip-unified-rl-agent.diff`（rsl_rl 训练启动时自动 dump）显示：
该 run 训练时在分支 `codex/ll-height-stability @ 96e1b66`，工作区唯一改动是
**注释掉 `FKReachableEECommand`**（该类没有任何 task/配置引用，`rough_env_cfg.py` 里只有一行
注释指向它 ⇒ 死代码，注释掉不影响训练）。

`main` 与它在这条低层链上的差异（`git diff 96e1b66 main -- source/rl_training/.../velocity ...`）
只有以下四类，**没有动力学 / 观测布局变化**：

1. `SceneEntityCfg(body_names="")` / `joint_names=""` → `None` 的占位符清理（weight=0 的死配置）；
2. `stance_width=float`（类型对象）→ 数值（同样是 weight=0）；
3. 新增启动期自检事件 `mdp.check_policy_layout`（只打印 + 断言，不改动力学）；
4. `observations.py` 的索引空间断言与文档注释、`deeprobotics.py` 删掉一段标定注释、
   `flat_env_wbc_cfg.py` 里旧文档路径改指 `DEFECT_LOG_zh.md`。

⇒ **结论**：在当前口径下"main 的代码 = 训出这个策略的代码"成立。要复现旧 checkpoint 时
只需注意别改 `ee_ik` 这类 action 维度（DEF-013：布局一变旧 checkpoint 静默失效）。
详见 `DEFECT_LOG_zh.md` DEF-022。
