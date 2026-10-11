# 历史编码器 / 特权编码器损失的"悬崖式暴跌"：定位与机制

分析对象：`History-Adaptation-Deeprobotics-M20-v0`（`PPORoA` + `ActorCriticHistory`）在
TensorBoard 里 `Loss/constraint`、`Loss/latent_distance`、`Loss/imitation` 三条曲线的突降。

复现工具：`scripts/reinforcement_learning/rsl_rl/probe_loss_cliff.py`（纯 tensorboard + numpy，不用起 Isaac）

---

## 1. 结论（一句话）

**悬崖不是"学生终于学会了"，而是"教师塌缩成常数"**：在 `constraint_coef`（λ）从 0 变正的那一刻，
第一次有梯度进 `PrivilegedEncoder`，而它跟策略共用同一个 Adam（per-parameter 归一化 ⇒ λ 的量级
被抹平），于是教师在几十次迭代内被拉向学生、**输入敏感度掉到原来的 4%**，两个 latent 一起变成近似常数，
MSE 因此塌到 ~0。

---

## 2. 时间点：悬崖 = λ 的延迟

`constraint_coef_schedule = (start, end, delay, ramp)`，`λ = clip((counter-delay)/ramp,0,1)·(end-start)+start`；
`counter` 在 `update()` 与 `update_dagger()` 里都会 +1（≈ 迭代号）。

| run | 调度 | 观察到的悬崖 | 相对跌幅 |
|---|---|---|---|
| `2026-09-30_19-36-00_cloud_cap12_20k` | `(0.0, 0.1, 3000, 7000)` | `latent_distance` **it 3045**：0.188 → 0.031；`constraint` **it 3046**：1.2e-3 → 4.3e-5 | −78% / −94% |
| `2026-09-30_00-09-25_cloud_soft20k` | 同上 | `latent_distance` **it 3044**：0.219 → 0.030 | −81% |
| `2026-06-29_17-18-16` | `(0.0, 1.0, **5000**, 5000)` | `latent_distance` **it 5038**：0.322 → 0.043；`constraint` **it 5048** | −82% / −97% |
| `2026-06-26_11-47-17` | `(0.0, 1.0, **0**, 1)`（λ 一开始就是 1） | **没有悬崖**：it 1 已是 0.26、it 21 降到 0.117、it 158 到 0.065 | — |

⇒ λ 的 `delay` 从 3000 改成 5000，悬崖**跟着挪到 5038**；λ 从一开始就 >0 时**根本不出现**这个悬崖。

**同时刻其它变量都排除了**：

* `Curriculum/*` 全部在 it≈2600 之后就是平线（`body_pitch_rew_s3`、`stand_still_*_ramp`、
  `standing_env_ratio_ramp` …）⇒ 不是课程台阶；
* `Loss/learning_rate` 在 delay=5000 那条的悬崖前后基本没变（1.1e-4 → 1.7e-4 → 1.1e-4）；
  cap12 那条的 LR 在悬崖前后都在 1e-5~3.8e-4 之间乱跳 ⇒ 不是学习率事件。
  （另有一个**无关**的 LR 现象：自适应 KL 调度把 LR 反复压到 1e-5 地板再弹回，最大"悬崖"在 it≈11400。）

---

## 3. 机制：教师先被"冻结"3000 步，然后被 λ 一开就拉塌

1. **λ=0 期间教师拿不到任何梯度**：`PPORoA.update()` 里进教师参数的唯一路径是
   `λ · MSE(z_priv, sg(z_hist))`；λ=0 时它恒为 0（`sg` 也挡住了学生→教师的其它路径）。
   `update_dagger()` 只更新历史编码器。⇒ 教师在 delay 之前**一动不动**。

   实测（拿 checkpoint 直接算 `PrivilegedEncoder` 对输入的雅可比范数 ‖∂z/∂x‖_F，与输入分布无关）：

   | it | 0 | 1000 | 2500 | 3000 | 3500 | 5000 | 10000 | 19999 |
   |---|---|---|---|---|---|---|---|---|
   | ‖∂z/∂x‖_F | 0.8832 | 0.8832 | 0.8832 | **0.8832** | 0.198 | 0.040 | 0.034 | 0.038 |
   | 相对初值 | 1.00x | 1.00x | 1.00x | **1.00x** | 0.22x | 0.05x | 0.04x | **0.04x** |

   前四列**逐位相同** ⇒ 教师确实被冻结；λ 一开，教师敏感度掉到 22%，最终只剩 **4%**。

2. **为什么 λ 那么小也立刻生效**：教师和策略**共用同一个 Adam**（`self.optimizer` 覆盖
   `self.policy.parameters()`）。Adam 按参数做二阶矩归一化 ⇒ 更新步长≈学习率，**与梯度尺度无关**。
   所以悬崖处 λ 只有 6e-4（离终值 0.1 差两个数量级）也能立刻把教师推动。
3. **被推向哪**：Term 2 的方向是 `z_priv → sg(z_hist)`，也就是"教师去靠近学生"。学生那边同时被
   PPO + dagger 拉，于是两者一起收敛到同一个近似常数 —— MSE/距离当然暴跌，但**信息没了**。
4. 量化"信息没了"：把 89 维特权向量按域随机化范围采样（负载/惯量/增益/摩擦），
   看特权 latent 在环境参数上的分散度：

   | it | ≤3000（冻结） | 3500 | 5000 | 10000 | 19999 |
   |---|---|---|---|---|---|
   | z_priv 跨环境 std（均值） | 0.0253 | 0.0034 | 0.0007 | 0.0006 | 0.0006 |

   ⇒ 掉 **42 倍**。同期历史编码器**没有**塌（跨窗口 std 0.39~1.13、量级 5~10），
   所以这不是"两边一起训练到很近"，而是**教师单方面退化成常数**、学生去匹配那个常数。

---

## 4. 这意味着什么

* 那条"漂亮的下降曲线"是**假信号**：`Loss/constraint`、`Loss/latent_distance` 可以靠"两边一起变常数"
  骗到 ~1e-6，它**不能**证明 adaptation module 学到了环境信息。要看健康度必须另测
  （教师输入敏感度 / 跨环境分散度 / 学生 latent 与真实环境参数的相关性）。
* 由于 actor 只吃 `z_hist`，z_hist 在真机/真环境上的跨环境差异被这一机制压得很小
  ⇒ **这个模块有很大概率在 it≈3000 之后就已经"名存实亡"**（actor 基本退化成一个纯本体感受策略
  再加 32 个近乎常数的输入）。这与"history 臂相对纯 MLP 基线没有明显优势"的观察一致。
* 与文献对照：RMA 的做法是**两阶段** —— 先用特权信息把 teacher 训成有意义的目标并**冻结**，
  再让学生模仿。这里把两阶段压成"单阶段 + 双向正则"，而 Term 2 这条"让教师靠近学生"的正则，
  在**共享 Adam** 下会直接把教师的信息抹掉。

---

## 5. 建议的修法（按代价从小到大）

1. **只留 Term 3、不要 Term 2**：`constraint_coef_schedule=(0.0, 0.0, …)`（λ≡0），
   保留 `dagger_update_freq=20` ⇒ 学生追一个**冻结**的教师。零代码改动，一个 CLI 覆盖即可。
   （注意：此时教师是随机初始化的分支，信息量有限，但至少不会塌。）
2. **让教师先有意义再冻结**：用特权向量监督预训练教师（例如回归真实环境参数 / 自编码），
   之后**冻结**，学生只用 dagger 追。这是 RMA 的两阶段做法，也最可能真正有效。
3. **给教师独立优化器 + 小 LR**（现在共享 policy Adam，λ 的量级被 Adam 抹平）。
4. **保留 Term 2 的话必须换监控量**：把"教师输入敏感度/跨环境分散度"打进日志，
   不能只看 `Loss/constraint`。

对应的消融臂（都已在 `docs/history_encoder_ablation_flat_zh.md` 的框架里，可直接加）：

| 臂 | λ | dagger | 教师 | 目的 |
|---|---|---|---|---|
| A（现状） | 0→0.1 | 每 20 次 | 会动（塌） | 基线 |
| C（已有） | ≡0 | 关 | 冻结 | 只有 RL 梯度 |
| **D（建议新增）** | ≡0 | 每 20 次 | 冻结 | 学生追冻结随机教师 |
| **E（建议新增）** | ≡0 | 每 20 次 | **预训练后冻结** | RMA 两阶段 |
| **F** | 0→0.1 | 每 20 次 | 独立优化器 + lr/10 | 隔离"共享 Adam"这一项 |

---

## 6. 复现命令

```bash
# 定位悬崖（打印 step、前后平台值、同期的 λ）
python scripts/reinforcement_learning/rsl_rl/probe_loss_cliff.py \
    --run logs/rsl_rl/history_adaptation/2026-09-30_19-36-00_cloud_cap12_20k \
    --window 51 --topk 3 --constraint-schedule 0.0 0.1 3000 7000

# 对照：λ 延迟改成 5000 的那条 ⇒ 悬崖挪到 5038
python .../probe_loss_cliff.py --run logs/rsl_rl/history_adaptation/2026-06-29_17-18-16 \
    --tags Loss/latent_distance Loss/constraint --constraint-schedule 0.0 1.0 5000 5000
```

（教师敏感度/分散度那两组数，是直接拿 `model_<it>.pt` 里的
`privileged_encoder.encoder.*` 权重在前向/反向里算的，见本文第 3 节的表格。）
