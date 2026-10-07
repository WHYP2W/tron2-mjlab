# VeloWB 实验执行指南

本文档对应所提供 v1.3 方案的可运行仿真实现，不代表论文实验已经完成。
长期训练收敛、可重复的性能提升、负载适应能力和真机效果仍需验证。
修改动作或坐标系约定前，请先阅读[接口约定](interface_contract.md)。

## WBC 迁移预设

师兄的动作接口、执行器、奖励、归一化与 CTS 设计已迁移为独立的 `wbc` 配置，
不覆盖原 `plain_v0` / B0-B7。迁移的是控制设计，不是直接复用旧模型权重。
新接口是 14 维动作、95/96 维 Actor 观测，必须重新训练。

| 预设 | 控制定义 | 学习方法 |
| --- | --- | --- |
| `wbc_plain` | 世界系完整六维速度，无末端位姿误差 | PPO，启用观测归一化 |
| `wbc_cts` | 与 wbc_plain 相同 | 5 帧历史、64D latent、75% Teacher CTS |
| `wbc_height` | heading 系六维速度，Z 轴另加积分高度反馈 | PPO 工程对照，不是纯速度基线 |

先运行纯速度迁移版：

```bash
uv run velowb train --preset wbc_plain --seed 0
```

默认 256 环境、10000 次更新，不自动启动长训练。每回合上限 200 s。
腿臂默认零位，但重置加入随机位置/速度扰动；模型锁定 arm1/arm6，
臂改用位置残差，轮速比例 7.0，启用参考 PD、自碰撞、物理步时延和动力学随机化。
纯 PPO 先验证基本控制后，再训练 CTS，便于区分控制迁移和适应算法的效果。

只检查管线时：

```bash
uv run velowb train --preset wbc_plain --num-envs 4 --iterations 2 \
  --steps-per-env 8 --learning-epochs 1 --mini-batches 1 --smoke
uv run velowb train --preset wbc_cts --num-envs 4 --iterations 2 \
  --steps-per-env 8 --learning-epochs 1 --mini-batches 1 --smoke
```

回放仍使用带手动速度面板的同一入口：

```bash
uv run velowb play --checkpoint-file PATH_TO_WBC_CHECKPOINT --viewer viser
```

直接观察工程高度反馈版本可训练 `--preset wbc_height`。如只做高度反馈消融、
不同时更换坐标系，追加 `--set command_frame=world`。
通过新的 wbc_plain G2 验收后再运行：

```bash
uv run velowb train --preset wbc_cts --seed 0 --gate-report PATH_TO_WBC_G2_REPORT
uv run velowb plan --suite wbc --output experiments/generated_wbc
```

WBC 矩阵为三种方法各三个种子，共九项；计划生成不启动训练。
G1 会按四个可动臂关节验证当前 WBC 模型。G2 会拒绝高度反馈模型，
并记录控制接口类型，防止误用原 15 维 B1 的验收报告。
评估 CSV 同时保留世界系运动量和指令坐标系中的跟踪量；
heading 下的轴向 RMSE/频响不能误按世界轴计算。高度工程版额外报告高度误差。

需要明确的适配差异：保留当前六维末端指令与 FK 估计器，没有复现旧五维腕关节指令；
MuJoCo 接触、严格安全终止、动作物理限幅、加速度差分、编码器网络结构与师兄系统不完全相同。
这些预设通过短训练/保存/加载/导出的验证，不代表复现了师兄 50000 次模型的最终控制水平。
详细参数及限制见[接口约定](interface_contract.md)。

## 1. 训练前验证

```bash
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run velowb validate-estimator --samples 128 --output logs/velowb/g1
uv run velowb plot --run logs/velowb/g1
```

G1 对每种运动模式采样 128 个状态，与 MuJoCo 真值对照，要求六轴最大绝对误差小于 1e-6。
输出 estimator.csv、g1.json，以及可选的 PNG/PDF 误差图。
每次训练前还会自动执行一次 G1 预检，每种运动模式采样 16 个状态。
使用 `--free-yaw` 可检查未锁定机械臂偏航关节的接口，使用 `--variant SFYG` 可检查 SFYG。

## 2. 先训练基础速度伺服

```bash
uv run velowb train --preset plain_v0 --seed 0 --output logs/velowb/plain_seed0
```

默认使用 256 个并行环境，每次更新采样 24 步，共更新 10,000 次，策略频率为 50 Hz，
总计 61,440,000 条状态转移样本。这是正式训练规模，不是快速演示。
`plain_v0` 开放 xyz 线速度和偏航角速度，不使用历史编码器、IK 或前瞻，
不加入传感器噪声和动力学随机化，零指令比例为 30%。它是 G2 验收的起点。

如仅需检查接口是否跑通，而非获取训练结果：

```bash
uv run velowb train --preset plain_v0 --num-envs 4 --iterations 1 \
  --steps-per-env 4 --learning-epochs 1 --mini-batches 1 --smoke \
  --output logs/velowb/my_smoke
```

冒烟测试模式最多执行两次更新，并在检查点、运行清单和图表中标记。
冒烟结果不能通过 G2。B3 的冒烟测试需要两次更新，教师阶段和学生阶段各一次。

输出目录必须是新目录。每次训练保存配置、随机种子、源码归档及哈希、XML/网格哈希、
依赖版本、interface.json、TensorBoard 事件、检查点、教师/学生训练阶段和 student_policy.pt。
git 提交记录为空表示工作区尚无提交，仍会保留源码快照。

恢复训练时，使用完全相同的 ServoConfig，并指定新的输出目录：

```bash
uv run velowb train --preset plain_v0 --seed 0 \
  --resume logs/velowb/plain_seed0/model_4999.pt --iterations 10000 \
  --output logs/velowb/plain_seed0_resume
```

恢复时会重置仿真环境，同时恢复优化器、训练阶段和课程进度。
这不是仿真状态及随机数生成器状态的逐位续接。
原有行走检查点的动作与观测接口不同，因此程序会拒绝加载。

### 交互式回放与手动调速

检查当前模型的动作时，使用 `play`，不是 `eval`：

```bash
uv run velowb play --checkpoint-file PATH_TO_SERVO_CHECKPOINT --viewer viser
```

打开终端显示的 Viser 地址。顶部 `EE Velocity (World)` 面板默认启用手动控制，
各速度初始为零，可调整末端世界系 `Vx、Vy、Vz、Wz`；只有已训练完整六维指令的
检查点才开放 `Wx、Wy`。`Stop` 将末端速度指令归零并切回手动模式，不是实机急停。
取消 `Manual control` 或启动时加入 `--auto`，可恢复训练时的自动指令。
`Simulation / Speed` 只控制回放速度，不是机器人速度指令。
该入口使用单环境、确定性学生策略，不执行训练或批量评估。
加入 `--paused` 可停在重置后的初态，点击 `Play` 后才开始执行策略。
当前默认机械臂关节初态已纠正为全零；旧非零初态检查点回放时会给出不匹配警告，
不能将这种回放当作相同初始条件下的性能比较，也不能把末端速度归零理解成关节回零。

## 3. 零速度实验与 G2

以下命令中的检查点路径应替换为训练实际输出的最终检查点：

```bash
uv run velowb eval --checkpoint logs/velowb/plain_seed0/model_9999.pt \
  --scenario zero --episodes 20 --seconds 10 --push-force 20 \
  --output logs/velowb/zero_seed0
uv run velowb eval --checkpoint logs/velowb/plain_seed0/model_9999.pt \
  --scenario ramp --episodes 20 --seconds 10 \
  --output logs/velowb/ramp_seed0
uv run velowb gate --zero-run logs/velowb/zero_seed0 \
  --tracking-run logs/velowb/ramp_seed0 --output logs/velowb/g2.json
```

推扰为沿世界系 X 方向的可重复外力，在回合时长的 40% 处开始，持续 0.2 s。
G2 要求使用同一个非冒烟基础伺服检查点：每个必测轴及其正负方向至少 5 次试验，
轨迹记录至少 5 s，跌倒率不超过 5%，归一化空间速度 RMSE 不超过 0.5，
零指令下的线速度扰动传递比不超过 0.7，基座引起的末端线速度 RMS 至少为 0.01 m/s。
这些是明确的工程初始阈值，不是已经确立的论文验收标准。
Python 接口 `g2_gate` 支持指定阈值；任何修改都应在评估前声明。
通过记录与机器人形态、机械臂偏航约束绑定。

G2 未通过也是有效结果。应先修复基础伺服，再推进后续阶段。
不能通过让基座完全不动、从而得到较小的末端速度来绕过扰动抑制验收。

## 4. 基线与前瞻消融

实验协议配置位于 [experiments/velowb.json](../experiments/velowb.json)。

| 预设 | 方法定义 |
| --- | --- |
| B0 | 直接跟踪六维位姿的 RL，不使用历史或 IK |
| B1 | 基础世界系空间速度伺服 |
| B2 | 历史 MLP 编码器，不使用特权教师 |
| B3 | 先训练教师 PPO，再冻结策略并蒸馏学生 |
| B4 | 教师/学生并发训练，共享伺服策略 |
| B5 | 轮子/机械臂 IK，固定腿部姿态，无隐藏平衡策略 |
| B6 | IK 作为观测，策略输出完整动作 |
| B7 | IK 残差控制 + CTS |
| K0 | 无运动先验 |
| K1 | 仅当前时刻 IK 残差控制 |
| K2 | K1 + 未来关节状态 |
| K3 | K1 + 未来关节/末端状态 |
| K4 | K1 + 质心/Jacobian 前瞻 |
| K5 | 完整前瞻 + CTS |

B0-B7 共用噪声/随机化设置、动作接口、策略网络和价值网络的隐藏层宽度、
采样预算及种子集合。B0 的任务奖励和任务特征必然不同。
编码器与输入尺寸导致的参数量差异会记录下来，不予隐藏。
B3 的教师/学生阶段分配同一份总数据预算；CTS 不额外增加环境数量。
各方法的辅助优化开销并不完全相同。

```bash
uv run velowb plan --suite baselines --gate-report logs/velowb/g2.json \
  --output experiments/generated_baselines
uv run velowb plan --suite preview --gate-report logs/velowb/g2.json \
  --output experiments/generated_preview
uv run velowb plan --suite ablations --gate-report logs/velowb/g2.json \
  --output experiments/generated_ablations
uv run velowb train --matrix experiments/generated_baselines/matrix.json \
  --run-id B4_seed0 --gate-report logs/velowb/g2.json
```

`plan` 保存固定的运行配置和 commands.txt，不启动任何任务。
基线计划包含 21 个训练任务和 3 个 IK 评估任务；前瞻计划包含 18 个任务，
均使用种子 0/1/2。`--suite all` 对应大规模研究矩阵，不建议作为首次运行。
偏航自由的消融实验需要对应偏航自由基础伺服的 G2 通过报告。

计划中的消融项均可配置：伺服输入、历史长度、特权信息子集、适应策略、
指令波形课程、零指令比例、偏航锁定、动作平滑权重、IK 接入方式和残差范围。

## 5. 独立测试集评估

```bash
uv run velowb eval --checkpoint PATH_TO_SERVO_CHECKPOINT \
  --scenario step --episodes 30 --seconds 10 --output logs/velowb/steps
uv run velowb eval --checkpoint PATH_TO_SERVO_CHECKPOINT \
  --scenario frequency --episodes 10 --output logs/velowb/frequency
uv run velowb eval --checkpoint PATH_TO_SERVO_CHECKPOINT \
  --scenario reaching --episodes 30 --seconds 10 --output logs/velowb/reaching
uv run velowb eval --checkpoint PATH_TO_SERVO_CHECKPOINT \
  --scenario figure_eight --episodes 20 --seconds 20 --output logs/velowb/figure_eight
```

其他场景参数包括：zero（零速度）、constant（常值）、ramp（斜坡）、sine（正弦）、
bandlimited（带限信号）、minimum_jerk（最小加加速度）、line（直线）、circle（圆）、
spline（样条）、orientation（纯姿态），以及 suite（组合测试）。
基本运动测试默认覆盖六轴的正负两个方向；`--axis 0` 可单独测试 vx。
扫频使用 0.1/0.2/0.5/1/2 Hz，每次试验至少覆盖四个周期；
若跌倒前的有效周期不足，则不输出增益/相位结果。
独立评估种子默认为 1000，可使用 1001/1002 重复试验。
不同方法必须使用相同的目标、轨迹、噪声、扰动和时长协议。

静态到达目标通过当前构型附近、满足关节限位的机械臂 FK 生成。
不同连续轨迹共用同一个内层检查点，不重新训练内层策略。
三维路径采用 SciPy 端点导数受约束的三次样条，并使用解析导数作为前馈。
位姿测试采用世界系 SO(3) 姿态误差。
B0 与速度方法应比较最终位姿误差、到达成功率、调节时间、超调、平滑性和平衡，
而不是直接比较定义不同的内层奖励值。

分布外（OOD）条件包括：`--ood payload`（负载）、friction（摩擦）、sensors（传感器噪声）、
latency（时延）、timestamp_skew（时间戳错位）、odometry_drift（里程计漂移）、
command_latency（指令时延）、actuators（执行器）和 push（推扰）。具体数值见 JSON 配置。
`--set field=JSON-value` 可指定预先声明的附加条件。
评估时不得修改策略网络结构或指令归一化方式。
默认 nominal 条件关闭训练随机化；其他 OOD 预设则在检查点的训练分布基础上，
覆盖指定因素。解释鲁棒性测试结果时，应明确这一区别。

使用 `--video` 输出 1280x720 视频；无显示器的 Linux 环境可能需要 EGL：

```bash
MUJOCO_GL=egl uv run velowb eval --checkpoint PATH_TO_SERVO_CHECKPOINT \
  --scenario zero --episodes 20 --seconds 10 --video --output logs/velowb/zero_video
```

视频与响应曲线固定展示第 0 个回合，包含失败情况，不从多个回合中挑选最好的一次。
所有试验的原始信号和失败标记均保存在 CSV 中。
试验跌倒时保留跌倒前的轨迹，不把自动重置后的新回合拼接进同一条轨迹。
被截断或时长不足的响应不能据此宣称已经进入稳态。

## 6. 统计表与结果溯源

```bash
uv run velowb aggregate --runs EVAL_DIR_SEED0 EVAL_DIR_SEED1 EVAL_DIR_SEED2 \
  --output logs/velowb/paper_summary
uv run velowb plot --run logs/velowb/paper_summary
```

输出包括各场景的原始轨迹 CSV、逐回合指标 metrics.csv、汇总表 summary.csv、
PNG/PDF 响应图和汇总图，以及关联来源检查点与哈希的 result_manifest.md。
汇总时先对每个训练种子内的多次试验求平均，再计算独立训练种子均值之间的均值和标准差。
只有一个种子时不报告标准差。
对于不学习的 B5，使用独立评估种子作为重复试验标识；该控制器没有训练种子或学习策略预算。
程序会拒绝重复试验、不兼容的评估协议、混合的冒烟/正式研究数据，以及不一致的学习采样预算。
未获得调节时间或增益、分母为零等情况下，指标保持缺失，不人为填零，并保留对应的有效样本数。

应报告：六轴 RMSE、独立的线速度/角速度误差、上升时间/调节时间/超调、频率增益/相位、
末端漂移、基座扰动传递比、基座倾角/跌倒率、动作变化率、关节/末端加加速度、
功率/能量，以及腿/轮/机械臂的速度参与程度。
必须保留失败样本，使用多个训练种子，并评估至少 20-50 个可达目标。
完成这些比较之前，不应宣称某个组件提升了性能。

## 后续研究验收

- G1：当前可以验证数值实现的正确性。
- G2：需要训练出能够动态平衡、基座运动时仍能抑制末端速度的策略。
- G3/G3.5：需要重复开展直接位姿跟踪对照以及 IK/前瞻对照。
- G4/G5：需要多个种子下 history/CTS 的分布外性能优势，不能只展示潜变量损失曲线。
- G6：需要真实传感器、时间同步、执行器接口标定、看门狗/限位/急停，
  先开展仅估计器测试，再分级推进硬件控制试验。
- 纯仿真输出不能替代 G6，也不能替代 RSS 级别的实验证据。
- MPC、坡地、滚动阻力标定、自碰撞验证、接触操作任务和实机驱动仍属于后续工作，
  不作为已经实现的能力。

## 已完成的实现验证（2026-10-05）

- 全部 18 项 unittest 检查通过，覆盖原有行走任务、B0/B1/B2/B3/B4/B6/B7 的 GPU PPO、
  仅学生策略导出，以及端到端实验流程。
- [G1 报告](../logs/velowb/verification_2026-10-05/g1/g1.json)：384 个随机状态，
  六轴最大绝对误差为 7.99e-15。单元测试还覆盖了两种机器人形态和两种偏航设置。
- B1、B3 和完整前瞻 B7 各更新两次的检查点、学生策略导出及源码快照位于
  `logs/velowb/verification_2026-10-05/`。这些产物仅用于冒烟测试。
- 已实际验证零速度、斜坡、静态到达、圆轨迹和 IK-only 评估入口、CSV/图表、
  [720p 视频](../logs/velowb/verification_2026-10-05/b1_zero_hd/zero.mp4)，以及
  [种子级汇总](../logs/velowb/verification_2026-10-05/summary/result_manifest.md)。
- [G2 报告](../logs/velowb/verification_2026-10-05/g2.json)：未通过。
  两个时长为一秒的零指令回合均跌倒；短冒烟测试本身也明确不具备通过资格。
- 已生成包含 138 个计划条目的完整实验矩阵，没有启动批量研究任务。
- 不宣称已经获得长期收敛、统计性能优势或真机实验结果。