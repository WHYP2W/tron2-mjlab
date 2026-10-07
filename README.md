# TRON2A YG 平地训练

基于 mjlab 的基础速度跟踪环境，仅包含 TRON2A 的两种 YG 形态。

| 任务 | 动作 |
| --- | --- |
| `Mjlab-Velocity-Flat-TRON2A-SFYG` | 10 个腿关节位置目标 |
| `Mjlab-Velocity-Flat-TRON2A-WFYG` | 8 个腿关节位置目标 + 2 个轮速目标 |

两种形态共用 [模型适配](src/tron2_mjlab/robot.py) 和 [环境配置](src/tron2_mjlab/env_cfgs.py)。
机械臂、夹爪按原 XML 零位固定，保留外观、质量和碰撞几何，不参与策略动作。
仅使用平面地形，无高度扫描或地形课程。WFYG 不使用抬脚、腾空、防滑等步行奖励，
轮子累计转角不进入位置观测和姿态奖励，横向速度指令固定为零。

## 安装

需要 Python 3.10-3.13、uv，以及驱动兼容的 NVIDIA CUDA GPU。
仿真依赖固定为 mjlab 1.1.0、MuJoCo/MuJoCo Warp 3.5.0、Warp 1.12.0。

```bash
uv sync --locked
uv run list_envs
```

若 PyPI 下载缓慢，可使用镜像：

```bash
uv sync --default-index https://pypi.tuna.tsinghua.edu.cn/simple
```

## 训练

```bash
uv run train Mjlab-Velocity-Flat-TRON2A-SFYG
uv run train Mjlab-Velocity-Flat-TRON2A-WFYG
```

默认 1024 个并行环境，物理步长 2 ms，策略频率 50 Hz，10000 次 PPO 迭代。
使用 TensorBoard 本地日志，无需 W&B 登录。日志和模型位于 `logs/rsl_rl/`。
显存不足时使用 `--env.scene.num-envs 256`。

快速检查训练入口：

```bash
uv run train Mjlab-Velocity-Flat-TRON2A-WFYG --env.scene.num-envs 16 --agent.max-iterations 1
```

## 预览

### 手动调速面板

```bash
uv run tron2-play SFYG --checkpoint-file /path/to/sfyg/model.pt
uv run tron2-play WFYG --checkpoint-file /path/to/wfyg/model.pt
```

打开终端输出的 Viser 地址，在 `Velocity Command` 面板中调节 `Forward (m/s)`、
`Lateral (m/s)` 和 `Yaw (rad/s)`。默认启用 `Manual control`，初始速度为零；
取消勾选恢复自动随机指令。WFYG 的横向滑块禁用。
`Stop` 将三个目标速度归零并切回手动模式，不是暂停仿真或实机急停。
原 `Simulation / Speed` 调整的是仿真播放速度，与机器人目标速度不同。

没有模型时可用 `uv run tron2-play SFYG --agent zero` 检查面板，但零动作策略不会跟踪速度。
SFYG 和 WFYG 的检查点不可混用；必须指定完整的 `.pt` 文件路径，不能只指定目录。

### 自动指令回放

```bash
uv run play Mjlab-Velocity-Flat-TRON2A-SFYG --agent zero --viewer viser
uv run play Mjlab-Velocity-Flat-TRON2A-WFYG --checkpoint-file /path/to/model.pt --viewer viser
```

零动作仅用于检查模型，不代表已经学会平衡。PD 增益、初始姿态和奖励是训练起点，
尚未验证长期收敛或实机部署，不应直接用于实机控制。

## 测试

```bash
uv run python -m unittest discover -s tests -v
```

覆盖两个任务注册、模型编译、平地配置、手动调速与重置、GPU 上 100 步接触检测及一次 PPO 更新；
没有 CUDA 时跳过 GPU 测试。

## VeloWB 论文实验

已根据提供的 v1.3 方案新增独立实验路径，原有两种行走任务与 10 维动作保持不变。
VeloWB 启用原模型中的机械臂，默认使用 WFYG、锁定 `arm1_Joint`，共 15 维全身动作。
原行走 checkpoint 不能直接用于新的末端伺服任务。

实现包括世界系六维末端 FK/Jacobian 估计、速度指令课程、外层位姿 PD、B0-B7
对照、IK residual、100-400 ms 运动学前瞻、history/CTS/两阶段蒸馏、噪声与时延、
载荷/摩擦/执行器随机化，以及多种轨迹评估、CSV/PNG/PDF/视频和实验溯源。

**当前验证的是代码与短训练链路，不代表已经学会动态平衡或完成论文实验。**
正式长期训练、多 seed 性能优势和真机实验仍需执行；程序会拒绝用冒烟结果通过 G2。

```bash
uv run velowb validate-estimator --samples 128 --output logs/velowb/g1
uv run velowb train --preset plain_v0 --seed 0 --output logs/velowb/plain_seed0
uv run velowb eval --checkpoint logs/velowb/plain_seed0/model_9999.pt \
	--scenario zero --episodes 20 --seconds 10 --output logs/velowb/zero_seed0
```

默认训练为 256 个环境、10000 次更新。只检查链路时使用：

```bash
uv run velowb train --preset plain_v0 --num-envs 4 --iterations 1 \
	--steps-per-env 4 --learning-epochs 1 --mini-batches 1 --smoke
```

完整执行顺序、基线与消融、G2 门槛及结果汇总见
[实验执行文档](docs/experiment_plan.md)，坐标系、观测/动作顺序和部署边界见
[接口约定](docs/interface_contract.md)。实验矩阵配置位于
[experiments/velowb.json](experiments/velowb.json)。
真机驱动、传感器标定及安全控制未实现，导出的 Student 不能直接接管实机。

### 师兄 WBC 设计迁移

新增独立预设 `wbc_plain`、`wbc_cts`、`wbc_height`，迁移位置残差动作、14 维关节控制、
分组 PD、观测归一化、臂腿分工与平滑/接触奖励、物理步时延和动力学随机化。
原 `plain_v0` 保留。新接口必须重新训练，不能直接续用旧 15 维模型或师兄的旧 ONNX。

```bash
uv run velowb train --preset wbc_plain --seed 0
uv run velowb play --checkpoint-file PATH_TO_WBC_CHECKPOINT --viewer viser
```

`wbc_plain` 是世界系纯速度对照；`wbc_height` 带高度位置反馈，不能作为纯速度模型通过 G2。
CTS 正式训练仍需同接口的 G2 报告。细节见[实验执行文档](docs/experiment_plan.md)。

## 资源

模型来自 [LimX TRON2 robot description](https://github.com/limxdynamics/tron2-robot-description)，
原始提交为 `f547f5bc949f2a4c98e076e61cf6d3ca73d179a0`。
仅保留 SFYG_TRON2A、WFYG_TRON2A 的 XML 和网格；参考项目、TRON2B、其他形态、
URDF、USD、Xacro 和嵌套 Git 元数据已移除。原 XML 不改写，由加载器完成训练适配。

上游许可与来源声明保留在 [资源目录](src/tron2_mjlab/tron2-robot-description/LICENSE)。
[上游资源清单](src/tron2_mjlab/tron2-robot-description/ASSETS.md) 为原始仓库快照，
其中仍标注 TRON2A 网格来源确认待完成；公开再分发前应向上游确认。
