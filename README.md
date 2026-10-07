# TRON2A SFYG 平地训练

基于 mjlab 的基础速度跟踪环境，仅包含 TRON2A 的 SFYG 形态。

| 任务 | 动作 |
| --- | --- |
| `Mjlab-Velocity-Flat-TRON2A-SFYG` | 10 个腿关节位置目标 |

实现见 [模型适配](src/tron2_mjlab/robot.py) 和 [环境配置](src/tron2_mjlab/env_cfgs.py)。
机械臂、夹爪按原 XML 零位固定，保留外观、质量和碰撞几何，不参与策略动作。
仅使用平面地形，无高度扫描或地形课程。

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
```

默认 1024 个并行环境，物理步长 2 ms，策略频率 50 Hz，10000 次 PPO 迭代。
使用 TensorBoard 本地日志，无需 W&B 登录。日志和模型位于 `logs/rsl_rl/`。
显存不足时使用 `--env.scene.num-envs 256`。

快速检查训练入口：

```bash
uv run train Mjlab-Velocity-Flat-TRON2A-SFYG --env.scene.num-envs 16 --agent.max-iterations 1
```

## 预览

### 手动调速面板

```bash
uv run tron2-play SFYG --checkpoint-file /path/to/sfyg/model.pt
```

打开终端输出的 Viser 地址，在 `Velocity Command` 面板中调节 `Forward (m/s)`、
`Lateral (m/s)` 和 `Yaw (rad/s)`。默认启用 `Manual control`，初始速度为零；
取消勾选恢复自动随机指令。
`Stop` 将三个目标速度归零并切回手动模式，不是暂停仿真或实机急停。
原 `Simulation / Speed` 调整的是仿真播放速度，与机器人目标速度不同。

没有模型时可用 `uv run tron2-play SFYG --agent zero` 检查面板，但零动作策略不会跟踪速度。
必须指定 SFYG 模型完整的 `.pt` 文件路径，不能只指定目录。

### 自动指令回放

```bash
uv run play Mjlab-Velocity-Flat-TRON2A-SFYG --agent zero --viewer viser
uv run play Mjlab-Velocity-Flat-TRON2A-SFYG --checkpoint-file /path/to/model.pt --viewer viser
```

零动作仅用于检查模型，不代表已经学会平衡。PD 增益、初始姿态和奖励是训练起点，
尚未验证长期收敛或实机部署，不应直接用于实机控制。

## 测试

```bash
uv run python -m unittest discover -s tests -v
```

覆盖 SFYG 任务注册、模型编译、平地配置、手动调速与重置、GPU 上 100 步接触检测及一次 PPO 更新；
没有 CUDA 时跳过 GPU 测试。

## 资源

模型来自 [LimX TRON2 robot description](https://github.com/limxdynamics/tron2-robot-description)，
原始提交为 `f547f5bc949f2a4c98e076e61cf6d3ca73d179a0`。
仅保留 SFYG_TRON2A 的 XML 和网格；参考项目、TRON2B、其他形态、
URDF、USD、Xacro 和嵌套 Git 元数据已移除。原 XML 不改写，由加载器完成训练适配。

上游许可与来源声明保留在 [资源目录](src/tron2_mjlab/tron2-robot-description/LICENSE)。
[上游资源清单](src/tron2_mjlab/tron2-robot-description/ASSETS.md) 及上游说明文档为原始仓库快照，不代表本项目包含的形态，
其中仍标注 TRON2A 网格来源确认待完成；公开再分发前应向上游确认。
