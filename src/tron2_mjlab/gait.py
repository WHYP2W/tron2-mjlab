"""Shared alternating-foot clock for locomotion observations and rewards."""

from dataclasses import dataclass, field
from math import expm1, isfinite, tau
from typing import TYPE_CHECKING

import torch
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(frozen=True)
class GaitClockCfg:
    slow_period: float = 0.9
    fast_period: float = 0.6
    slow_speed: float = 0.2
    fast_speed: float = 1.5
    fast_yaw_speed: float = 1.0
    smoothing_time: float = 0.2
    linear_threshold: float = 0.05
    angular_threshold: float = 0.05

    def __post_init__(self):
        if not all(isfinite(value) for value in vars(self).values()):
            raise ValueError("Gait parameters must be finite")
        if not 0 < self.fast_period <= self.slow_period:
            raise ValueError("Gait periods must satisfy 0 < fast <= slow")
        if not 0 <= self.linear_threshold <= self.slow_speed < self.fast_speed:
            raise ValueError("Gait speeds must satisfy 0 <= threshold <= slow < fast")
        if not 0 <= self.angular_threshold < self.fast_yaw_speed or self.smoothing_time <= 0:
            raise ValueError("Gait yaw range and smoothing time must be positive")


class GaitClock:
    def __init__(self, cfg: GaitClockCfg, num_envs: int, device: str):
        self.cfg = cfg
        self.phase = torch.zeros(num_envs, device=device)
        self.frequency = torch.full_like(self.phase, 1.0 / cfg.slow_period)

    def moving(self, command: torch.Tensor) -> torch.Tensor:
        return (
            (torch.linalg.vector_norm(command[:, :2], dim=1) > self.cfg.linear_threshold)
            | (command[:, 2].abs() > self.cfg.angular_threshold)
        )

    def target_period(self, command: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        linear = (
            (torch.linalg.vector_norm(command[:, :2], dim=1) - cfg.slow_speed)
            / (cfg.fast_speed - cfg.slow_speed)
        ).clamp(0, 1)
        angular = (
            (command[:, 2].abs() - cfg.angular_threshold)
            / (cfg.fast_yaw_speed - cfg.angular_threshold)
        ).clamp(0, 1)
        blend = torch.maximum(linear, angular)
        return cfg.slow_period + blend * (cfg.fast_period - cfg.slow_period)

    def advance(self, command: torch.Tensor, dt: float, active: torch.Tensor) -> None:
        if not isfinite(dt) or dt <= 0:
            raise ValueError("Gait time step must be finite and positive")
        alpha = -expm1(-dt / self.cfg.smoothing_time)
        frequency = self.frequency + alpha * (self.target_period(command).reciprocal() - self.frequency)
        self.frequency.copy_(torch.where(active, frequency, self.frequency))
        increment = dt * self.frequency * (active & self.moving(command))
        self.phase.copy_(torch.remainder(self.phase + increment, 1.0))

    def reset(self, env_ids: torch.Tensor | slice | None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self.phase[env_ids] = 0.0
        self.frequency[env_ids] = 1.0 / self.cfg.slow_period


class GaitVelocityCommand(UniformVelocityCommand):
    def __init__(self, cfg: "GaitVelocityCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.gait_clock = GaitClock(cfg.gait, self.num_envs, self.device)

    def compute(self, dt: float) -> None:
        super().compute(dt)
        # Commands update once per policy step, after rewards and episode resets.
        self.gait_clock.advance(self.command, dt, self._env.episode_length_buf > 0)

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        extras = super().reset(env_ids)
        self.gait_clock.reset(env_ids)
        return extras


@dataclass(kw_only=True)
class GaitVelocityCommandCfg(UniformVelocityCommandCfg):
    gait: GaitClockCfg = field(default_factory=GaitClockCfg)

    def build(self, env: "ManagerBasedRlEnv") -> GaitVelocityCommand:
        return GaitVelocityCommand(self, env)


def _clock(env: "ManagerBasedRlEnv", command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, GaitVelocityCommand):
        raise ValueError(f"Gait requires a GaitVelocityCommand named {command_name!r}")
    return command.gait_clock.phase, command.gait_clock.moving(command.command)


def gait_phase(
    env: "ManagerBasedRlEnv",
    command_name: str,
) -> torch.Tensor:
    """Return sin/cos of the left-foot cycle; zero both values when standing."""
    phase, moving = _clock(env, command_name)
    angle = tau * phase
    return torch.stack((angle.sin(), angle.cos()), dim=-1) * moving[:, None]


def alternating_contacts(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    stance_ratio: float,
    command_name: str,
) -> torch.Tensor:
    """Reward matching left/right stance phases, or double support at rest."""
    if not isfinite(stance_ratio) or not 0.5 <= stance_ratio < 1.0:
        raise ValueError("Gait stance ratio must be in [0.5, 1.0)")
    phase, moving = _clock(env, command_name)
    sensor: ContactSensor = env.scene[sensor_name]
    found = sensor.data.found
    if found is None or found.shape != (env.num_envs, 2):
        raise ValueError("Gait requires exactly two contact values ordered left, right")
    foot_phase = torch.stack((phase, torch.remainder(phase + 0.5, 1.0)), dim=-1)
    stance = (foot_phase < stance_ratio) | ~moving[:, None]
    return ((found > 0) == stance).float().mean(dim=1)
