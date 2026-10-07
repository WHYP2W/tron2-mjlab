"""Manual velocity commands and Viser playback for TRON2A."""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import ViserPlayViewer

from .robot import VARIANTS
from .gait import GaitVelocityCommand, GaitVelocityCommandCfg, gait_phase


class ManualVelocityCommand(GaitVelocityCommand):
    def __init__(self, cfg, env):
        self.manual_target: tuple[float, float, float] | None = (0.0, 0.0, 0.0)
        super().__init__(cfg, env)

    def _resample_command(self, env_ids):
        if self.manual_target is None:
            super()._resample_command(env_ids)
        else:
            self._update_command()

    def _update_command(self):
        if self.manual_target is None:
            super()._update_command()
        else:
            for axis, value in enumerate(self.manual_target):
                self.vel_command_b[:, axis] = value


@dataclass(kw_only=True)
class ManualVelocityCommandCfg(GaitVelocityCommandCfg):
    def build(self, env):
        return ManualVelocityCommand(self, env)


class VelocityViewer(ViserPlayViewer):
    def setup(self):
        super().setup()
        self.command = self.env.unwrapped.command_manager.get_term("twist")
        ranges = self.command.cfg.ranges
        with self._server.gui.add_folder("Velocity Command", order=-1):
            self.manual = self._server.gui.add_checkbox("Manual control", initial_value=True)
            self.sliders = [
                self._server.gui.add_slider(label, min=limits[0], max=limits[1],
                                           step=0.05, initial_value=0.0)
                for label, limits in (
                    ("Forward (m/s)", ranges.lin_vel_x),
                    ("Lateral (m/s)", ranges.lin_vel_y),
                    ("Yaw (rad/s)", ranges.ang_vel_z),
                )
            ]
            stop = self._server.gui.add_button("Stop", icon=viser.Icon.PLAYER_STOP)
            stop.on_click(self._stop)

    def _stop(self, event):
        with self._server.atomic():
            self.manual.value = True
            for slider in self.sliders:
                slider.value = 0.0

    def sync_viewer_to_env(self):
        manual = self.manual.value
        was_manual = self.command.manual_target is not None
        for slider in self.sliders:
            slider.disabled = not manual
        self.command.manual_target = (
            (self.sliders[0].value,
             self.sliders[1].value,
             self.sliders[2].value) if manual else None
        )
        if was_manual and not manual:
            self.command._resample_command(torch.arange(self.env.num_envs, device=self.env.device))
        self.command._update_command()
        self.env.unwrapped.observation_manager._obs_buffer = None
        observations = self.env.unwrapped.observation_manager.compute()
        actor = observations["actor"]
        if not isinstance(actor, torch.Tensor) or actor.ndim != 3:
            raise ValueError("CTS playback requires actor observations shaped (batch, history, features)")
        # Refresh the live command without appending extra simulation history frames.
        actor[:, -1, -3:] = self.command.command
        actor[:, -1, -5:-3] = gait_phase(self.env.unwrapped, command_name="twist")


def main():
    parser = argparse.ArgumentParser(description="TRON2A CTS student playback (default: LBC-CTS)")
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--baseline", action="store_true",
                        help="Use Baseline-CTS instead of LBC-CTS, without arm-state observations")
    parser.add_argument("--checkpoint-file", type=Path)
    parser.add_argument("--agent", choices=("trained", "zero"), default="trained")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.agent == "trained" and (args.checkpoint_file is None or not args.checkpoint_file.is_file()):
        parser.error("--checkpoint-file must point to a .pt file; use --agent zero for model inspection")
    task = f"Mjlab-Velocity-Flat-TRON2A-{args.variant}" + ("-Baseline-CTS" if args.baseline else "-LBC-CTS")
    cfg = load_env_cfg(task, play=True)
    cfg.commands["twist"] = ManualVelocityCommandCfg(**vars(cfg.commands["twist"]))
    runner_cfg = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
        if args.agent == "zero":
            def policy(obs):
                return torch.zeros((env.num_envs, wrapped.num_actions), device=env.device)
        else:
            runner = load_runner_cls(task)(wrapped, asdict(runner_cfg), device=args.device)
            runner.load(str(args.checkpoint_file), load_cfg={"actor": True},
                        strict=True, map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)
        VelocityViewer(wrapped, policy).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()