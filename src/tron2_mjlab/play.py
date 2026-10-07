"""Manual velocity commands and Viser playback for TRON2A."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.viewer import ViserPlayViewer

from .env_cfgs import tron2_flat_env_cfg
from .robot import VARIANTS


class ManualVelocityCommand(UniformVelocityCommand):
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
class ManualVelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self, env):
        return ManualVelocityCommand(self, env)


class VelocityViewer(ViserPlayViewer):
    def setup(self):
        super().setup()
        self.command = self.env.unwrapped.command_manager.get_term("twist")
        ranges = self.command.cfg.ranges
        self._lateral_enabled = ranges.lin_vel_y[0] != ranges.lin_vel_y[1]
        with self._server.gui.add_folder("Velocity Command", order=-1):
            self.manual = self._server.gui.add_checkbox("Manual control", initial_value=True)
            self.sliders = [
                self._server.gui.add_slider(label, min=limits[0], max=limits[1],
                                           step=0.05, initial_value=0.0)
                for label, limits in (
                    ("Forward (m/s)", ranges.lin_vel_x),
                    ("Lateral (m/s)", ranges.lin_vel_y if self._lateral_enabled else (-0.3, 0.3)),
                    ("Yaw (rad/s)", ranges.ang_vel_z),
                )
            ]
            self.sliders[1].disabled = not self._lateral_enabled
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
        for axis, slider in enumerate(self.sliders):
            slider.disabled = not manual or (axis == 1 and not self._lateral_enabled)
        self.command.manual_target = (
            (self.sliders[0].value,
             self.sliders[1].value if self._lateral_enabled else 0.0,
             self.sliders[2].value) if manual else None
        )
        if was_manual and not manual:
            self.command._resample_command(torch.arange(self.env.num_envs, device=self.env.device))
        self.command._update_command()
        self.env.unwrapped.observation_manager._obs_buffer = None


class ServoVelocityViewer(VelocityViewer):
    def __init__(self, env, policy, *, manual_control=True, frame_rate=30.0):
        super().__init__(env, policy, frame_rate=frame_rate)
        self._manual_default = manual_control
        self._was_manual = False

    def setup(self):
        ViserPlayViewer.setup(self)
        self.command = self.env.unwrapped.command_manager.get_term("ee_twist")
        settings = self.command.settings
        full_angular = settings.command_axes == "full" or (
            settings.command_axes == "curriculum"
            and self.env.unwrapped.common_step_counter >= settings.curriculum_steps // 3
        )
        self.enabled_axes = (True, True, True, full_angular, full_angular, True)
        frame = "World" if settings.command_frame == "world" else "Heading"
        title = f"EE Velocity ({frame})" + (" + Z Hold" if settings.height_tracking else "")
        with self._server.gui.add_folder(title, order=-1):
            self.manual = self._server.gui.add_checkbox(
                "Manual control", initial_value=self._manual_default,
            )
            self.sliders = [
                self._server.gui.add_slider(
                    label, min=-limit, max=limit, step=0.01, initial_value=0.0,
                    disabled=not self._manual_default or not enabled,
                )
                for label, limit, enabled in zip(
                    ("Vx (m/s)", "Vy (m/s)", "Vz (m/s)",
                     "Wx (rad/s)", "Wy (rad/s)", "Wz (rad/s)"),
                    settings.command_limits, self.enabled_axes,
                )
            ]
            self.stop = self._server.gui.add_button("Stop", icon=viser.Icon.PLAYER_STOP)
            self.stop.on_click(self._stop)
        self.sync_viewer_to_env()

    def sync_viewer_to_env(self):
        manual = self.manual.value
        for slider, enabled in zip(self.sliders, self.enabled_axes):
            slider.disabled = not manual or not enabled
        if manual:
            target = torch.tensor([
                slider.value if enabled else 0.0
                for slider, enabled in zip(self.sliders, self.enabled_axes)
            ], device=self.env.device).expand(self.env.num_envs, -1)
            self.command.set_twist(target)
        elif self._was_manual:
            self.command.external = False
            self.command.external_pose = False
            self.command._resample(torch.arange(self.env.num_envs, device=self.env.device))
            self.command._update_command()
        self._was_manual = manual
        self.env.unwrapped.observation_manager._obs_buffer = None


class ServoPlayPolicy:
    def __init__(self, env, actor):
        from mjlab.utils.buffers import CircularBuffer

        self.env, self.actor = env, actor
        settings = env.command_manager.get_term("ee_twist").settings
        self.history = (
            CircularBuffer(max(1, round(settings.history_seconds / env.step_dt)),
                           env.num_envs, env.device)
            if settings.encoder_mode != "none" else None
        )

    def __call__(self, observations):
        if self.history is None:
            return self.actor(observations)
        from .velowb.env import servo_observation

        reset_ids = (self.env.episode_length_buf == 0).nonzero().flatten()
        self.history.reset(batch_ids=reset_ids)
        self.history.append(servo_observation(self.env))
        return self.actor({"actor": observations["actor"],
                           "history": self.history.buffer.flatten(1)})


def play_servo(checkpoint_file: Path, *, device="cuda:0", automatic=False, paused=False):
    from .velowb.env import ServoConfig, make_env_cfg
    from .velowb.evaluation import checkpoint_metadata
    from .velowb.learning import ServoActor

    saved, metadata = checkpoint_metadata(checkpoint_file)
    settings = ServoConfig(**{**metadata["servo_config"], "num_envs": 1})
    if settings.task != "velocity":
        raise ValueError("EE velocity playback requires a velocity-servo checkpoint, not B0 pose control")
    env = ManagerBasedRlEnv(make_env_cfg(settings, evaluation=True), device=device)
    try:
        recorded_interface = checkpoint_file.parent / "interface.json"
        if recorded_interface.is_file():
            with recorded_interface.open(encoding="utf-8") as stream:
                interface = json.load(stream)
            defaults = dict(zip(interface["joint_state_order"], interface["default_joint_positions"]))
            robot = env.scene["robot"]
            changed = [name for index, name in enumerate(robot.joint_names)
                       if name in defaults and abs(defaults[name] - robot.data.default_joint_pos[0, index].item()) > 1e-6]
            if changed:
                print("WARNING: Initial joint positions differ from this checkpoint's training setup: "
                      + ", ".join(changed) + ". This playback is not a matched-configuration evaluation.", flush=True)
        env.common_step_counter = metadata["env_steps"]
        wrapped = RslRlVecEnvWrapper(env)
        config = metadata["runner_config"]
        actor_options = {key: value for key, value in config["actor"].items() if key != "class_name"}
        actor = ServoActor(wrapped.get_observations(), config["obs_groups"], "actor",
                           wrapped.num_actions, **actor_options).to(device)
        actor.load_state_dict(saved["actor_state_dict"])
        actor.eval()
        print(f"VeloWB play: {checkpoint_file}", flush=True)
        viewer = ServoVelocityViewer(wrapped, ServoPlayPolicy(env, actor),
                                     manual_control=not automatic)
        if paused:
            viewer.pause()
        viewer.run()
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="TRON2A Viser manual velocity playback")
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--checkpoint-file", type=Path)
    parser.add_argument("--agent", choices=("trained", "zero"), default="trained")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.agent == "trained" and (args.checkpoint_file is None or not args.checkpoint_file.is_file()):
        parser.error("--checkpoint-file must point to a .pt file; use --agent zero for model inspection")
    cfg = tron2_flat_env_cfg(args.variant, play=True)
    cfg.commands["twist"] = ManualVelocityCommandCfg(**vars(cfg.commands["twist"]))
    task = f"Mjlab-Velocity-Flat-TRON2A-{args.variant}"
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