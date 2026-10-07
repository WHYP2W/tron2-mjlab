"""Held-out commands and trajectories with complete per-trial failure accounting."""

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.buffers import CircularBuffer
from mjlab.utils.lab_api.math import quat_from_matrix

from .control import waveform
from .env import ServoConfig, make_env_cfg, servo_observation
from .experiment import file_hash, provenance, read_json, write_json
from .learning import ServoActor
from .metrics import AXES, evaluate_trace, read_csv, summarize, write_csv


POSE_SCENARIOS = ("reaching", "line", "circle", "figure_eight", "spline", "orientation")
SCENARIOS = ("zero", "constant", "step", "ramp", "sine", "bandlimited", "minimum_jerk", "frequency", *POSE_SCENARIOS, "suite")


@dataclass
class Case:
    name: str
    kind: str
    axis: int = -1
    sign: int = 1
    frequency: float = 0.2
    seconds: float = 10.0


def make_cases(scenario, *, axes=tuple(range(6)), seconds=10.0, frequency=0.2):
    if scenario not in SCENARIOS or seconds < 0.06 or frequency <= 0:
        raise ValueError("Invalid scenario, duration or frequency")
    if not axes or any(axis not in range(6) for axis in axes):
        raise ValueError("Primitive axes must be selected from 0..5")
    if scenario == "suite":
        cases = []
        for kind in ("zero", "step", "ramp", "sine", "bandlimited", *POSE_SCENARIOS):
            cases += make_cases(kind, axes=axes, seconds=seconds, frequency=frequency)
        return cases
    if scenario == "frequency":
        return [Case(f"frequency_{AXES[axis]}_{rate:g}Hz", "sine", axis, 1, rate,
                     max(seconds, 4 / rate + 0.02))
                for axis in axes for rate in (0.1, 0.2, 0.5, 1.0, 2.0)]
    if scenario in ("constant", "step", "ramp", "sine"):
        return [Case(f"{scenario}_{AXES[axis]}_{'pos' if sign > 0 else 'neg'}", scenario,
                     axis, sign, frequency, seconds) for axis in axes for sign in (-1, 1)]
    return [Case(scenario, scenario, frequency=frequency, seconds=seconds)]


def trajectory_samples(kind, time, *, amplitude=0.05, frequency=0.2):
    angle, rate = 2 * np.pi * frequency * time, 2 * np.pi * frequency
    position = np.zeros((len(time), 3))
    velocity = np.zeros_like(position)
    rotvec, angular = np.zeros_like(position), np.zeros_like(position)
    if kind == "line":
        position[:, 0] = amplitude * np.sin(angle)
        velocity[:, 0] = amplitude * rate * np.cos(angle)
    elif kind == "circle":
        position[:, 0], position[:, 1] = amplitude * (np.cos(angle) - 1), amplitude * np.sin(angle)
        velocity[:, 0], velocity[:, 1] = -amplitude * rate * np.sin(angle), amplitude * rate * np.cos(angle)
    elif kind == "figure_eight":
        position[:, 0], position[:, 1] = amplitude * np.sin(angle), 0.5 * amplitude * np.sin(2 * angle)
        velocity[:, 0], velocity[:, 1] = amplitude * rate * np.cos(angle), amplitude * rate * np.cos(2 * angle)
    elif kind == "spline":
        knots = np.array([[0, 0, 0], [1, 0.4, 0.6], [0, -0.6, 1], [-0.5, 0.4, 0.5], [0, 0, 0]]) * amplitude
        spline = CubicSpline(np.linspace(0, time[-1], len(knots)), knots, bc_type="clamped")
        position, velocity = spline(time), spline(time, 1)
    elif kind == "orientation":
        rotvec[:, 2] = 0.2 * np.sin(angle)
        angular[:, 2] = 0.2 * rate * np.cos(angle)
    elif kind != "reaching":
        raise ValueError(f"Unknown pose trajectory: {kind}")
    return position, Rotation.from_rotvec(rotvec).as_matrix(), np.column_stack((velocity, angular))


def checkpoint_metadata(checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = saved.get("infos") or {}
    if metadata.get("schema_version") != 1 or "servo_config" not in metadata:
        raise ValueError("This is not a VeloWB checkpoint; fixed-arm locomotion policies cannot control this task")
    if metadata["servo_config"]["baseline"] == "B3" and saved.get("adaptation_phase") != "distill":
        raise ValueError("B3 must complete student distillation before deployable evaluation")
    return saved, metadata


def plot_trace(csv_path, output, *, title="VeloWB", smoke=False):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    rows = [row for row in read_csv(csv_path) if row["episode"] == 0]
    if not rows:
        return
    time = np.array([row["time"] for row in rows])
    figure, panels = plt.subplots(3, 2, figsize=(10, 8), sharex=True, constrained_layout=True)
    for axis, panel in enumerate(panels.flat):
        for prefix, label, color, style in (("command", "Command", "#222222", "--"),
                                            ("twist", "MuJoCo", "#0072B2", "-"),
                                            ("estimate", "Estimator", "#D55E00", ":")):
            panel.plot(time, [row[f"{prefix}_{AXES[axis]}"] for row in rows],
                       label=label, color=color, linestyle=style, linewidth=1.2)
        panel.set_ylabel(f"{AXES[axis]} ({'m/s' if axis < 3 else 'rad/s'})")
        panel.grid(alpha=0.2)
    panels[-1, 0].set_xlabel("Time (s)")
    panels[-1, 1].set_xlabel("Time (s)")
    panels[0, 0].legend(fontsize=8)
    figure.suptitle(title + (" [SMOKE / NOT VALIDATED]" if smoke else ""))
    output = Path(output)
    figure.savefig(output.with_suffix(".png"), dpi=160)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)
    if "zero" in title.lower():
        figure, panels = plt.subplots(2, 1, figsize=(9, 5), sharex=True, constrained_layout=True)
        for prefix, label in (("base_position", "Base"), ("position", "EE")):
            position = np.array([[row[f"{prefix}_{axis}"] for axis in range(3)] for row in rows])
            panels[0].plot(time, np.linalg.norm(position - position[0], axis=-1), label=label)
        for prefix, label in (("base_induced", "Base-induced"), ("twist", "EE")):
            velocity = np.array([[row[f"{prefix}_{axis}"] for axis in AXES[:3]] for row in rows])
            panels[1].plot(time, np.linalg.norm(velocity, axis=-1), label=label)
        panels[0].set_ylabel("World displacement (m)")
        panels[1].set_ylabel("Linear speed (m/s)")
        panels[1].set_xlabel("Time (s)")
        for panel in panels:
            panel.legend()
            panel.grid(alpha=0.2)
        figure.suptitle(title + (" [SMOKE / NOT VALIDATED]" if smoke else ""))
        figure.savefig(output.with_name(output.name + "_stabilization").with_suffix(".png"), dpi=160)
        plt.close(figure)


def _trace_rows(trace, episode):
    rows = []
    vector_axes = {"command", "received_command", "target_twist", "tracking_command", "tracking_twist",
                   "twist", "estimate", "base_twist", "base_induced"}
    for sample, time in enumerate(trace["time"]):
        row = {"episode": episode, "time": float(time)}
        for name, values in trace.items():
            if name == "time":
                continue
            value = values[sample]
            if np.ndim(value) == 0:
                row[name] = float(value)
            else:
                labels = AXES if name in vector_axes else range(len(value))
                row.update({f"{name}_{label}": float(component) for label, component in zip(labels, value)})
        rows.append(row)
    return rows


def evaluate_run(settings, directory, *, checkpoint=None, scenario="zero", episodes=20,
                 seconds=10.0, axes=tuple(range(6)), frequency=0.2, amplitude_fraction=0.6,
                 device="cuda:0", condition="nominal", video=False, push_force=0.0,
                 label=None, make_plots=True):
    if episodes < 1 or not 0 < amplitude_fraction <= 1 or push_force < 0:
        raise ValueError("Invalid evaluation trial count, amplitude or disturbance")
    cases = make_cases(scenario, axes=axes, seconds=seconds, frequency=frequency)
    saved, metadata = checkpoint_metadata(checkpoint) if checkpoint else (None, {})
    if saved is None and settings.ik_mode != "only":
        raise ValueError("Learned baselines require a checkpoint; only B5 can be evaluated without one")
    source_settings = ServoConfig(**metadata["servo_config"]) if metadata else settings
    for name in ("variant", "baseline", "lock_arm_yaw", "encoder", "ik", "preview", "history_seconds", "servo_inputs",
                 "control_profile", "command_frame", "height_tracking"):
        if getattr(settings, name) != getattr(source_settings, name):
            raise ValueError(f"Cannot change the trained architecture during evaluation: {name}")
    if tuple(settings.command_limits) != tuple(source_settings.command_limits):
        raise ValueError("Evaluation cannot change the trained command normalization/envelope")
    source_manifest = {}
    if checkpoint and (Path(checkpoint).parent / "manifest.json").is_file():
        source_manifest = read_json(Path(checkpoint).parent / "manifest.json")
    smoke = bool(metadata.get("smoke_only", source_manifest.get("smoke_only", False)))
    label = label or metadata.get("runner_config", {}).get("run_name") or source_settings.baseline
    settings = replace(settings, num_envs=episodes,
                       episode_seconds=max(case.seconds for case in cases) + 1.0)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {**provenance(directory), "kind": "evaluation", "status": "running",
                "smoke_only": smoke, "method": label, "settings": asdict(settings),
                "training_settings": asdict(source_settings), "training_seed": source_settings.seed,
                "training_transitions": metadata.get("env_steps", 0) * source_settings.num_envs,
                "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
                "checkpoint_sha256": file_hash(checkpoint) if checkpoint else None,
                "condition": condition, "amplitude_fraction": amplitude_fraction,
                "tracking_frame": settings.command_frame, "trace_frame": "world",
                "height_feedback": settings.height_tracking,
                "video_resolution": [1280, 720] if video else None,
                "push_force_n": push_force, "cases": [asdict(case) for case in cases],
                "plotted_episode": 0, "baseline_note": "IK-only with fixed leg posture; no independent balance controller"
                if settings.ik_mode == "only" else None}
    write_json(directory / "manifest.json", manifest)
    env = None
    all_metrics = []
    artifacts = {}
    try:
        configuration = make_env_cfg(settings, evaluation=True)
        if video:
            configuration.viewer.width, configuration.viewer.height = 1280, 720
        env = ManagerBasedRlEnv(configuration, device=device,
                                render_mode="rgb_array" if video else None)
        observations, _ = env.reset()
        actor = None
        if saved:
            config = metadata["runner_config"]
            actor_config = {key: value for key, value in config["actor"].items() if key != "class_name"}
            actor = ServoActor(TensorDict(observations, batch_size=[episodes]), config["obs_groups"], "actor",
                               env.action_manager.total_action_dim, **actor_config).to(device)
            actor.load_state_dict(saved["actor_state_dict"])
            actor.eval()
        servo = env.command_manager.get_term("ee_twist")
        for case in cases:
            servo.external = False
            env.reset(seed=settings.seed)
            servo.sense()
            initial_position, initial_rotation = servo.estimate.position.clone(), servo.estimate.rotation.clone()
            target_position, target_rotation = initial_position.clone(), initial_rotation.clone()
            if case.kind == "reaching":
                generator = torch.Generator(device=device).manual_seed(settings.seed + 10_000)
                goal_position = servo.q[:, servo.arm_ids] + 0.5 * (
                    torch.rand(servo.q[:, servo.arm_ids].shape, device=device, generator=generator) - 0.5)
                goal = servo.chain.forward(goal_position.clamp(servo.chain.limits[:, 0], servo.chain.limits[:, 1]))
                target_position = servo.base_pose[:, :3] + (servo.base_rotation @ goal.position.unsqueeze(-1)).squeeze(-1)
                target_rotation = servo.base_rotation @ goal.rotation
            count = max(3, math.ceil(case.seconds / env.step_dt))
            time = np.arange(count) * env.step_dt
            trajectory = (trajectory_samples(case.kind, time, frequency=case.frequency)
                          if case.kind in POSE_SCENARIOS else None)
            history = (CircularBuffer(max(1, round(settings.history_seconds / env.step_dt)), episodes, device)
                       if settings.encoder_mode != "none" else None)
            active = np.ones(episodes, dtype=bool)
            samples, active_samples, failures = [], [], []
            signal_target = torch.zeros((episodes, 6), device=device)
            if case.axis >= 0:
                signal_target[:, case.axis] = amplitude_fraction * servo.limits[case.axis] * case.sign
            else:
                signal_target[:] = amplitude_fraction * servo.limits
            phase = torch.linspace(0, 2 * math.pi, 6, device=device)[None]
            previous_command = torch.zeros_like(signal_target)
            writer = None
            if video:
                import imageio.v2 as imageio
                writer = imageio.get_writer(str(directory / f"{case.name}.mp4"), fps=round(1 / env.step_dt))
            try:
                with torch.inference_mode():
                    for index, moment in enumerate(time):
                        if trajectory is not None:
                            offset, rotation_offset, feedforward = trajectory
                            if case.kind != "reaching":
                                target_position = initial_position + torch.tensor(offset[index], device=device, dtype=torch.float32)
                                target_rotation = torch.tensor(rotation_offset[index], device=device, dtype=torch.float32) @ initial_rotation
                            servo.set_pose(target_position, target_rotation,
                                           torch.tensor(feedforward[index], device=device, dtype=torch.float32).expand(episodes, 6))
                        else:
                            command = waveform(case.kind, torch.full((episodes, 1), moment, device=device),
                                                case.seconds, torch.zeros_like(signal_target), signal_target,
                                                case.frequency, phase if case.kind == "bandlimited" else 0.0)
                            if settings.task == "pose":
                                target_position += env.step_dt * previous_command[:, :3]
                                increments = Rotation.from_rotvec(
                                    env.step_dt * previous_command[:, 3:].cpu().numpy()).as_matrix()
                                target_rotation = torch.tensor(increments, device=device, dtype=torch.float32) @ target_rotation
                                servo.set_pose(target_position, target_rotation)
                                servo.command[:] = command
                                servo.requested_command[:] = command
                            else:
                                servo.set_twist(command)
                            previous_command[:] = command
                        current = env.observation_manager.compute_group("actor")
                        policy_inputs = {"actor": current}
                        if history is not None:
                            history.append(servo_observation(env))
                            policy_inputs["history"] = history.buffer.flatten(1)
                        actions = (actor(TensorDict(policy_inputs, batch_size=[episodes])) if actor else
                                   torch.zeros((episodes, servo.action_dim), device=device))
                        data = servo.robot.data
                        pose = data.site_pose_w[:, servo.ee_id]
                        base_velocity = data.root_link_vel_w
                        offset = pose[:, :3] - data.root_link_pos_w
                        induced = torch.cat((base_velocity[:, :3] + torch.linalg.cross(
                            base_velocity[:, 3:], offset, dim=-1), base_velocity[:, 3:]), dim=-1)
                        requested_world = servo.transform_twist(servo.requested_command, to_world=True, estimated=False)
                        values = {
                            "command": requested_world,
                            "received_command": servo.transform_twist(servo.command, to_world=True, estimated=False),
                            "tracking_command": servo.requested_command,
                            "tracking_twist": servo.transform_twist(data.site_vel_w[:, servo.ee_id], estimated=False),
                            "target_twist": (torch.tensor(trajectory[2][index], device=device, dtype=torch.float32).expand(episodes, 6)
                                             if trajectory is not None else requested_world),
                            "twist": data.site_vel_w[:, servo.ee_id],
                            "estimate": servo.estimate.twist, "position": pose[:, :3], "quaternion": pose[:, 3:],
                            "base_position": data.root_link_pos_w, "base_quaternion": data.root_link_quat_w,
                            "base_twist": base_velocity, "base_induced": induced,
                            "action": env.action_manager.get_term("whole_body").applied,
                            "joint_position": data.joint_pos, "joint_velocity": data.joint_vel,
                            "torque": env.sim.data.qfrc_actuator[:, servo.robot.indexing.joint_v_adr],
                            "target_position": target_position, "target_quaternion": quat_from_matrix(target_rotation),
                            "encoder_timestamp": torch.full((episodes, 1), moment, device=device) - servo.sensor_delay[:, None] * env.step_dt,
                            "odometry_timestamp": torch.full((episodes, 1), moment, device=device) - servo.odom_delay[:, None] * env.step_dt,
                        }
                        if settings.height_tracking:
                            values["height_target"] = servo.height_target[:, None]
                        lengths = {name: value.shape[1] for name, value in values.items()}
                        samples.append(torch.cat(list(values.values()), dim=-1).cpu().numpy())
                        active_samples.append(active.copy())
                        if writer is not None and active[0]:
                            frame = env.render()
                            writer.append_data(frame)
                            if index == 0:
                                imageio.imwrite(directory / f"{case.name}_initial.png", frame)
                        base_id = servo.chain.base_id
                        env.sim.data.xfrc_applied[:, base_id, 0] = (
                            push_force if 0.4 * case.seconds <= moment < 0.4 * case.seconds + 0.2 else 0.0)
                        _, _, terminated, truncated, _ = env.step(actions)
                        done = (terminated | truncated).cpu().numpy().astype(bool)
                        failures.append(done & active)
                        active &= ~done
                        if history is not None:
                            history.reset(batch_ids=torch.as_tensor(np.flatnonzero(done), device=device))
                        if not active.any():
                            break
            finally:
                if writer is not None:
                    writer.close()
            packed = np.stack(samples)
            masks, failures = np.stack(active_samples), np.stack(failures)
            traces = {}
            offset = 0
            for name, length in lengths.items():
                traces[name] = packed[:, :, offset:offset + length]
                offset += length
            rows = []
            for episode in range(episodes):
                selected = masks[:, episode]
                trace = {name: value[selected, episode] for name, value in traces.items()}
                trace.update(time=time[:len(samples)][selected], fallen=failures[selected, episode])
                rows.extend(_trace_rows(trace, episode))
                metrics = (evaluate_trace(trace, command_limits=settings.command_limits,
                                           step_axis=case.axis if case.kind == "step" else None,
                                           frequency=case.frequency if case.kind == "sine" else 0,
                                           pose_task=case.kind in POSE_SCENARIOS)
                           if selected.sum() >= 3 else {"fallen": 1, "insufficient_samples": 1})
                for group, indices in (("leg", servo.leg_ids), ("wheel", servo.wheel_ids), ("arm", servo.arm_ids)):
                    metrics[f"{group}_joint_velocity_rms"] = (
                        float(np.sqrt(np.mean(trace["joint_velocity"][:, indices]**2))) if indices else None)
                metrics.update(method=label, baseline=settings.baseline, variant=settings.variant,
                               control_profile=settings.control_profile, command_frame=settings.command_frame,
                               height_tracking=int(settings.height_tracking),
                               scenario=case.name, condition=condition, episode=episode, seed=settings.seed,
                               training_seed=source_settings.seed, smoke_only=int(smoke))
                all_metrics.append(metrics)
            csv_path = directory / f"{case.name}_trace.csv"
            write_csv(csv_path, rows)
            artifacts[csv_path.name] = file_hash(csv_path)
            if make_plots:
                plot_trace(csv_path, directory / case.name, title=f"{label}: {case.name}", smoke=smoke)
        write_csv(directory / "metrics.csv", all_metrics)
        write_csv(directory / "summary.csv", summarize(all_metrics, ("method", "baseline", "scenario", "condition")))
        manifest.update(status="completed", artifacts=artifacts, metrics_sha256=file_hash(directory / "metrics.csv"),
                        trials=len(all_metrics), failures=sum(row["fallen"] for row in all_metrics),
                        joint_names=list(servo.robot.joint_names))
        write_json(directory / "manifest.json", manifest)
        lines = ["# Result Manifest", "", f"Method: {label}; smoke_only: {smoke}.",
             f"Checkpoint SHA256: {manifest['checkpoint_sha256']}",
             "All trials, including failures, are in metrics.csv. No best-trial selection.", "",
             "## Figures", ""]
        lines.extend(f"- {case.name}.png / .pdf: {case.name}_trace.csv, episode 0, SHA256 {artifacts[f'{case.name}_trace.csv']}"
                 for case in cases)
        (directory / "result_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return manifest
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        write_json(directory / "manifest.json", manifest)
        raise
    finally:
        if env is not None:
            env.close()