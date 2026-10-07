"""Reproducible experiment CLI; long runs require explicit user invocation."""

import argparse
import copy
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
import tarfile

import mujoco
import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from ..robot import get_robot_cfg
from .env import ServoConfig, make_env_cfg
from .kinematics import ArmKinematics
from .learning import runner_config
from .metrics import AXES, write_csv


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "experiments" / "velowb.json"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(directory):
    directory = Path(directory)
    files = sorted((ROOT / "src" / "tron2_mjlab").rglob("*.py"))
    files += [ROOT / "pyproject.toml", ROOT / "uv.lock", ROOT / "README.md"]
    files += sorted((ROOT / "experiments").glob("*.json"))
    assets = ROOT / "src" / "tron2_mjlab" / "tron2-robot-description"
    files += sorted(assets.glob("tron2a/*/xml/robot.xml"))
    files += [assets / "LICENSE", assets / "NOTICE", assets / "THIRD_PARTY_NOTICES.md"]
    files += sorted((ROOT / "docs").glob("*.md"))
    hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in files if path.is_file()}
    asset_hashes = {str(path.relative_to(ROOT)): file_hash(path)
                    for path in sorted(assets.glob("tron2a/*/meshes/*")) if path.is_file()}
    with tarfile.open(directory / "source.tar.gz", "w:gz") as archive:
        for relative in hashes:
            archive.add(ROOT / relative, arcname=relative)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                             capture_output=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True,
                             capture_output=True, check=False)
    versions = {}
    for package in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "rsl-rl-lib", "numpy", "matplotlib"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "git_dirty": bool(status.stdout.strip()), "source_sha256": hashes,
            "mesh_sha256": asset_hashes,
            "source_archive_sha256": file_hash(directory / "source.tar.gz"), "versions": versions}


def load_settings(config_path=DEFAULT_CONFIG, preset="plain_v0", overrides=None):
    document = read_json(config_path)
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported experiment configuration schema")
    if preset not in document["presets"]:
        raise ValueError(f"Unknown experiment preset: {preset}")
    values = {**document.get("defaults", {}), **document["presets"][preset], **(overrides or {})}
    unknown = set(values) - {item.name for item in fields(ServoConfig)}
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    settings = ServoConfig(**values)
    settings.validate()
    return settings, document.get("training", {})


def parse_overrides(assignments):
    result = {}
    for assignment in assignments or []:
        if "=" not in assignment:
            raise ValueError("Overrides must have the form field=JSON-value")
        key, value = assignment.split("=", 1)
        try:
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            result[key] = value
    return result


def validate_estimator(directory, *, variant="WFYG", locked=True, samples=128, seed=42,
                       control_profile="standard"):
    if samples < 1:
        raise ValueError("At least one estimator sample is required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model = Entity(get_robot_cfg(variant, active_arm=True, lock_arm_yaw=locked,
                                  control_profile=control_profile)).spec.compile()
    chain = ArmKinematics(model, dtype=torch.float64)
    data = mujoco.MjData(model)
    generator = np.random.default_rng(seed)
    rows = []
    for mode in ("base", "arm", "combined"):
        for sample in range(samples):
            data.qpos[:] = model.qpos0
            data.qpos[:3] = generator.normal(size=3)
            quaternion = generator.normal(size=4)
            data.qpos[3:7] = quaternion / np.linalg.norm(quaternion)
            limits = model.jnt_range[chain.joint_ids]
            data.qpos[chain.qpos_ids] = generator.uniform(limits[:, 0], limits[:, 1])
            data.qvel[:] = 0
            if mode != "arm":
                data.qvel[:6] = generator.normal(size=6)
            if mode != "base":
                data.qvel[chain.dof_ids] = generator.normal(size=len(chain.dof_ids))
            mujoco.mj_forward(model, data)
            tensor = lambda value: torch.tensor(value, dtype=torch.float64)[None]
            estimate = chain.estimate(tensor(data.qpos[chain.qpos_ids]), tensor(data.qvel[chain.dof_ids]),
                                      tensor(data.qpos[:3]), tensor(data.qpos[3:7]), tensor(data.qvel[:3]),
                                      tensor(data.xmat[chain.base_id].reshape(3, 3) @ data.qvel[3:6]))
            truth = np.empty(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, model.site("ee").id, truth, 0)
            truth = np.r_[truth[3:], truth[:3]]
            predicted = estimate.twist[0].numpy()
            row = {"mode": mode, "sample": sample}
            for axis, name in enumerate(AXES):
                row.update({f"gt_{name}": float(truth[axis]), f"estimate_{name}": float(predicted[axis]),
                            f"error_{name}": float(predicted[axis] - truth[axis])})
            rows.append(row)
    write_csv(directory / "estimator.csv", rows)
    errors = np.array([[row[f"error_{name}"] for name in AXES] for row in rows])
    report = {"gate": "G1", "passed": bool(np.abs(errors).max() < 1e-6), "variant": variant,
              "control_profile": control_profile,
              "lock_arm_yaw": locked, "seed": seed, "samples_per_mode": samples,
              "max_abs_error": float(np.abs(errors).max()),
              "axis_rmse": dict(zip(AXES, np.sqrt(np.mean(errors**2, axis=0)).tolist())),
              "threshold": 1e-6, "csv_sha256": file_hash(directory / "estimator.csv")}
    write_json(directory / "g1.json", report)
    return report


class ExperimentRunner(OnPolicyRunner):
    def __init__(self, env, config, settings, directory, device, smoke=False):
        self.settings = settings
        self.smoke = smoke
        self.config_snapshot = copy.deepcopy(config)
        super().__init__(env, config, str(directory), device)

    def save(self, path, infos=None):
        metadata = {"schema_version": 1, "servo_config": asdict(self.settings),
                    "smoke_only": self.smoke,
                    "runner_config": self.config_snapshot,
                    "env_steps": self.env.unwrapped.common_step_counter}
        super().save(path, infos={**metadata, **(infos or {})})


def train_run(settings, directory, *, iterations=10_000, steps_per_env=24,
              learning_epochs=5, mini_batches=4, teacher_fraction=0.7,
              device="cuda:0", smoke=False, gate_report=None, resume=None, label=None):
    settings.validate()
    if min(iterations, steps_per_env, learning_epochs, mini_batches) < 1:
        raise ValueError("Training budgets must be positive")
    if settings.num_envs * steps_per_env < mini_batches:
        raise ValueError("There are fewer rollout samples than minibatches")
    if smoke and iterations > 2:
        raise ValueError("Smoke mode is limited to two updates and is not a research result")
    if settings.encoder_mode == "two_stage" and (iterations < 2 or not 0 < teacher_fraction < 1):
        raise ValueError("Two-stage training needs at least two updates and a teacher fraction in (0, 1)")
    advanced = settings.encoder_mode != "none" or settings.ik_mode != "none"
    gate = read_json(gate_report) if gate_report else None
    if advanced and not smoke and (not gate or gate.get("gate") != "G2" or not gate.get("passed")
                                   or gate.get("variant") != settings.variant
                                   or gate.get("lock_arm_yaw") != settings.lock_arm_yaw
                                   or gate.get("control_profile", "standard") != settings.control_profile):
        raise ValueError("Advanced training requires a passing G2 report (--gate-report); use --smoke only for plumbing checks")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {**provenance(directory), "kind": "training", "status": "running",
                "smoke_only": smoke, "settings": asdict(settings), "gate_report": gate,
                "requested_iterations": iterations, "steps_per_env": steps_per_env,
                "teacher_fraction": teacher_fraction, "device": device}
    write_json(directory / "manifest.json", manifest)
    env = None
    try:
        report = validate_estimator(directory / "estimator", variant=settings.variant,
                                     locked=settings.lock_arm_yaw, samples=16, seed=settings.seed,
                                     control_profile=settings.control_profile)
        if not report["passed"]:
            raise RuntimeError("G1 estimator validation failed; training is disabled")
        env = ManagerBasedRlEnv(make_env_cfg(settings), device=device)
        wrapped = RslRlVecEnvWrapper(env)
        servo = env.command_manager.get_term("ee_twist")
        joint_names = list(servo.robot.joint_names)
        action = env.action_manager.get_term("whole_body")
        groups = [("legs", servo.leg_ids), ("arm", servo.arm_ids), ("wheels", servo.wheel_ids)]
        if settings.control_profile == "standard":
            groups = [groups[0], groups[2], groups[1]]
        write_json(directory / "interface.json", {
            "version": 1, "deployment_ready": False, "quaternion": "wxyz",
            "twist_order": list(AXES), "twist_frame": settings.command_frame, "base_reference": "base_Link origin",
            "control_profile": settings.control_profile, "height_tracking": settings.height_tracking,
            "policy_uses_pose_error": settings.height_tracking or settings.task == "pose",
            "ee_site": "gripper_pick/ee", "policy_dt_s": env.step_dt, "physics_dt_s": env.physics_dt,
            "joint_state_order": joint_names,
            "action_order": {name: [joint_names[index] for index in indices] for name, indices in groups},
            "action_vector_order": [joint_names[index] for _, indices in groups for index in indices],
            "observation_shapes": {name: list(shape) for name, shape in env.observation_manager.group_obs_dim.items()},
            "default_joint_positions": servo.robot.data.default_joint_pos[0].cpu().tolist(),
            "command_limits": list(settings.command_limits), "ik_mode": settings.ik_mode,
            "history_order": "oldest to newest", "history_samples": max(1, round(settings.history_seconds / env.step_dt)),
            "scales": ({"position_joint_names": [joint_names[index] for index in action.position_ids],
                        "position_rad": action.position_scale.cpu().tolist(), "wheel_velocity_rad_s": 7.0}
                       if settings.control_profile == "wbc" else
                       {"leg_position_rad": 0.25, "wheel_velocity_rad_s": 10.0,
                        "arm_integrated_velocity_rad_s": 1.5, "residual_bound": settings.residual_bound}),
            "arm_target_mode": "position_residual" if settings.control_profile == "wbc" else "integrated_velocity",
            "actuator_delay_physics_steps": [0, settings.actuator_delay_steps],
        })
        config = runner_config(settings, iterations=iterations, steps_per_env=steps_per_env)
        config["run_name"] = label or settings.baseline
        boundary = max(1, min(iterations - 1, round(iterations * teacher_fraction)))
        if settings.encoder_mode == "two_stage":
            config["teacher_iterations"] = boundary
        config["algorithm"].update(num_learning_epochs=learning_epochs, num_mini_batches=mini_batches)
        write_json(directory / "config.json", {"settings": asdict(settings), "runner": config})
        runner = ExperimentRunner(wrapped, copy.deepcopy(config), settings, directory, device, smoke)
        start = 0
        if resume:
            loaded = torch.load(resume, map_location=device, weights_only=False)
            metadata = loaded.get("infos") or {}
            previous_settings = metadata.get("servo_config")
            if (previous_settings is None or
                    json.dumps(asdict(ServoConfig(**previous_settings)), sort_keys=True) !=
                    json.dumps(asdict(settings), sort_keys=True)):
                raise ValueError("Resume requires the exact same servo configuration; locomotion checkpoints are not compatible")
            runner.load(str(resume), map_location=device)
            start = runner.current_learning_iteration + 1
            runner.current_learning_iteration = start
            env.common_step_counter = metadata["env_steps"]
            if settings.encoder_mode == "two_stage":
                boundary = metadata["runner_config"].get("teacher_iterations", boundary)
                runner.config_snapshot["teacher_iterations"] = boundary
            manifest["resumed_from"] = {"path": str(Path(resume).resolve()), "sha256": file_hash(resume)}
        if start >= iterations:
            raise ValueError("Requested iteration budget has already been completed")
        if settings.encoder_mode == "two_stage":
            if start < boundary:
                runner.learn(boundary - start)
                runner.current_learning_iteration = boundary
            runner.alg.set_distillation_phase()
            env.command_manager.get_term("ee_twist").teacher_mask.zero_()
            env.reset()
            runner.learn(iterations - max(boundary, start))
        else:
            runner.learn(iterations - start)
        checkpoint = directory / f"model_{runner.current_learning_iteration}.pt"
        runner.alg.actor.eval()
        exported = torch.jit.script(runner.alg.actor.as_jit().cpu())
        exported.save(str(directory / "student_policy.pt"))
        manifest.update(status="completed", checkpoint=checkpoint.name, checkpoint_sha256=file_hash(checkpoint),
                        interface_sha256=file_hash(directory / "interface.json"),
                        completed_iterations=runner.current_learning_iteration + 1,
                        environment_steps=env.common_step_counter,
                        rollout_transitions=env.common_step_counter * settings.num_envs,
                        actor_parameters=sum(value.numel() for value in runner.alg.actor.parameters()),
                        critic_parameters=sum(value.numel() for value in runner.alg.critic.parameters()),
                        adaptation_phase=runner.alg.phase,
                        student_policy_sha256=file_hash(directory / "student_policy.pt"))
        write_json(directory / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=str(error))
        write_json(directory / "manifest.json", manifest)
        raise
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description="VeloWB staged paper experiments (simulation only)")
    commands = parser.add_subparsers(dest="command", required=True)
    estimator = commands.add_parser("validate-estimator")
    estimator.add_argument("--variant", choices=("WFYG", "SFYG"), default="WFYG")
    estimator.add_argument("--free-yaw", action="store_true")
    estimator.add_argument("--control-profile", choices=("standard", "wbc"), default="standard")
    estimator.add_argument("--samples", type=int, default=128)
    estimator.add_argument("--seed", type=int, default=42)
    estimator.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--preset", default="plain_v0")
    train.add_argument("--matrix", type=Path)
    train.add_argument("--run-id")
    train.add_argument("--set", action="append", default=[])
    train.add_argument("--seed", type=int)
    train.add_argument("--num-envs", type=int)
    train.add_argument("--iterations", type=int)
    train.add_argument("--steps-per-env", type=int)
    train.add_argument("--learning-epochs", type=int)
    train.add_argument("--mini-batches", type=int)
    train.add_argument("--output", type=Path)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--resume", type=Path)
    train.add_argument("--gate-report", type=Path)
    train.add_argument("--smoke", action="store_true")
    play = commands.add_parser("play", help="Interactive Viser playback with EE velocity controls")
    play.add_argument("--checkpoint-file", "--checkpoint", type=Path, required=True)
    play.add_argument("--device", default="cuda:0")
    play.add_argument("--viewer", choices=("viser",), default="viser")
    play.add_argument("--auto", action="store_true", help="Start with automatic training commands")
    play.add_argument("--paused", action="store_true", help="Start paused at the reset pose")
    evaluate = commands.add_parser("eval")
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    evaluate.add_argument("--preset", default="B5")
    evaluate.add_argument("--set", action="append", default=[])
    evaluate.add_argument("--ood", default="nominal")
    evaluate.add_argument("--scenario", default="zero")
    evaluate.add_argument("--episodes", type=int, default=20)
    evaluate.add_argument("--seconds", type=float, default=10.0)
    evaluate.add_argument("--axis", default="all")
    evaluate.add_argument("--frequency", type=float, default=0.2)
    evaluate.add_argument("--amplitude-fraction", type=float, default=0.6)
    evaluate.add_argument("--push-force", type=float, default=0.0)
    evaluate.add_argument("--seed", type=int, default=1000)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--video", action="store_true")
    evaluate.add_argument("--output", type=Path, required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    plan.add_argument("--suite", choices=("baselines", "preview", "ablations", "wbc", "all"), default="baselines")
    plan.add_argument("--gate-report", type=Path)
    plan.add_argument("--output", type=Path, required=True)
    gate = commands.add_parser("gate")
    gate.add_argument("--zero-run", type=Path, required=True)
    gate.add_argument("--tracking-run", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--runs", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    plot = commands.add_parser("plot")
    plot.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "validate-estimator":
        report = validate_estimator(args.output, variant=args.variant, locked=not args.free_yaw,
                                     samples=args.samples, seed=args.seed, control_profile=args.control_profile)
        print(json.dumps(report, indent=2))
    elif args.command == "train":
        overrides = parse_overrides(args.set)
        for name in ("seed", "num_envs"):
            if getattr(args, name) is not None:
                overrides[name] = getattr(args, name)
        label = args.preset
        if args.matrix:
            matrix = read_json(args.matrix)
            selected = [run for run in matrix["runs"] if run["id"] == args.run_id]
            if len(selected) != 1 or selected[0]["kind"] != "train":
                raise ValueError("Select a valid training --run-id from the matrix")
            record = selected[0]
            settings = ServoConfig(**{**record["settings"], **overrides})
            budget = dict(record["training"])
            label = record["method"]
        else:
            settings, budget = load_settings(args.config, args.preset, overrides)
        for name in ("iterations", "steps_per_env", "learning_epochs", "mini_batches"):
            if getattr(args, name) is not None:
                budget[name] = getattr(args, name)
        directory = args.output or ROOT / "logs" / "velowb" / (
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f") + f"_{label}_seed{settings.seed}")
        manifest = train_run(settings, directory, **budget, device=args.device, smoke=args.smoke,
                             gate_report=args.gate_report, resume=args.resume, label=label)
        print(f"Training artifacts: {directory.resolve()}")
        print(f"Checkpoint: {manifest['checkpoint']}; smoke_only={manifest['smoke_only']}")
    elif args.command == "play":
        from ..play import play_servo
        play_servo(args.checkpoint_file, device=args.device, automatic=args.auto, paused=args.paused)
    elif args.command == "eval":
        from .evaluation import checkpoint_metadata, evaluate_run
        source = read_json(args.config)
        if args.ood not in source["ood"]:
            raise ValueError(f"Unknown OOD condition: {args.ood}")
        overrides = {**source["ood"][args.ood], **parse_overrides(args.set), "seed": args.seed}
        if args.checkpoint:
            _, metadata = checkpoint_metadata(args.checkpoint)
            settings = ServoConfig(**{**metadata["servo_config"], **overrides})
        else:
            settings, _ = load_settings(args.config, args.preset, overrides)
        axes = tuple(range(6)) if args.axis == "all" else (int(args.axis),)
        manifest = evaluate_run(settings, args.output, checkpoint=args.checkpoint, scenario=args.scenario,
                                 episodes=args.episodes, seconds=args.seconds, axes=axes,
                                 frequency=args.frequency, amplitude_fraction=args.amplitude_fraction,
                                 push_force=args.push_force, device=args.device, condition=args.ood, video=args.video)
        print(f"Evaluation artifacts: {args.output.resolve()}")
        print(f"Trials: {manifest['trials']}; failures: {manifest['failures']}; smoke_only={manifest['smoke_only']}")
    elif args.command == "plan":
        from .study import plan_study
        matrix = plan_study(args.output, config_path=args.config, suite=args.suite, gate_report=args.gate_report)
        print(f"Planned {len(matrix['runs'])} runs in {args.output.resolve()}; no jobs launched")
    elif args.command == "gate":
        from .study import g2_gate
        report = g2_gate(args.zero_run, args.tracking_run, args.output)
        print(json.dumps(report, indent=2))
    elif args.command == "aggregate":
        from .study import aggregate_runs
        summaries = aggregate_runs(args.runs, args.output)
        print(f"Wrote {len(summaries)} seed-level summaries to {args.output.resolve()}")
    elif args.command == "plot":
        from .study import plot_run
        plot_run(args.run)
        print(f"Regenerated plots from recorded CSV: {args.run.resolve()}")


if __name__ == "__main__":
    main()