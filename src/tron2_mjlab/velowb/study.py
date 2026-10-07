"""Predeclared experiment matrices, stage gates and seed-level aggregation."""

from dataclasses import asdict
from pathlib import Path
import shlex

import numpy as np

from .env import ServoConfig
from .experiment import DEFAULT_CONFIG, file_hash, load_settings, read_json, write_json
from .metrics import read_csv, summarize, write_csv


def plan_study(directory, *, config_path=DEFAULT_CONFIG, suite="baselines", gate_report=None):
    document = read_json(config_path)
    if suite not in ("baselines", "preview", "ablations", "wbc", "all"):
        raise ValueError("Unknown experiment suite")
    methods = []
    if suite in ("baselines", "all"):
        methods.extend((f"B{index}", f"B{index}", {}) for index in range(8))
    if suite in ("wbc", "all"):
        methods.extend((name, name, {}) for name in ("wbc_plain", "wbc_cts", "wbc_height"))
    if suite in ("preview", "all"):
        methods.extend((name, name, {}) for name in document["presets"] if name.startswith("K"))
    if suite in ("ablations", "all"):
        for name, specification in document["ablations"].items():
            methods.extend((f"{name}_{index}", specification["preset"], {specification["field"]: value})
                           for index, value in enumerate(specification["values"]))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    matrix_path = directory / "matrix.json"
    runs = []
    for label, preset, overrides in methods:
        for seed in document["seeds"]:
            settings, training = load_settings(config_path, preset, {**overrides, "seed": seed})
            run_id = f"{label}_seed{seed}"
            learnable = settings.ik_mode != "only"
            if learnable:
                command = ["uv", "run", "velowb", "train", "--matrix", str(matrix_path.resolve()), "--run-id", run_id]
                if gate_report:
                    command += ["--gate-report", str(Path(gate_report).resolve())]
            else:
                command = ["uv", "run", "velowb", "eval", "--config", str(Path(config_path).resolve()),
                           "--preset", preset, "--scenario", "suite", "--seed", str(1000 + seed),
                           "--output", str((directory / run_id).resolve())]
            runs.append({"id": run_id, "method": label, "kind": "train" if learnable else "ik_evaluate",
                         "settings": asdict(settings), "training": training,
                         "transitions": settings.num_envs * training["steps_per_env"] * training["iterations"]
                         if learnable else 0, "command": command})
    matrix = {"schema_version": 1, "suite": suite, "config": str(Path(config_path).resolve()),
              "config_sha256": file_hash(config_path), "runs": runs,
              "notes": ["No jobs were launched.", "All learned methods use the same total rollout budget.",
                        "B3 divides that budget into teacher and student stages.",
                        "Advanced non-smoke training requires a passing G2 report."]}
    write_json(matrix_path, matrix)
    (directory / "commands.txt").write_text("\n".join(shlex.join(run["command"]) for run in runs) + "\n", encoding="utf-8")
    return matrix


def g2_gate(zero_run, tracking_run, output, *, max_error=0.5, max_fall_rate=0.05,
            max_transmission=0.7, min_base_motion=0.01, min_duration=5.0, min_trials=5):
    zero_run, tracking_run = Path(zero_run), Path(tracking_run)
    zero_manifest, tracking_manifest = read_json(zero_run / "manifest.json"), read_json(tracking_run / "manifest.json")
    zeros = [row for row in read_csv(zero_run / "metrics.csv") if row["scenario"] == "zero"]
    tracking = [row for row in read_csv(tracking_run / "metrics.csv")
                if str(row["scenario"]).startswith(("step_", "ramp_", "constant_"))]
    settings = ServoConfig(**zero_manifest["training_settings"])

    def mean(rows, key):
        values = [row[key] for row in rows if isinstance(row.get(key), (float, int)) and np.isfinite(row[key])]
        return float(np.mean(values)) if values else None

    fall_rate = mean(zeros + tracking, "fallen")
    error = mean(tracking, "weighted_twist_rmse")
    transmission = mean(zeros, "disturbance_transmission_linear")
    motion = mean(zeros, "base_induced_linear_rms")
    required_axes = ("vx", "vy", "vz", "wz") if settings.command_axes == "linear_yaw" else ("vx", "vy", "vz", "wx", "wy", "wz")
    covered = all(sum(f"_{axis}_{sign}" in row["scenario"] for row in tracking) >= min_trials
                  for axis in required_axes for sign in ("pos", "neg"))
    all_rows = zeros + tracking
    criteria = {
        "plain_servo": settings.baseline == "B1" and settings.encoder_mode == "none"
        and settings.ik_mode == "none" and settings.preview == "none"
        and not settings.height_tracking and settings.command_frame == "world",
        "completed_evaluations": zero_manifest["status"] == tracking_manifest["status"] == "completed",
        "not_smoke": not zero_manifest.get("smoke_only", True) and not tracking_manifest.get("smoke_only", True),
        "same_checkpoint": zero_manifest.get("checkpoint_sha256") is not None
        and zero_manifest["checkpoint_sha256"] == tracking_manifest.get("checkpoint_sha256"),
        "enough_trials_and_axes": len(zeros) >= min_trials and covered,
        "duration": bool(all_rows) and all(row.get("duration_s", 0) >= min_duration for row in all_rows),
        "survival": fall_rate is not None and fall_rate <= max_fall_rate,
        "tracking": error is not None and error <= max_error,
        "base_is_moving": motion is not None and motion >= min_base_motion,
        "disturbance_rejection": transmission is not None and transmission <= max_transmission,
    }
    report = {"gate": "G2", "passed": all(criteria.values()), "variant": settings.variant,
              "control_profile": settings.control_profile,
              "lock_arm_yaw": settings.lock_arm_yaw, "criteria": criteria,
              "measurements": {"fall_rate": fall_rate, "weighted_twist_rmse": error,
                               "transmission": transmission, "base_induced_linear_rms": motion},
              "thresholds": {"max_error": max_error, "max_fall_rate": max_fall_rate,
                             "max_transmission": max_transmission, "min_base_motion": min_base_motion,
                             "min_duration": min_duration, "min_trials": min_trials},
              "sources": [{"path": str(path.resolve()), "metrics_sha256": file_hash(path / "metrics.csv"),
                           "manifest_sha256": file_hash(path / "manifest.json")} for path in (zero_run, tracking_run)]}
    write_json(output, report)
    return report


def aggregate_runs(runs, directory):
    if not runs:
        raise ValueError("At least one evaluation run is required")
    rows, manifests, sources = [], [], []
    unique_trials = set()
    protocols = {}
    for run in map(Path, runs):
        manifest = read_json(run / "manifest.json")
        if manifest["kind"] != "evaluation" or manifest["status"] != "completed":
            raise ValueError(f"Not a completed evaluation: {run}")
        manifests.append(manifest)
        for case in manifest["cases"]:
            key = (case["name"], manifest["condition"])
            protocol = (case, manifest["amplitude_fraction"], manifest["push_force_n"],
                        tuple(manifest["settings"]["command_limits"]),
                        manifest["settings"].get("command_frame", "world"))
            if key in protocols and protocols[key] != protocol:
                raise ValueError(f"Mismatched evaluation protocols for {key}")
            protocols[key] = protocol
        for row in read_csv(run / "metrics.csv"):
            key = tuple(row[name] for name in ("method", "scenario", "condition", "training_seed", "seed", "episode"))
            if key in unique_trials:
                raise ValueError("Duplicate trials would create pseudoreplication")
            unique_trials.add(key)
            rows.append(row)
        sources.append({"path": str(run.resolve()), "metrics_sha256": file_hash(run / "metrics.csv"),
                        "checkpoint_sha256": manifest["checkpoint_sha256"]})
    if len({manifest["smoke_only"] for manifest in manifests}) > 1:
        raise ValueError("Smoke and research evaluations must not be pooled")
    budgets = {manifest.get("training_transitions") for manifest in manifests if manifest["checkpoint"]}
    if len(budgets) > 1:
        raise ValueError("Learned baselines have different training rollout budgets")
    seed_groups = ("method", "baseline", "variant", "scenario", "condition", "training_seed")
    seed_summary = summarize(rows, seed_groups)
    seed_rows = []
    for summary in seed_summary:
        row = {key: summary[key] for key in seed_groups}
        row.update({key[:-5]: value for key, value in summary.items() if key.endswith("_mean")})
        row["trial_count"] = summary["n"]
        seed_rows.append(row)
    summaries = summarize(seed_rows, seed_groups[:-1])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_csv(directory / "seed_metrics.csv", seed_rows)
    write_csv(directory / "summary.csv", summaries)
    write_json(directory / "manifest.json", {
        "kind": "aggregate", "status": "completed", "smoke_only": manifests[0]["smoke_only"],
        "sources": sources, "unit_of_replication": "training_seed",
        "summary_sha256": file_hash(directory / "summary.csv"),
        "note": "Means first pool trials within a training seed. Standard deviations are across seed means; n=1 has no std.",
    })
    _plot_summary(summaries, directory, smoke=manifests[0]["smoke_only"])
    lines = ["# Result Manifest", "", "Replication unit: training seed. Failures are retained.", "",
             "Tables: seed_metrics.csv; summary.csv. Figures are generated from summary.csv.", ""]
    lines.extend(f"- {source['path']}: checkpoint SHA256 {source['checkpoint_sha256']}; metrics SHA256 {source['metrics_sha256']}"
                 for source in sources)
    (directory / "result_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summaries


def _plot_summary(rows, directory, *, smoke=False):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    methods = list(dict.fromkeys(row["method"] for row in rows))
    scenarios = list(dict.fromkeys((row["scenario"], row["condition"]) for row in rows))
    for metric in ("weighted_twist_rmse", "fallen", "position_rmse_m", "disturbance_transmission_linear"):
        table = np.full((len(methods), len(scenarios)), np.nan)
        for row in rows:
            if f"{metric}_mean" in row:
                table[methods.index(row["method"]), scenarios.index((row["scenario"], row["condition"]))] = row[f"{metric}_mean"]
        if not np.isfinite(table).any():
            continue
        figure, panel = plt.subplots(figsize=(max(7, 0.45 * len(scenarios)), max(3, 0.4 * len(methods))),
                                      constrained_layout=True)
        image = panel.imshow(np.ma.masked_invalid(table), aspect="auto", cmap="viridis")
        panel.set_yticks(range(len(methods)), methods)
        panel.set_xticks(range(len(scenarios)), [f"{scene}/{condition}" for scene, condition in scenarios], rotation=90)
        panel.set_title(metric + (" [SMOKE / NOT VALIDATED]" if smoke else ""))
        figure.colorbar(image, ax=panel)
        figure.savefig(Path(directory) / f"{metric}.png", dpi=160)
        figure.savefig(Path(directory) / f"{metric}.pdf")
        plt.close(figure)


def plot_run(directory):
    directory = Path(directory)
    if (directory / "g1.json").is_file():
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        rows = read_csv(directory / "estimator.csv")
        figure, panel = plt.subplots(figsize=(9, 4), constrained_layout=True)
        for axis in ("vx", "vy", "vz", "wx", "wy", "wz"):
            panel.plot([row[f"error_{axis}"] for row in rows], label=axis, linewidth=0.8)
        panel.set_xlabel("Validation sample (base / arm / combined)")
        panel.set_ylabel("Estimate - MuJoCo (m/s or rad/s)")
        panel.legend(ncol=6)
        figure.savefig(directory / "estimator_error.png", dpi=160)
        figure.savefig(directory / "estimator_error.pdf")
        plt.close(figure)
        return
    manifest = read_json(directory / "manifest.json")
    if manifest["kind"] == "aggregate":
        _plot_summary(read_csv(directory / "summary.csv"), directory, smoke=manifest["smoke_only"])
    elif manifest["kind"] == "evaluation":
        from .evaluation import plot_trace
        for case in manifest["cases"]:
            plot_trace(directory / f"{case['name']}_trace.csv", directory / case["name"],
                       title=f"{manifest['method']}: {case['name']}", smoke=manifest["smoke_only"])
    else:
        raise ValueError("Plot an evaluation, aggregate, or estimator output directory")