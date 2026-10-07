"""Control-system metrics; unavailable measurements stay missing, never zero."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


AXES = ("vx", "vy", "vz", "wx", "wy", "wz")


def settling_time(time, inside, *, origin=0.0, minimum_hold=0.5):
    outside = np.flatnonzero(~np.asarray(inside, dtype=bool))
    index = int(outside[-1] + 1) if len(outside) else 0
    if index >= len(time) or time[-1] - time[index] < minimum_hold:
        return None
    return float(time[index] - origin)


def step_response(time, command, measured, *, relative_tolerance=0.02,
                  absolute_tolerance=0.005, minimum_hold=0.5):
    result = {"rise_time_s": None, "settling_time_s": None, "overshoot_pct": None}
    changes = np.flatnonzero(np.abs(np.diff(command)) > 1e-8)
    if len(changes) != 1:
        return result
    onset = int(changes[0] + 1)
    amplitude = command[-1] - command[0]
    if abs(amplitude) < 1e-8:
        return result
    response = (measured[onset:] - command[0]) / amplitude
    low, high = np.flatnonzero(response >= 0.1), np.flatnonzero(response >= 0.9)
    if len(low) and len(high):
        result["rise_time_s"] = float(time[onset + high[0]] - time[onset + low[0]])
    result["overshoot_pct"] = float(max(0.0, np.max(response) - 1) * 100)
    tolerance = max(absolute_tolerance, relative_tolerance * abs(amplitude))
    result["settling_time_s"] = settling_time(
        time[onset:], np.abs(measured[onset:] - command[-1]) <= tolerance,
        origin=time[onset], minimum_hold=minimum_hold)
    return result


def frequency_response(time, command, measured, frequency):
    result = {"gain": None, "phase_lag_deg": None}
    if frequency <= 0 or (time[-1] - time[0]) * frequency < 3:
        return result
    selected = time >= time[0] + 1 / frequency
    phase = 2 * np.pi * frequency * time[selected]
    design = np.column_stack((np.sin(phase), np.cos(phase), np.ones_like(phase)))
    command_fit = np.linalg.lstsq(design, command[selected], rcond=None)[0]
    measured_fit = np.linalg.lstsq(design, measured[selected], rcond=None)[0]
    amplitude = np.hypot(command_fit[0], command_fit[1])
    if amplitude <= 1e-8:
        return result
    result["gain"] = float(np.hypot(measured_fit[0], measured_fit[1]) / amplitude)
    difference = np.arctan2(command_fit[1], command_fit[0]) - np.arctan2(measured_fit[1], measured_fit[0])
    result["phase_lag_deg"] = float(np.degrees(np.angle(np.exp(1j * difference))))
    return result


def rms(value):
    return float(np.sqrt(np.mean(np.square(value))))


def _rotation(quaternion):
    return Rotation.from_quat(np.asarray(quaternion)[:, [1, 2, 3, 0]])


def evaluate_trace(trace, *, command_limits=(0.15, 0.1, 0.1, 0.25, 0.25, 0.3),
                   step_axis=None, frequency=0.0, pose_task=False,
                   position_tolerance=0.02, orientation_tolerance=0.05):
    time = np.asarray(trace["time"], dtype=float)
    if len(time) < 3 or not np.all(np.diff(time) > 0):
        raise ValueError("Metrics require at least three strictly increasing samples")
    command = np.asarray(trace["target_twist"] if pose_task and "target_twist" in trace
                         else trace.get("tracking_command", trace["command"]))
    world_twist = np.asarray(trace["twist"])
    measured = world_twist if pose_task else np.asarray(trace.get("tracking_twist", world_twist))
    if command.shape != (len(time), 6) or measured.shape != command.shape:
        raise ValueError("Twist arrays must have shape (samples, 6)")
    error = measured - command
    fallen = bool(np.any(trace.get("fallen", False)))
    result = {
        "duration_s": float(time[-1] - time[0]), "fallen": int(fallen),
        "velocity_rmse": rms(error[:, :3]), "angular_velocity_rmse": rms(error[:, 3:]),
        "weighted_twist_rmse": rms(error / np.asarray(command_limits)),
        "ee_linear_rms": rms(measured[:, :3]), "ee_angular_rms": rms(measured[:, 3:]),
        "steady_linear_error": rms(error[-max(3, len(time) // 5):, :3]),
        "steady_angular_error": rms(error[-max(3, len(time) // 5):, 3:]),
        "ee_acceleration_rms": rms(np.gradient(world_twist[:, :3], time, axis=0)),
        "ee_jerk_rms": rms(np.gradient(np.gradient(world_twist[:, :3], time, axis=0), time, axis=0)),
    }
    for axis, name in enumerate(AXES):
        result[f"rmse_{name}"] = rms(error[:, axis])
        if frequency > 0:
            for metric, value in frequency_response(time, command[:, axis], measured[:, axis], frequency).items():
                result[f"{metric}_{name}"] = value
    if step_axis is not None:
        result.update(step_response(time, command[:, step_axis], measured[:, step_axis]))
        if fallen:
            result["settling_time_s"] = None
    if "estimate" in trace:
        result["estimator_rmse"] = rms(trace["estimate"] - world_twist)
    if "height_target" in trace:
        result["height_error_rmse_m"] = rms(np.asarray(trace["height_target"]).reshape(-1) - trace["position"][:, 2])
    if "position" in trace:
        displacement = trace["position"] - trace["position"][0]
        result["ee_position_drift_max_m"] = float(np.linalg.norm(displacement, axis=-1).max())
        result["ee_position_drift_final_m"] = float(np.linalg.norm(displacement[-1]))
    if "quaternion" in trace:
        rotations = _rotation(trace["quaternion"])
        drift = (rotations * rotations[0].inv()).magnitude()
        result["ee_orientation_drift_max_rad"] = float(drift.max())
    if "base_quaternion" in trace:
        angles = _rotation(trace["base_quaternion"]).as_euler("xyz")
        result["base_roll_rms_rad"], result["base_pitch_rms_rad"] = rms(angles[:, 0]), rms(angles[:, 1])
        result["base_tilt_peak_rad"] = float(np.arccos(np.clip(
            _rotation(trace["base_quaternion"]).as_matrix()[:, 2, 2], -1, 1)).max())
    if "base_twist" in trace:
        result["base_linear_rms"] = rms(trace["base_twist"][:, :3])
        result["base_angular_rms"] = rms(trace["base_twist"][:, 3:])
    if "base_induced" in trace:
        for label, selected in (("linear", slice(0, 3)), ("angular", slice(3, 6))):
            denominator = rms(trace["base_induced"][:, selected])
            result[f"base_induced_{label}_rms"] = denominator
            result[f"disturbance_transmission_{label}"] = (
                rms(measured[:, selected]) / denominator if denominator > 1e-6 else None)
    if "action" in trace:
        result["action_rate_rms"] = rms(np.gradient(trace["action"], time, axis=0))
    if "joint_velocity" in trace:
        result["joint_jerk_rms"] = rms(np.gradient(
            np.gradient(trace["joint_velocity"], time, axis=0), time, axis=0))
        if "torque" in trace:
            power = np.abs(trace["torque"] * trace["joint_velocity"]).sum(axis=-1)
            result["mean_power_w"] = float(power.mean())
            result["energy_j"] = float(np.trapezoid(power, time))
    if pose_task:
        position_error = np.linalg.norm(trace["target_position"] - trace["position"], axis=-1)
        orientation_error = (_rotation(trace["target_quaternion"]) * _rotation(trace["quaternion"]).inv()).magnitude()
        inside = (position_error <= position_tolerance) & (orientation_error <= orientation_tolerance)
        result.update(
            position_rmse_m=rms(position_error), orientation_rmse_rad=rms(orientation_error),
            final_position_error_m=float(position_error[-1]),
            final_orientation_error_rad=float(orientation_error[-1]),
            pose_success=int(not fallen and bool(inside[-1])),
            pose_settling_time_s=None if fallen else settling_time(time, inside, origin=time[0]),
        )
        displacement = trace["target_position"][0] - trace["position"][0]
        distance = float(np.linalg.norm(displacement))
        if distance > 1e-6 and np.allclose(trace["target_position"], trace["target_position"][0]):
            progress = (trace["position"] - trace["position"][0]) @ displacement / distance**2
            result["pose_overshoot_pct"] = float(max(0.0, progress.max() - 1) * 100)
        else:
            result["pose_overshoot_pct"] = None
    return result


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to write an empty result table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value == "":
                row[key] = None
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


def summarize(rows, group_by=("baseline", "scenario")):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_by)].append(row)
    summaries = []
    excluded = set(group_by) | {"episode", "seed", "training_seed"}
    for group, samples in groups.items():
        summary = dict(zip(group_by, group))
        summary["n"] = len(samples)
        for key in dict.fromkeys(key for sample in samples for key in sample):
            if key in excluded:
                continue
            values = [sample.get(key) for sample in samples]
            values = [float(value) for value in values
                      if isinstance(value, (float, int, np.number)) and np.isfinite(value)]
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
                summary[f"{key}_n"] = len(values)
        summaries.append(summary)
    return summaries