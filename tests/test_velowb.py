"""VeloWB physics, control, learning and experiment regression tests."""

import unittest
import tempfile
import gc
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from tron2_mjlab.robot import VARIANTS, get_robot_cfg
from tron2_mjlab.play import ServoPlayPolicy, ServoVelocityViewer
from tron2_mjlab.velowb.control import (
    DelayBuffer, PoseRegulator, damped_least_squares, kinematic_preview,
    limit_joint_velocity, pose_error, waveform,
)
from tron2_mjlab.velowb.kinematics import ArmKinematics
from tron2_mjlab.velowb.env import ServoConfig, make_env_cfg, servo_observation
from tron2_mjlab.velowb.learning import ServoActor, runner_config
from tron2_mjlab.velowb.metrics import evaluate_trace, frequency_response, step_response, summarize
from tron2_mjlab.velowb.experiment import DEFAULT_CONFIG, load_settings, read_json, train_run


class KinematicsTests(unittest.TestCase):
    def test_wbc_model_profile(self):
        cfg = get_robot_cfg("WFYG", active_arm=True, control_profile="wbc")
        model = Entity(cfg).spec.compile()
        self.assertEqual((model.nq, model.nv, model.nu), (21, 20, 14))
        self.assertEqual(cfg.init_state.joint_pos, {".*": 0.0})
        self.assertEqual(cfg.init_state.pos, (0.0, 0.0, 0.85))
        for name in ("arm1_Joint", "arm6_Joint"):
            self.assertEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name), -1)
        for name, stiffness, damping, effort in (
            ("proximal_pitch_L_Joint", 139.41, 17.75, 130),
            ("proximal_yaw_L_Joint", 182.14, 23.19, 60),
            ("arm2_Joint", 50, 2, 100), ("arm5_Joint", 30, 3, 100),
        ):
            joint_id = model.joint(name).id
            actuator_id = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)[0]
            self.assertAlmostEqual(model.actuator_gainprm[actuator_id, 0], stiffness)
            self.assertAlmostEqual(model.actuator_biasprm[actuator_id, 2], -damping)
            self.assertAlmostEqual(model.actuator_forcerange[actuator_id, 1], effort)
        chain = ArmKinematics(model)
        self.assertEqual(chain.joint_names, [f"arm{number}_Joint" for number in range(2, 6)])
        self.assertTrue(torch.isfinite(chain.forward(torch.zeros((1, 4))).jacobian).all())
        standard = Entity(get_robot_cfg("WFYG", active_arm=True)).spec.compile()
        self.assertEqual(standard.nu, 15)
        with self.assertRaises(ValueError):
            get_robot_cfg("SFYG", active_arm=True, control_profile="wbc")

    def test_estimator_against_mujoco(self):
        rng = np.random.default_rng(42)
        for variant in VARIANTS:
            for locked in (True, False):
                model = Entity(get_robot_cfg(
                    variant, active_arm=True, lock_arm_yaw=locked,
                )).spec.compile()
                chain = ArmKinematics(model, dtype=torch.float64)
                data = mujoco.MjData(model)
                for mode in ("base", "arm", "combined"):
                    with self.subTest(variant=variant, locked=locked, mode=mode):
                        for _ in range(8):
                            data.qpos[:] = model.qpos0
                            data.qpos[:3] = rng.normal(size=3)
                            quaternion = rng.normal(size=4)
                            data.qpos[3:7] = quaternion / np.linalg.norm(quaternion)
                            limits = model.jnt_range[chain.joint_ids]
                            data.qpos[chain.qpos_ids] = rng.uniform(limits[:, 0], limits[:, 1])
                            data.qvel[:] = 0
                            if mode != "arm":
                                data.qvel[:6] = rng.normal(size=6)
                            if mode != "base":
                                data.qvel[chain.dof_ids] = rng.normal(size=len(chain.dof_ids))
                            mujoco.mj_forward(model, data)
                            base_rotation = data.xmat[chain.base_id].reshape(3, 3)
                            tensor = lambda value: torch.tensor(value, dtype=torch.float64)[None]
                            estimate = chain.estimate(
                                tensor(data.qpos[chain.qpos_ids]),
                                tensor(data.qvel[chain.dof_ids]), tensor(data.qpos[:3]),
                                tensor(data.qpos[3:7]), tensor(data.qvel[:3]),
                                tensor(base_rotation @ data.qvel[3:6]),
                            )
                            truth = np.empty(6)
                            site_id = model.site("ee").id
                            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE,
                                                    site_id, truth, 0)
                            np.testing.assert_allclose(estimate.twist[0].numpy(),
                                                       np.r_[truth[3:], truth[:3]], atol=1e-9)
                            np.testing.assert_allclose(estimate.position[0].numpy(),
                                                       data.site_xpos[site_id], atol=1e-9)
                            np.testing.assert_allclose(estimate.rotation[0].numpy(),
                                                       data.site_xmat[site_id].reshape(3, 3), atol=1e-9)
                            masses = model.body_mass[chain.arm_body_ids]
                            com = np.average(data.xipos[chain.arm_body_ids], axis=0, weights=masses)
                            np.testing.assert_allclose(estimate.arm_com[0].numpy(), com, atol=1e-9)


class ControlTests(unittest.TestCase):
    def test_ik_and_preview(self):
        model = Entity(get_robot_cfg("WFYG", active_arm=True)).spec.compile()
        chain = ArmKinematics(model)
        position = chain.limits.mean(dim=-1)[None].repeat(2, 1)
        jacobian = chain.forward(position).jacobian
        zero = torch.zeros((2, 6))
        self.assertFalse(damped_least_squares(jacobian, zero).any())
        self.assertTrue(torch.isfinite(damped_least_squares(jacobian * 0, zero + 1)).all())
        velocity = damped_least_squares(jacobian, zero + 100)
        self.assertLessEqual(velocity.abs().max().item(), 1.5)
        bounded = limit_joint_velocity(chain.limits[:, 1][None], velocity.abs(), chain.limits, 0.1)
        self.assertFalse(bounded.any())
        for mode, per_point in (("joint", 5), ("joint_ee", 20), ("full", 53)):
            preview = kinematic_preview(chain, position, zero, mode)
            self.assertEqual(preview.shape, (2, 4 * per_point))
            for point in preview.reshape(2, 4, per_point).unbind(dim=1):
                torch.testing.assert_close(point[:, :5], position)
        with self.assertRaises(ValueError):
            kinematic_preview(chain, position, zero, horizons=(0.2, 0.1))

    def test_world_rotation_error_and_outer_limits(self):
        from scipy.spatial.transform import Rotation
        actual = torch.tensor(Rotation.from_euler("y", 0.7).as_matrix(), dtype=torch.float32)[None]
        delta = torch.tensor(Rotation.from_euler("z", 0.4).as_matrix(), dtype=torch.float32)[None]
        position = torch.zeros((1, 3))
        error = pose_error(position, delta @ actual, position, actual)
        torch.testing.assert_close(error, torch.tensor([[0., 0., 0., 0., 0., 0.4]]))
        regulator = PoseRegulator(1)
        previous = regulator.command.clone()
        for _ in range(100):
            command = regulator.step(position + 10, delta @ actual, position, actual,
                                     torch.zeros((1, 6)))
            self.assertTrue((command.abs() <= regulator.limits).all())
            self.assertTrue(((command - previous).abs() <= regulator.slew * 0.02 + 1e-7).all())
            previous = command
        regulator.reset()
        self.assertFalse(regulator.command.any())

    def test_profiles_and_delay_reset(self):
        time = torch.linspace(0, 2, 101)[:, None]
        target = torch.tensor([[0.15, 0.1, 0.1, 0.25, 0.25, 0.3]])
        for profile in ("zero", "constant", "step", "ramp", "trapezoid", "sine",
                        "bandlimited", "minimum_jerk"):
            signal = waveform(profile, time, 2.0, target * 0, target)
            self.assertEqual(signal.shape, (101, 6))
            self.assertTrue((signal.abs() <= target + 1e-6).all())
        buffer = DelayBuffer(2, torch.zeros(3, 2))
        delays = torch.tensor([0, 1, 2])
        for value in range(1, 4):
            result = buffer.push(torch.full((3, 2), float(value)), delays)
        torch.testing.assert_close(result[:, 0], torch.tensor([3., 2., 1.]))
        buffer.reset(torch.tensor([1]), torch.full((3, 2), 10.0))
        result = buffer.push(torch.full((3, 2), 4.0), delays)
        torch.testing.assert_close(result[:, 0], torch.tensor([4., 10., 2.]))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class ServoEnvironmentTests(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def test_wbc_randomized_ppo(self):
        for preset in ("wbc_plain", "wbc_cts", "wbc_height", "wbc_two_stage"):
            with self.subTest(preset=preset), tempfile.TemporaryDirectory() as directory:
                selected = "wbc_cts" if preset == "wbc_two_stage" else preset
                overrides = {"num_envs": 4}
                if preset == "wbc_two_stage":
                    overrides["baseline"] = "B3"
                settings, _ = load_settings(DEFAULT_CONFIG, selected, overrides)
                env = ManagerBasedRlEnv(make_env_cfg(settings), device="cuda:0")
                try:
                    wrapped = RslRlVecEnvWrapper(env)
                    servo = env.command_manager.get_term("ee_twist")
                    if preset == "wbc_cts":
                        self.assertEqual(servo.teacher_mask.sum().item(), 3)
                    base_id = servo.robot.indexing.root_body_id
                    nominal = env.sim.get_default_field("body_mass")[base_id]
                    self.assertTrue(((env.sim.model.body_mass[:, base_id] - nominal) >= -1).all())
                    self.assertTrue(((env.sim.model.body_mass[:, base_id] - nominal) <= 3).all())
                    config = runner_config(settings, iterations=1, steps_per_env=4)
                    config["algorithm"].update(num_learning_epochs=1, num_mini_batches=1, alignment_epochs=1)
                    runner = OnPolicyRunner(wrapped, config, directory, "cuda:0")
                    runner.learn(1)
                    if preset == "wbc_two_stage":
                        runner.alg.set_distillation_phase()
                        servo.teacher_mask.zero_()
                        actor_mean = runner.alg.actor.obs_normalizer.mean.clone()
                        teacher_mean = runner.alg.actor.teacher_normalizer.mean.clone()
                        runner.learn(1)
                        torch.testing.assert_close(actor_mean, runner.alg.actor.obs_normalizer.mean)
                        torch.testing.assert_close(teacher_mean, runner.alg.actor.teacher_normalizer.mean)
                    actor = runner.alg.actor.eval()
                    obs = wrapped.get_observations()
                    exported = torch.jit.script(actor.as_jit())
                    history = obs["history"] if "history" in obs else obs["actor"][:, :0]
                    torch.testing.assert_close(actor(obs), exported(obs["actor"], history))
                    self.assertTrue(all(torch.isfinite(value).all() for value in actor.parameters()))
                    observations, _ = env.reset()
                    self.assertTrue(all(torch.isfinite(value).all() for value in observations.values()))
                finally:
                    env.close()

    def test_wbc_heading_and_height(self):
        settings = ServoConfig(num_envs=2, control_profile="wbc", command_frame="heading",
                               height_tracking=True)
        env = ManagerBasedRlEnv(make_env_cfg(settings), device="cuda:0")
        try:
            env.reset()
            servo = env.command_manager.get_term("ee_twist")
            initial = servo.estimate.position[:, 2].clone()
            torch.testing.assert_close(servo.height_target, initial)
            command = torch.zeros((2, 6), device=env.device)
            command[:, 2] = 0.1
            servo.set_twist(command)
            torch.testing.assert_close(servo.height_target, initial + 0.002)
            once = servo.height_target.clone()
            servo.set_twist(command)
            torch.testing.assert_close(servo.height_target, once)
            for _ in range(150):
                env.common_step_counter += 1
                servo.sensor_step = env.common_step_counter
                servo.set_twist(command)
            self.assertTrue(((servo.height_target - initial).abs() <= 0.150001).all())
            servo.sense()
            identity = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], device=env.device)
            servo.base_rotation[:] = identity
            heading_x = torch.tensor([[1., 0., 0., 1., 0., 0.]], device=env.device).repeat(2, 1)
            world_y = torch.tensor([[0., 1., 0., 0., 1., 0.]], device=env.device).repeat(2, 1)
            torch.testing.assert_close(servo.transform_twist(heading_x, to_world=True), world_y,
                                       atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(servo.transform_twist(world_y), heading_x, atol=1e-6, rtol=1e-6)
            env.reset(env_ids=torch.tensor([1], device=env.device))
            torch.testing.assert_close(servo.height_target[1], servo.estimate.position[1, 2])
            self.assertEqual(servo_observation(env).shape[1], 96)
        finally:
            env.close()

    def test_wbc_position_actions_and_delay(self):
        settings = ServoConfig(num_envs=2, control_profile="wbc", actuator_delay_steps=4)
        env = ManagerBasedRlEnv(make_env_cfg(settings), device="cuda:0")
        try:
            observations, _ = env.reset()
            action = env.action_manager.get_term("whole_body")
            servo = env.command_manager.get_term("ee_twist")
            self.assertEqual(env.action_manager.total_action_dim, 14)
            self.assertAlmostEqual(env.physics_dt, 0.005)
            self.assertAlmostEqual(env.step_dt, 0.02)
            self.assertIn("arm_first", env.cfg.rewards)
            self.assertIn("wheel_slip", env.cfg.rewards)
            self.assertAlmostEqual(servo.wheel_radius, 0.1)
            desired = torch.tensor([[0.1] * 8 + [0.5, -0.5, 0.2, -0.2, 0.3, -0.3]],
                                   device=env.device).repeat(2, 1)
            action.physics_delay[:] = torch.tensor([0, 4], device=env.device)
            action.process_actions(desired)
            action.apply_actions()
            self.assertFalse(action.applied[1].any())
            for _ in range(4):
                action.apply_actions()
            expected = torch.tensor([[0.5, -0.5, 0.1, -0.1]], device=env.device).repeat(2, 1)
            torch.testing.assert_close(action.arm_target, expected)
            torch.testing.assert_close(servo.robot.data.joint_vel_target[:, servo.wheel_ids],
                                       torch.tensor([[2.1, -2.1]], device=env.device).repeat(2, 1))
            action.process_actions(desired)
            action.apply_actions()
            torch.testing.assert_close(action.arm_target, expected)
            wide = desired.clone()
            wide[:, 8] = 2.0
            wide[:, -2:] = 100.0
            action.physics_delay[:] = 0
            action.process_actions(wide)
            action.apply_actions()
            torch.testing.assert_close(action.arm_target[:, 0], torch.full((2,), 2.0, device=env.device))
            self.assertTrue((servo.robot.data.joint_vel_target[:, servo.wheel_ids].abs() <= 41.891).all())
            for _ in range(20):
                observations, rewards, _, _, _ = env.step(desired * 0)
                self.assertTrue(torch.isfinite(rewards).all())
                self.assertTrue(all(torch.isfinite(value).all() for value in observations.values()))
                for config in env.cfg.rewards.values():
                    self.assertTrue(torch.isfinite(config.func(env, **config.params)).all())
            env.reset()
            self.assertFalse(action.physics_buffer.buffer.any())
            self.assertFalse(action.arm_target.any())
        finally:
            env.close()

    def test_manual_servo_playback(self):
        for axes in ("linear_yaw", "full", "wbc"):
            with self.subTest(axes=axes):
                env = ManagerBasedRlEnv(make_env_cfg(
                    ServoConfig(num_envs=1, command_axes="full" if axes == "wbc" else axes,
                                control_profile="wbc" if axes == "wbc" else "standard",
                                command_frame="heading" if axes == "wbc" else "world",
                                height_tracking=axes == "wbc", history_seconds=0.1 if axes == "wbc" else 0.4,
                                encoder="none" if axes == "linear_yaw" else "history"), evaluation=True,
                ), device="cuda:0")
                viewer = None
                try:
                    wrapped = RslRlVecEnvWrapper(env)
                    captured = []

                    def actor(observations):
                        captured.append(observations)
                        return torch.zeros((1, wrapped.num_actions), device=env.device)

                    policy = ServoPlayPolicy(env, actor)
                    viewer = ServoVelocityViewer(wrapped, policy)
                    viewer.setup()
                    self.assertEqual(viewer.sliders[3].disabled, axes == "linear_yaw")
                    self.assertEqual(viewer.sliders[4].disabled, axes == "linear_yaw")
                    self.assertFalse(viewer.sliders[1].disabled)
                    self.assertFalse(viewer.command.command.any())
                    for slider in viewer.sliders:
                        slider.value = 0.05
                    viewer.sync_viewer_to_env()
                    expected = torch.tensor([[0.05] * 6], device=env.device)
                    if axes == "linear_yaw":
                        expected[:, 3:5] = 0
                    command_start = -19 if axes == "wbc" else -18
                    torch.testing.assert_close(viewer.command.command, expected)
                    torch.testing.assert_close(
                        wrapped.get_observations()["actor"][:, command_start:command_start + 6],
                        expected / viewer.command.limits,
                    )
                    viewer.step_simulation()
                    if policy.history is not None:
                        history = captured[-1]["history"].reshape(1, policy.history.max_length, -1)
                        torch.testing.assert_close(history[:, -1, command_start:command_start + 6],
                                                   expected / viewer.command.limits)
                    viewer.reset_environment()
                    viewer.sync_viewer_to_env()
                    torch.testing.assert_close(viewer.command.command, expected)
                    viewer.manual.value = False
                    viewer.sync_viewer_to_env()
                    self.assertFalse(viewer.command.external)
                    self.assertTrue(all(slider.disabled for slider in viewer.sliders))
                    viewer._stop(None)
                    viewer.sync_viewer_to_env()
                    self.assertTrue(viewer.manual.value)
                    self.assertTrue(viewer.command.external)
                    self.assertFalse(viewer.command.command.any())
                    self.assertFalse(wrapped.get_observations()["actor"][:, command_start:command_start + 6].any())
                    viewer.reset_environment()
                    viewer.sync_viewer_to_env()
                    viewer.step_simulation()
                    if policy.history is not None:
                        history = captured[-1]["history"].reshape(1, policy.history.max_length, -1)
                        torch.testing.assert_close(history, history[:, -1:].expand_as(history))
                finally:
                    if viewer is not None:
                        viewer.close()
                    env.close()

    def test_learning_updates(self):
        for baseline in ("B0", "B1", "B2", "B3", "B4", "B6", "B7"):
            with self.subTest(baseline=baseline), tempfile.TemporaryDirectory() as directory:
                settings = ServoConfig(num_envs=4, baseline=baseline)
                env = ManagerBasedRlEnv(make_env_cfg(settings), device="cuda:0")
                try:
                    wrapped = RslRlVecEnvWrapper(env)
                    config = runner_config(settings, iterations=1, steps_per_env=4)
                    config["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
                    runner = OnPolicyRunner(wrapped, config, directory, "cuda:0")
                    runner.learn(1)
                    self.assertTrue(all(torch.isfinite(value).all() for value in runner.alg.actor.parameters()))
                    if baseline == "B3":
                        runner.alg.set_distillation_phase()
                        env.command_manager.get_term("ee_twist").teacher_mask.zero_()
                        teacher_before = [value.clone() for value in runner.alg.actor.teacher_encoder.parameters()]
                        policy_before = [value.clone() for value in runner.alg.actor.mlp.parameters()]
                        runner.learn(1)
                        for before, after in zip(teacher_before + policy_before,
                                                  list(runner.alg.actor.teacher_encoder.parameters()) +
                                                  list(runner.alg.actor.mlp.parameters())):
                            torch.testing.assert_close(before, after)
                finally:
                    env.close()

    def test_randomized_history_environment(self):
        settings = ServoConfig(num_envs=4, baseline="B7", preview="joint_ee", sensor_noise=1.0,
                               sensor_delay_steps=2, timestamp_skew_steps=1, odometry_drift=0.01,
                               command_delay_steps=2, action_delay_steps=2, payload_mass=(0.5, 0.5),
                               payload_com=0.02, friction=(0.4, 0.4), motor_strength=(0.8, 0.8))
        env = ManagerBasedRlEnv(make_env_cfg(settings, evaluation=True), device="cuda:0")
        try:
            observations, _ = env.reset()
            servo = env.command_manager.get_term("ee_twist")
            self.assertFalse(servo.teacher_mask.any())
            payload_id = env.sim.mj_model.body("robot/payload").id
            torch.testing.assert_close(env.sim.model.body_mass[:, payload_id],
                                       torch.full((4,), 0.5001, device=env.device))
            self.assertEqual(observations["history"].shape[1], servo_observation(env).shape[1] * 20)
            for _ in range(8):
                observations, rewards, _, _, _ = env.step(torch.zeros((4, 15), device=env.device))
                self.assertTrue(torch.isfinite(rewards).all())
                self.assertTrue(all(torch.isfinite(value).all() for value in observations.values()))
            env.reset(env_ids=torch.tensor([1], device=env.device))
            torch.testing.assert_close(env.sim.model.body_mass[:, payload_id],
                                       torch.full((4,), 0.5001, device=env.device))
            history = env.obs_buf["history"][1].reshape(20, -1)
            torch.testing.assert_close(history, history[-1:].expand_as(history))
            servo.command_delay[:] = 2
            for _ in range(3):
                servo.set_twist(torch.ones((4, 6), device=env.device))
                queued = servo.command.clone()
                servo.set_twist(torch.ones((4, 6), device=env.device))
                torch.testing.assert_close(servo.command, queued)
                env.step(torch.zeros((4, 15), device=env.device))
            torch.testing.assert_close(servo.command, servo.limits[None].expand(4, -1))
        finally:
            env.close()


class LearningTests(unittest.TestCase):
    def test_wbc_learning_normalization_and_export(self):
        observations = TensorDict({
            "actor": torch.randn(8, 20) + 10, "history": torch.randn(8, 100) + 10,
            "privileged": torch.randn(8, 11) + 50,
            "role": torch.tensor([[1.]] * 6 + [[0.]] * 2),
        }, batch_size=[8])
        actor = ServoActor(observations, {"actor": ["actor"]}, "actor", 14,
                           encoder_mode="cts", hidden_dims=(32,), encoder_hidden=(32,),
                           latent_dim=8, obs_normalization=True, detach_student_ppo=True)
        actor.update_normalization(observations)
        actor(observations).square().mean().backward()
        self.assertTrue(all(value.grad is None for value in actor.student_encoder.parameters()))
        self.assertTrue(any(value.grad is not None for value in actor.teacher_encoder.parameters()))
        actor.zero_grad()
        actor.alignment_loss(observations).backward()
        self.assertTrue(any(value.grad is not None for value in actor.student_encoder.parameters()))
        self.assertTrue(all(value.grad is None for value in actor.teacher_encoder.parameters()))
        actor.eval()
        expected = actor(observations.select("actor", "history"))
        exported = torch.jit.script(actor.as_jit())
        torch.testing.assert_close(expected, exported(observations["actor"], observations["history"]))
        config = runner_config(ServoConfig(control_profile="wbc", baseline="B4"))
        self.assertTrue(config["actor"]["obs_normalization"])
        self.assertEqual(config["actor"]["latent_dim"], 64)
        self.assertEqual(config["algorithm"]["clip_rewards"], 5.0)

    def test_learning_information_boundary(self):
        observations = TensorDict({
            "actor": torch.randn(8, 20), "history": torch.randn(8, 100),
            "privileged": torch.randn(8, 11),
            "role": torch.tensor([[1.]] * 4 + [[0.]] * 4),
        }, batch_size=[8])
        actor = ServoActor(observations, {"actor": ["actor"]}, "actor", 15,
                           encoder_mode="cts", hidden_dims=(32,), encoder_hidden=(32,),
                           latent_dim=8, stochastic=True, noise_std_type="log")
        actor.alignment_loss(observations).backward()
        self.assertTrue(any(value.grad is not None for value in actor.student_encoder.parameters()))
        self.assertTrue(all(value.grad is None for value in actor.teacher_encoder.parameters()))
        actor.zero_grad()
        actor(observations).square().mean().backward()
        self.assertTrue(any(value.grad is not None and bool(value.grad.any())
                            for value in actor.teacher_encoder.parameters()))
        actor.eval()
        deployable = observations.select("actor", "history")
        expected = actor(deployable)
        observations["privileged"] *= 1e6
        torch.testing.assert_close(expected, actor(observations))
        exported = torch.jit.script(actor.as_jit())
        torch.testing.assert_close(expected, exported(deployable["actor"], deployable["history"]))


class MetricsTests(unittest.TestCase):
    def test_wbc_metric_frames(self):
        time = np.linspace(0, 1, 51)
        world_command = np.zeros((len(time), 6))
        world_command[:, 1] = 0.5
        world_twist = world_command * 0.8
        heading_command = np.zeros_like(world_command)
        heading_command[:, 0] = 0.5
        position = np.zeros((len(time), 3))
        position[:, 2] = 1.0
        metrics = evaluate_trace({
            "time": time, "command": world_command, "twist": world_twist,
            "tracking_command": heading_command, "tracking_twist": heading_command * 0.8,
            "estimate": world_twist, "position": position,
            "height_target": np.full((len(time), 1), 1.1),
        }, command_limits=(0.5, 0.2, 0.5, 0.5, 0.5, 0.5))
        self.assertAlmostEqual(metrics["weighted_twist_rmse"], 0.2 / np.sqrt(6))
        self.assertEqual(metrics["estimator_rmse"], 0.0)
        self.assertAlmostEqual(metrics["height_error_rmse_m"], 0.1)

    def test_metrics_step_and_frequency(self):
        time = np.linspace(0, 5, 501)
        command = np.where(time < 1, 0.0, -1.0)
        measured = command.copy()
        measured[120] = -1.2
        metrics = step_response(time, command, measured)
        self.assertAlmostEqual(metrics["overshoot_pct"], 20.0)
        self.assertAlmostEqual(metrics["settling_time_s"], 0.21)
        self.assertEqual(metrics["rise_time_s"], 0.0)
        untracked = step_response(time, command, np.zeros_like(time))
        self.assertIsNone(untracked["rise_time_s"])
        self.assertIsNone(untracked["settling_time_s"])
        frequency = frequency_response(time, np.sin(2 * np.pi * time),
                                       0.8 * np.sin(2 * np.pi * time - 0.3), 1.0)
        self.assertAlmostEqual(frequency["gain"], 0.8)
        self.assertAlmostEqual(frequency["phase_lag_deg"], np.degrees(0.3))

    def test_metrics_failures_and_undefined_ratio(self):
        time = np.linspace(0, 2, 101)
        zero = np.zeros((len(time), 6))
        trace = {"time": time, "command": zero, "twist": zero,
                 "base_induced": zero, "fallen": np.array([False] * 100 + [True])}
        metrics = evaluate_trace(trace)
        self.assertEqual(metrics["fallen"], 1)
        self.assertIsNone(metrics["disturbance_transmission_linear"])
        rows = [{"baseline": "B1", "scenario": "zero", "score": 1., "settling": None},
                {"baseline": "B1", "scenario": "zero", "score": 3., "settling": 2.}]
        summary = summarize(rows)[0]
        self.assertEqual(summary["score_mean"], 2)
        self.assertEqual(summary["settling_n"], 1)
        self.assertEqual(summary["n"], 2)


class ExperimentTests(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_wbc_experiment_end_to_end(self):
        from tron2_mjlab.velowb.evaluation import evaluate_run
        from tron2_mjlab.velowb.study import g2_gate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings, _ = load_settings(DEFAULT_CONFIG, "wbc_height", {"num_envs": 4})
            trained = train_run(settings, root / "train", iterations=1, steps_per_env=4,
                                learning_epochs=1, mini_batches=1, smoke=True, label="wbc_height")
            interface = read_json(root / "train" / "interface.json")
            self.assertEqual(interface["control_profile"], "wbc")
            self.assertEqual(interface["twist_frame"], "heading")
            self.assertTrue(interface["policy_uses_pose_error"])
            self.assertEqual(interface["action_vector_order"][-2:], ["wheel_L_Joint", "wheel_R_Joint"])
            self.assertEqual(read_json(root / "train" / "estimator" / "g1.json")["control_profile"], "wbc")
            checkpoint = root / "train" / trained["checkpoint"]
            for scenario in ("zero", "ramp"):
                result = evaluate_run(settings, root / scenario, checkpoint=checkpoint,
                                      scenario=scenario, episodes=2, seconds=0.12, axes=(0,),
                                      make_plots=False)
                self.assertEqual(result["tracking_frame"], "heading")
            gate = g2_gate(root / "zero", root / "ramp", root / "g2.json")
            self.assertFalse(gate["criteria"]["plain_servo"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_experiment_end_to_end(self):
        from tron2_mjlab.velowb.evaluation import evaluate_run
        from tron2_mjlab.velowb.study import aggregate_runs, g2_gate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = ServoConfig(num_envs=4)
            trained = train_run(settings, root / "train", iterations=1, steps_per_env=4,
                                learning_epochs=1, mini_batches=1, smoke=True)
            checkpoint = root / "train" / trained["checkpoint"]
            self.assertTrue((root / "train" / "interface.json").is_file())
            for scenario in ("zero", "step", "reaching"):
                evaluated = evaluate_run(settings, root / scenario, checkpoint=checkpoint,
                                         scenario=scenario, episodes=2, seconds=0.12, axes=(0,),
                                         make_plots=False)
                self.assertEqual(evaluated["status"], "completed")
                self.assertTrue(evaluated["smoke_only"])
                self.assertTrue((root / scenario / "result_manifest.md").is_file())
            report = g2_gate(root / "zero", root / "step", root / "g2.json")
            self.assertFalse(report["passed"])
            self.assertFalse(report["criteria"]["not_smoke"])
            summary = aggregate_runs([root / "zero", root / "step"], root / "aggregate")
            self.assertTrue(all(row["n"] == 1 for row in summary))
            self.assertTrue(all(row.get("weighted_twist_rmse_std") is None for row in summary))
            legacy = torch.load(checkpoint, map_location="cpu", weights_only=False)
            for key in ("control_profile", "actuator_delay_steps", "command_frame", "height_tracking",
                        "base_mass_offset", "base_com_range", "leg_mass_scale"):
                legacy["infos"]["servo_config"].pop(key)
            torch.save(legacy, root / "legacy.pt")
            resumed = train_run(settings, root / "resumed", iterations=2, steps_per_env=4,
                                learning_epochs=1, mini_batches=1, smoke=True, resume=root / "legacy.pt")
            self.assertEqual(resumed["completed_iterations"], 2)

    def test_experiment_matrix(self):
        from tron2_mjlab.velowb.study import plan_study
        with tempfile.TemporaryDirectory() as directory:
            matrix = plan_study(Path(directory) / "baseline", suite="baselines")
            self.assertEqual(len(matrix["runs"]), 24)
            learned = [run for run in matrix["runs"] if run["kind"] == "train"]
            self.assertEqual(len(learned), 21)
            self.assertEqual(len({run["transitions"] for run in learned}), 1)
            preview = plan_study(Path(directory) / "preview", suite="preview")
            self.assertEqual(len(preview["runs"]), 18)
            wbc = plan_study(Path(directory) / "wbc", suite="wbc")
            self.assertEqual(len(wbc["runs"]), 9)
            self.assertTrue(all(run["settings"]["control_profile"] == "wbc" for run in wbc["runs"]))

    def test_experiment_trajectories(self):
        from tron2_mjlab.velowb.evaluation import make_cases, trajectory_samples
        self.assertEqual(len(make_cases("step")), 12)
        self.assertEqual(len(make_cases("frequency")), 30)
        time = np.linspace(0, 10, 1001)
        for kind in ("line", "circle", "figure_eight", "spline", "orientation"):
            position, rotation, twist = trajectory_samples(kind, time)
            self.assertEqual(rotation.shape, (len(time), 3, 3))
            np.testing.assert_allclose(np.gradient(position, time, axis=0)[2:-2],
                                       twist[2:-2, :3], atol=2e-5)

    def test_experiment_configuration_and_gate(self):
        for preset in ("plain_v0", "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7",
                       "K0_no_prior", "K1_current_ik", "K2_joint_rollout", "K3_joint_ee_rollout",
                       "K4_com_jacobian", "K5_preview_cts"):
            settings, budget = load_settings(DEFAULT_CONFIG, preset)
            settings.validate()
            self.assertEqual(budget["iterations"], 10_000)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "G2"):
                train_run(ServoConfig(baseline="B4"), Path(directory) / "blocked")
            self.assertFalse((Path(directory) / "blocked").exists())
        with self.assertRaises(ValueError):
            load_settings(DEFAULT_CONFIG, "B1", {"misspelled": 1})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_servo_step(self):
        settings = ServoConfig(num_envs=4, command_axes="full")
        env = ManagerBasedRlEnv(make_env_cfg(settings), device="cuda:0")
        try:
            observations, _ = env.reset()
            servo = env.command_manager.get_term("ee_twist")
            self.assertEqual(env.action_manager.total_action_dim, 15)
            self.assertFalse(env.scene["robot"].data.joint_pos[:, servo.arm_ids].any())
            self.assertFalse(env.scene["robot"].data.joint_vel[:, servo.arm_ids].any())
            self.assertFalse(env.action_manager.get_term("whole_body").arm_target.any())
            before = servo_observation(env).clone()
            servo.target_position += 10
            torch.testing.assert_close(before, servo_observation(env))
            for _ in range(30):
                observations, rewards, _, _, _ = env.step(torch.zeros((4, 15), device=env.device))
                self.assertTrue(torch.isfinite(rewards).all())
                self.assertTrue(all(torch.isfinite(value).all() for value in observations.values()))
                torch.testing.assert_close(servo.estimate.twist,
                                           env.scene["robot"].data.site_vel_w[:, servo.ee_id],
                                           atol=2e-4, rtol=2e-4)
            servo.set_twist(torch.zeros((4, 6), device=env.device))
            self.assertFalse(servo.prior().any())
        finally:
            env.close()