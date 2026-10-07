"""Run with: uv run python -m unittest discover -s tests -v."""

import tempfile
import unittest
from dataclasses import asdict
from itertools import product
from math import pi
from unittest.mock import patch

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

from tron2_mjlab.env_cfgs import tron2_flat_env_cfg
from tron2_mjlab.gait import GaitVelocityCommand, alternating_contacts, gait_phase
from tron2_mjlab.play import ManualVelocityCommandCfg, VelocityViewer, main as play_main
from tron2_mjlab.robot import ARM_JOINTS, ASSET_ROOT, LEG_JOINTS, VARIANTS, get_robot_cfg, get_spec


class Tron2Tests(unittest.TestCase):
    def test_models_and_configs(self):
        self.assertEqual(VARIANTS, ("SFYG",))
        self.assertFalse((ASSET_ROOT / "WFYG_TRON2A").exists())
        self.assertEqual(
            [task for task in list_tasks() if "TRON2" in task],
            [f"Mjlab-Velocity-Flat-TRON2A-{variant}{suffix}"
             for variant, suffix in product(
                 VARIANTS, ("-Baseline-CTS", "-LBC-CTS"),
             )],
        )
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = Entity(get_robot_cfg(variant)).spec.compile()
                self.assertEqual((model.nq, model.nv, model.nu), (23, 22, 16))
                for side, angle in (("L", -pi), ("R", pi)):
                    joint = model.joint(f"proximal_yaw_{side}_Joint")
                    self.assertAlmostEqual(model.key_qpos[0, joint.qposadr[0]], angle)
                    low, high = joint.range
                    margin = (high - low) * 0.05
                    self.assertGreater(angle, low + margin)
                    self.assertLess(angle, high - margin)
                for name in ARM_JOINTS:
                    self.assertGreaterEqual(model.joint(name).id, 0)
                for joint_id in range(model.njnt):
                    self.assertFalse(
                        (model.joint(joint_id).name or "").startswith("gripper")
                    )
                self.assertFalse((model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).any())
                cfg = tron2_flat_env_cfg(variant)
                self.assertEqual(cfg.scene.terrain.terrain_type, "plane")
                self.assertIsNone(cfg.scene.terrain.terrain_generator)
                self.assertFalse(cfg.curriculum)
                for group in cfg.observations.values():
                    self.assertNotIn("height_scan", group.terms)
                play = tron2_flat_env_cfg(variant, play=True)
                self.assertFalse(play.observations["actor"].enable_corruption)
                self.assertTrue(cfg.observations["actor"].enable_corruption)
                self.assertEqual(set(cfg.actions), {"joint_pos"})
                self.assertEqual(cfg.actions["joint_pos"].actuator_names, LEG_JOINTS)
                for reward in ("pose", "dof_pos_limits"):
                    self.assertEqual(
                        cfg.rewards[reward].params["asset_cfg"].joint_names, LEG_JOINTS,
                    )
                self.assertIn("foot_slip", cfg.rewards)
                ranges = cfg.commands["twist"].ranges
                self.assertEqual(ranges.lin_vel_x, (-1.0, 2.0))
                self.assertEqual(ranges.lin_vel_y, (-0.6, 0.6))
                self.assertEqual(ranges.ang_vel_z, (-1.0, 1.0))
                gait = cfg.rewards["gait_contact"]
                self.assertIs(gait.func, alternating_contacts)
                self.assertEqual(gait.weight, 1.0)
                self.assertEqual(cfg.commands["twist"].gait.slow_period, 0.9)
                self.assertEqual(cfg.commands["twist"].gait.fast_period, 0.6)
                self.assertEqual(gait.params["stance_ratio"], 0.6)
                for name in ("actor", "critic"):
                    phase = cfg.observations[name].terms["gait_phase"]
                    self.assertIs(phase.func, gait_phase)
                    self.assertIsNone(phase.noise)
                    for name, value in phase.params.items():
                        self.assertEqual(gait.params[name], value)
        for variant in ("WFYG", "SFYG_TRON2B"):
            with self.subTest(unsupported_variant=variant):
                with self.assertRaises(ValueError):
                    tron2_flat_env_cfg(variant)
                with self.assertRaises(ValueError):
                    get_spec(variant)
        for suffix in ("LBC", "Baseline"):
            with self.assertRaises(KeyError):
                load_env_cfg(f"Mjlab-Velocity-Flat-TRON2A-SFYG-{suffix}")

    def test_baseline_configs(self):
        task = "Mjlab-Velocity-Flat-TRON2A-SFYG"
        for play in (False, True):
            with self.subTest(play=play):
                full = load_env_cfg(task + "-LBC-CTS", play=play)
                baseline = load_env_cfg(task + "-Baseline-CTS", play=play)
                for group in ("actor", "critic"):
                    for term in ("joint_pos", "joint_vel"):
                        self.assertEqual(
                            baseline.observations[group].terms[term].params["asset_cfg"].joint_names,
                            LEG_JOINTS,
                        )
                        self.assertEqual(
                            full.observations[group].terms[term].params["asset_cfg"].joint_names,
                            LEG_JOINTS + ARM_JOINTS,
                        )
                for field in ("rewards", "actions", "events", "commands"):
                    self.assertEqual(
                        {name: asdict(value) for name, value in getattr(full, field).items()},
                        {name: asdict(value) for name, value in getattr(baseline, field).items()},
                    )
                model = Entity(baseline.scene.entities["robot"]).spec.compile()
                self.assertEqual((model.nq, model.nv, model.nu), (23, 22, 16))
                self.assertEqual(baseline.observations["actor"].enable_corruption, not play)
                self.assertFalse(baseline.observations["critic"].enable_corruption)
                for cfg in (full, baseline):
                    ranges = cfg.commands["twist"].ranges
                    self.assertEqual(ranges.lin_vel_x, (-1.0, 2.0))
                    self.assertEqual(ranges.lin_vel_y, (-0.6, 0.6))
                    self.assertEqual(ranges.ang_vel_z, (-1.0, 1.0))
        full_rl = load_rl_cfg(task + "-LBC-CTS")
        baseline_rl = load_rl_cfg(task + "-Baseline-CTS")
        self.assertEqual(full_rl.experiment_name, "tron2a_sfyg_velocity_lbc_cts")
        self.assertEqual(baseline_rl.experiment_name, "tron2a_sfyg_velocity_baseline_cts")
        baseline_rl.experiment_name = full_rl.experiment_name
        self.assertEqual(asdict(full_rl), asdict(baseline_rl))

    def test_play_task_selection(self):
        for baseline in (False, True):
            task = "Mjlab-Velocity-Flat-TRON2A-SFYG" + ("-Baseline-CTS" if baseline else "-LBC-CTS")
            argv = ["tron2-play", "SFYG", "--agent", "zero", "--device", "cpu"]
            if baseline:
                argv.append("--baseline")
            with (
                self.subTest(baseline=baseline),
                patch("sys.argv", argv),
                patch("tron2_mjlab.play.load_env_cfg", wraps=load_env_cfg) as load_env,
                patch("tron2_mjlab.play.load_rl_cfg", wraps=load_rl_cfg) as load_rl,
                patch("tron2_mjlab.play.ManagerBasedRlEnv") as env_cls,
                patch("tron2_mjlab.play.RslRlVecEnvWrapper") as wrapper_cls,
                patch("tron2_mjlab.play.VelocityViewer") as viewer_cls,
            ):
                env_cls.return_value.num_envs = 1
                env_cls.return_value.device = "cpu"
                wrapper_cls.return_value.num_actions = 10
                play_main()
                load_env.assert_called_once_with(task, play=True)
                load_rl.assert_called_once_with(task)
                viewer_cls.return_value.run.assert_called_once_with()
                policy = viewer_cls.call_args.args[1]
                self.assertEqual(policy({}).shape, (1, 10))
                env_cls.return_value.close.assert_called_once_with()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU smoke test")
    def test_manual_command(self):
        for variant, baseline in product(VARIANTS, (False, True)):
            with self.subTest(variant=variant, baseline=baseline):
                cfg = tron2_flat_env_cfg(variant, play=True, observe_arm=not baseline)
                cfg.commands["twist"] = ManualVelocityCommandCfg(**vars(cfg.commands["twist"]))
                env = ManagerBasedRlEnv(cfg, device="cuda:0")
                try:
                    env.reset()
                    robot = env.scene["robot"]
                    yaw_ids, _ = robot.find_joints(
                        ("proximal_yaw_L_Joint", "proximal_yaw_R_Joint"),
                        preserve_order=True,
                    )
                    yaw = torch.tensor([[-pi, pi]], device=env.device)
                    torch.testing.assert_close(robot.data.joint_pos[:, yaw_ids], yaw)
                    torch.testing.assert_close(robot.data.default_joint_pos[:, yaw_ids], yaw)
                    env.action_manager.process_action(torch.zeros((1, 10), device=env.device))
                    env.action_manager.apply_action()
                    torch.testing.assert_close(
                        robot.data.joint_pos_target[:, yaw_ids],
                        yaw - robot.data.encoder_bias[:, yaw_ids],
                    )
                    joint_count = 10 if baseline else 16
                    obs = env.observation_manager.compute_group("actor")
                    torch.testing.assert_close(
                        obs[:, -1, 9:9 + joint_count],
                        torch.zeros((1, joint_count), device=env.device),
                    )
                    command = env.command_manager.get_term("twist")
                    self.assertIsInstance(command, GaitVelocityCommand)
                    clock = command.gait_clock
                    torch.testing.assert_close(clock.phase, torch.zeros_like(clock.phase))
                    target = (2.0, 0.6, -1.0)
                    command.manual_target = target
                    command.time_left[:] = 0.0
                    env.step(torch.zeros((1, 10), device=env.device))
                    torch.testing.assert_close(clock.phase, env.step_dt * clock.frequency)
                    torch.testing.assert_close(command.command[0], torch.tensor(target, device=env.device))
                    phase_cfg = cfg.observations["actor"].terms["gait_phase"]
                    torch.testing.assert_close(
                        env.observation_manager.compute_group("actor")[:, -1, -5:-3],
                        gait_phase(env, **phase_cfg.params),
                    )
                    env.reset()
                    torch.testing.assert_close(clock.phase, torch.zeros_like(clock.phase))
                    torch.testing.assert_close(clock.frequency, torch.full_like(clock.frequency, 1 / 0.9))
                    torch.testing.assert_close(robot.data.joint_pos[:, yaw_ids], yaw)
                    torch.testing.assert_close(command.command[0], torch.tensor(target, device=env.device))
                    command.manual_target = (0.0, 0.0, 0.0)
                    previous_phase = clock.phase.clone()
                    command._update_command()
                    torch.testing.assert_close(clock.phase, previous_phase)
                    self.assertFalse(command.command.any())
                    command.manual_target = None
                    command._resample_command(torch.arange(env.num_envs, device=env.device))
                    self.assertTrue(torch.isfinite(command.command).all())
                    wrapped = RslRlVecEnvWrapper(env)
                    viewer = VelocityViewer(wrapped, lambda obs: torch.zeros((1, 10), device=env.device))
                    viewer.setup()
                    try:
                        for slider, limits in zip(
                            viewer.sliders, ((-1.0, 2.0), (-0.6, 0.6), (-1.0, 1.0)),
                        ):
                            self.assertEqual((slider.min, slider.max), limits)
                        for slider, value in zip(viewer.sliders, target):
                            slider.value = value
                        viewer.sync_viewer_to_env()
                        torch.testing.assert_close(wrapped.get_observations()["actor"][0, -1, -3:],
                                                   torch.tensor(target, device=env.device))
                        self.assertFalse(viewer.sliders[1].disabled)
                        viewer.manual.value = False
                        viewer.sync_viewer_to_env()
                        self.assertIsNone(command.manual_target)
                        viewer._stop(None)
                        viewer.sync_viewer_to_env()
                        self.assertFalse(command.command.any())
                        torch.testing.assert_close(
                            wrapped.get_observations()["actor"][:, -1, -5:-3],
                            torch.zeros((1, 2), device=env.device),
                        )
                    finally:
                        viewer.close()
                finally:
                    env.close()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU smoke test")
    def test_arm_observations_and_external_targets(self):
        cfg = tron2_flat_env_cfg("SFYG", play=True)
        cfg.scene.num_envs = 2
        env = ManagerBasedRlEnv(cfg, device="cuda:0")
        try:
            env.reset()
            robot = env.scene["robot"]
            arm_ids, arm_names = robot.find_joints(ARM_JOINTS, preserve_order=True)
            self.assertEqual(tuple(arm_names), ARM_JOINTS)
            self.assertEqual(env.action_manager.total_action_dim, 10)
            torch.testing.assert_close(
                robot.data.joint_pos_target[:, arm_ids],
                torch.zeros((2, 6), device=env.device),
            )
            position = torch.tensor(
                [[0.1, 0.2, -0.3, 0.4, -0.5, 0.6],
                 [-0.2, 0.3, -0.4, -0.5, 0.6, -0.7]], device=env.device,
            )
            velocity = torch.tensor(
                [[0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
                 [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]], device=env.device,
            )
            robot.write_joint_state_to_sim(position, velocity, joint_ids=arm_ids)
            env.sim.forward()
            for group, width in (("actor", 56), ("critic", 68)):
                obs = env.observation_manager.compute_group(group, update_history=True)
                if group == "actor":
                    self.assertEqual(obs.shape, (2, 5, width))
                    obs = obs[:, -1]
                else:
                    self.assertEqual(obs.shape, (2, width))
                torch.testing.assert_close(obs[:, 19:25], position)
                torch.testing.assert_close(obs[:, 35:41], velocity)
            robot.set_joint_position_target(position, joint_ids=arm_ids)
            env.action_manager.process_action(torch.full((2, 10), 0.2, device=env.device))
            env.action_manager.apply_action()
            robot.write_data_to_sim()
            torch.testing.assert_close(robot.data.joint_pos_target[:, arm_ids], position)
            arm_actuators = robot.actuators[-1]
            torch.testing.assert_close(env.sim.data.ctrl[:, arm_actuators.ctrl_ids], position)
            obs, _, _, _, _ = env.step(torch.zeros((2, 10), device=env.device))
            torch.testing.assert_close(robot.data.joint_pos_target[:, arm_ids], position)
            torch.testing.assert_close(obs["actor"][:, -1, 19:25], robot.data.joint_pos[:, arm_ids])
            torch.testing.assert_close(obs["actor"][:, -1, 35:41], robot.data.joint_vel[:, arm_ids])
            self.assertGreater((obs["actor"][:, -1, 19:25] - position).abs().max().item(), 1e-5)
            env.reset()
            torch.testing.assert_close(
                robot.data.joint_pos_target[:, arm_ids],
                torch.zeros_like(position),
            )
            torch.testing.assert_close(
                robot.data.joint_pos[:, arm_ids], torch.zeros_like(position),
            )
        finally:
            env.close()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU smoke test")
    def test_baseline_excludes_arm_state(self):
        cfg = load_env_cfg("Mjlab-Velocity-Flat-TRON2A-SFYG-Baseline-CTS", play=True)
        env = ManagerBasedRlEnv(cfg, device="cuda:0")
        try:
            env.reset()
            robot = env.scene["robot"]
            arm_ids, _ = robot.find_joints(ARM_JOINTS, preserve_order=True)
            observations = {
                group: env.observation_manager.compute_group(group).clone()
                for group in ("actor", "critic")
            }
            position = torch.tensor([[0.1, 0.2, -0.3, 0.4, -0.5, 0.6]], device=env.device)
            velocity = torch.full_like(position, 0.5)
            robot.write_joint_state_to_sim(position, velocity, joint_ids=arm_ids)
            env.sim.forward()
            for group, width in (("actor", 44), ("critic", 56)):
                obs = env.observation_manager.compute_group(group, update_history=True)
                self.assertEqual(obs.shape, (1, 5, width) if group == "actor" else (1, width))
                torch.testing.assert_close(obs, observations[group])
            self.assertEqual(env.action_manager.total_action_dim, 10)
            robot.set_joint_position_target(position, joint_ids=arm_ids)
            env.step(torch.zeros((1, 10), device=env.device))
            torch.testing.assert_close(robot.data.joint_pos_target[:, arm_ids], position)
        finally:
            env.close()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU smoke test")
    def test_step_and_cts(self):
        for variant, baseline in product(VARIANTS, (False, True)):
            with self.subTest(variant=variant, baseline=baseline), tempfile.TemporaryDirectory() as log_dir:
                task = f"Mjlab-Velocity-Flat-TRON2A-{variant}" + ("-Baseline-CTS" if baseline else "-LBC-CTS")
                cfg = load_env_cfg(task)
                cfg.scene.num_envs = 4
                cfg.seed = 42
                env = ManagerBasedRlEnv(cfg, device="cuda:0")
                try:
                    env.reset()
                    actions = torch.zeros((4, 10), device=env.device)
                    clock = env.command_manager.get_term("twist").gait_clock
                    clock.phase[:] = torch.tensor([0.1, 0.2, 0.3, 0.4], device=env.device)
                    env.reset(env_ids=torch.tensor([1, 3], device=env.device))
                    torch.testing.assert_close(
                        clock.phase, torch.tensor([0.1, 0.0, 0.3, 0.0], device=env.device),
                    )
                    touched_ground = False
                    for step in range(100):
                        obs, reward, _, _, _ = env.step(actions)
                        self.assertEqual(obs["actor"].shape, (4, 5, 44 if baseline else 56))
                        self.assertEqual(obs["critic"].shape, (4, 56 if baseline else 68))
                        self.assertTrue(torch.isfinite(reward).all())
                        self.assertTrue(all(torch.isfinite(value).all() for value in obs.values()))
                        touched_ground |= bool(env.scene["feet_ground_contact"].data.found.any())
                    self.assertTrue(touched_ground)
                    runner_cfg = load_rl_cfg(task)
                    runner_cfg.num_steps_per_env = 8
                    runner_cfg.algorithm.num_learning_epochs = 1
                    runner_cfg.algorithm.num_mini_batches = 1
                    runner_cfg.algorithm.alignment_epochs = 1
                    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
                    runner = load_runner_cls(task)(wrapped, asdict(runner_cfg), log_dir, env.device)
                    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
                    self.assertTrue(all(torch.isfinite(param).all() for param in runner.alg.get_policy().parameters()))
                finally:
                    env.close()