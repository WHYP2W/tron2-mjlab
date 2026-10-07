"""Run with: uv run python -m unittest discover -s tests -v."""

import tempfile
import unittest
from dataclasses import asdict
from itertools import product
from unittest.mock import patch

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls

from tron2_mjlab.env_cfgs import tron2_flat_env_cfg
from tron2_mjlab.play import ManualVelocityCommandCfg, VelocityViewer, main as play_main
from tron2_mjlab.robot import ARM_JOINTS, ASSET_ROOT, LEG_JOINTS, VARIANTS, get_robot_cfg, get_spec


class Tron2Tests(unittest.TestCase):
    def test_models_and_configs(self):
        self.assertEqual(VARIANTS, ("SFYG",))
        self.assertFalse((ASSET_ROOT / "WFYG_TRON2A").exists())
        self.assertEqual(
            [task for task in list_tasks() if "TRON2" in task],
            [f"Mjlab-Velocity-Flat-TRON2A-{variant}{suffix}"
             for variant, suffix in product(VARIANTS, ("-Baseline", "-LBC"))],
        )
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = Entity(get_robot_cfg(variant)).spec.compile()
                self.assertEqual((model.nq, model.nv, model.nu), (23, 22, 16))
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
                self.assertEqual(cfg.commands["twist"].ranges.lin_vel_y, (-0.3, 0.3))
        for variant in ("WFYG", "SFYG_TRON2B"):
            with self.subTest(unsupported_variant=variant):
                with self.assertRaises(ValueError):
                    tron2_flat_env_cfg(variant)
                with self.assertRaises(ValueError):
                    get_spec(variant)

    def test_baseline_configs(self):
        task = "Mjlab-Velocity-Flat-TRON2A-SFYG"
        for play in (False, True):
            with self.subTest(play=play):
                full = load_env_cfg(task + "-LBC", play=play)
                baseline = load_env_cfg(task + "-Baseline", play=play)
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
        full_rl = load_rl_cfg(task + "-LBC")
        baseline_rl = load_rl_cfg(task + "-Baseline")
        self.assertEqual(full_rl.experiment_name, "tron2a_sfyg_velocity_lbc")
        self.assertEqual(baseline_rl.experiment_name, "tron2a_sfyg_velocity_baseline")
        baseline_rl.experiment_name = full_rl.experiment_name
        self.assertEqual(asdict(full_rl), asdict(baseline_rl))

    def test_play_task_selection(self):
        for baseline in (False, True):
            task = "Mjlab-Velocity-Flat-TRON2A-SFYG" + ("-Baseline" if baseline else "-LBC")
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
                    command = env.command_manager.get_term("twist")
                    target = (0.5, 0.2, -0.3)
                    command.manual_target = target
                    command.time_left[:] = 0.0
                    env.step(torch.zeros((1, 10), device=env.device))
                    torch.testing.assert_close(command.command[0], torch.tensor(target, device=env.device))
                    env.reset()
                    torch.testing.assert_close(command.command[0], torch.tensor(target, device=env.device))
                    command.manual_target = (0.0, 0.0, 0.0)
                    command._update_command()
                    self.assertFalse(command.command.any())
                    command.manual_target = None
                    command._resample_command(torch.arange(env.num_envs, device=env.device))
                    self.assertTrue(torch.isfinite(command.command).all())
                    wrapped = RslRlVecEnvWrapper(env)
                    viewer = VelocityViewer(wrapped, lambda obs: torch.zeros((1, 10), device=env.device))
                    viewer.setup()
                    try:
                        viewer.sliders[0].value = 0.5
                        viewer.sliders[1].value = 0.2
                        viewer.sliders[2].value = -0.3
                        viewer.sync_viewer_to_env()
                        torch.testing.assert_close(wrapped.get_observations()["actor"][0, -3:],
                                                   torch.tensor(target, device=env.device))
                        self.assertFalse(viewer.sliders[1].disabled)
                        viewer.manual.value = False
                        viewer.sync_viewer_to_env()
                        self.assertIsNone(command.manual_target)
                        viewer._stop(None)
                        viewer.sync_viewer_to_env()
                        self.assertFalse(command.command.any())
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
            for group, width in (("actor", 54), ("critic", 66)):
                obs = env.observation_manager.compute_group(group)
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
            torch.testing.assert_close(obs["actor"][:, 19:25], robot.data.joint_pos[:, arm_ids])
            torch.testing.assert_close(obs["actor"][:, 35:41], robot.data.joint_vel[:, arm_ids])
            self.assertGreater((obs["actor"][:, 19:25] - position).abs().max().item(), 1e-5)
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
        cfg = load_env_cfg("Mjlab-Velocity-Flat-TRON2A-SFYG-Baseline", play=True)
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
            for group, width in (("actor", 42), ("critic", 54)):
                obs = env.observation_manager.compute_group(group)
                self.assertEqual(obs.shape, (1, width))
                torch.testing.assert_close(obs, observations[group])
            self.assertEqual(env.action_manager.total_action_dim, 10)
            robot.set_joint_position_target(position, joint_ids=arm_ids)
            env.step(torch.zeros((1, 10), device=env.device))
            torch.testing.assert_close(robot.data.joint_pos_target[:, arm_ids], position)
        finally:
            env.close()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU smoke test")
    def test_step_and_ppo(self):
        for variant, baseline in product(VARIANTS, (False, True)):
            with self.subTest(variant=variant, baseline=baseline), tempfile.TemporaryDirectory() as log_dir:
                task = f"Mjlab-Velocity-Flat-TRON2A-{variant}" + ("-Baseline" if baseline else "-LBC")
                cfg = load_env_cfg(task)
                cfg.scene.num_envs = 4
                cfg.seed = 42
                env = ManagerBasedRlEnv(cfg, device="cuda:0")
                try:
                    env.reset()
                    actions = torch.zeros((4, 10), device=env.device)
                    touched_ground = False
                    for step in range(100):
                        obs, reward, _, _, _ = env.step(actions)
                        self.assertEqual(obs["actor"].shape, (4, 42 if baseline else 54))
                        self.assertEqual(obs["critic"].shape, (4, 54 if baseline else 66))
                        self.assertTrue(torch.isfinite(reward).all())
                        self.assertTrue(all(torch.isfinite(value).all() for value in obs.values()))
                        touched_ground |= bool(env.scene["feet_ground_contact"].data.found.any())
                    self.assertTrue(touched_ground)
                    runner_cfg = load_rl_cfg(task)
                    runner_cfg.num_steps_per_env = 8
                    runner_cfg.algorithm.num_learning_epochs = 1
                    runner_cfg.algorithm.num_mini_batches = 1
                    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
                    runner = load_runner_cls(task)(wrapped, asdict(runner_cfg), log_dir, env.device)
                    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
                    self.assertTrue(all(torch.isfinite(param).all() for param in runner.alg.get_policy().parameters()))
                finally:
                    env.close()