"""Run with: uv run python -m unittest discover -s tests -v."""

import tempfile
import unittest
from dataclasses import asdict

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_rl_cfg, load_runner_cls

from tron2_mjlab.env_cfgs import tron2_flat_env_cfg
from tron2_mjlab.play import ManualVelocityCommandCfg, VelocityViewer
from tron2_mjlab.robot import ASSET_ROOT, VARIANTS, get_robot_cfg, get_spec


class Tron2Tests(unittest.TestCase):
    def test_models_and_configs(self):
        self.assertEqual(VARIANTS, ("SFYG",))
        self.assertFalse((ASSET_ROOT / "WFYG_TRON2A").exists())
        self.assertEqual(
            [task for task in list_tasks() if "TRON2" in task],
            [f"Mjlab-Velocity-Flat-TRON2A-{variant}" for variant in VARIANTS],
        )
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = Entity(get_robot_cfg(variant)).spec.compile()
                self.assertEqual((model.nq, model.nv, model.nu), (17, 16, 10))
                for joint_id in range(model.njnt):
                    self.assertFalse(
                        (model.joint(joint_id).name or "").startswith(("arm", "gripper"))
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
                self.assertIn("foot_slip", cfg.rewards)
                self.assertEqual(cfg.commands["twist"].ranges.lin_vel_y, (-0.3, 0.3))
        for variant in ("WFYG", "SFYG_TRON2B"):
            with self.subTest(unsupported_variant=variant):
                with self.assertRaises(ValueError):
                    tron2_flat_env_cfg(variant)
                with self.assertRaises(ValueError):
                    get_spec(variant)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU smoke test")
    def test_manual_command(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                cfg = tron2_flat_env_cfg(variant, play=True)
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
    def test_step_and_ppo(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as log_dir:
                cfg = tron2_flat_env_cfg(variant)
                cfg.scene.num_envs = 4
                cfg.seed = 42
                env = ManagerBasedRlEnv(cfg, device="cuda:0")
                try:
                    env.reset()
                    actions = torch.zeros((4, 10), device=env.device)
                    touched_ground = False
                    for step in range(100):
                        obs, reward, _, _, _ = env.step(actions)
                        self.assertTrue(torch.isfinite(reward).all())
                        self.assertTrue(all(torch.isfinite(value).all() for value in obs.values()))
                        touched_ground |= bool(env.scene["feet_ground_contact"].data.found.any())
                    self.assertTrue(touched_ground)
                    task = f"Mjlab-Velocity-Flat-TRON2A-{variant}"
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