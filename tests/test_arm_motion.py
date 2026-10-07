"""Random arm trajectories stay smooth, bounded and separate from leg actions."""

import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from mjlab.tasks.registry import load_env_cfg

from tron2_mjlab.arm_motion import RandomArmMotion, RandomArmMotionCfg
from tron2_mjlab.robot import ARM_JOINTS


class ArmMotionTests(unittest.TestCase):
    def make_command(self, cfg=None):
        robot = Mock()
        targets = torch.zeros((8, 16))
        targets[:, :10] = 0.3
        robot.data = SimpleNamespace(
            joint_pos_target=targets,
            soft_joint_pos_limits=torch.tensor([-2.0, 2.0]).repeat(8, 16, 1),
            default_joint_pos=torch.zeros((8, 16)),
        )
        robot.find_joints.return_value = (list(range(10, 16)), ARM_JOINTS)

        def set_target(position, joint_ids):
            targets[:, joint_ids] = position

        robot.set_joint_position_target.side_effect = set_target
        env = SimpleNamespace(
            num_envs=8, device="cpu", scene={"robot": robot},
            episode_length_buf=torch.zeros(8, dtype=torch.long),
        )
        command = RandomArmMotion(cfg or RandomArmMotionCfg(nominal_probability=0), env)
        command.reset(torch.arange(8))
        return command, env, robot

    def test_configs_and_validation(self):
        configs = []
        for name in ("Baseline", "LBC"):
            task = f"Mjlab-Velocity-Flat-TRON2A-SFYG-{name}-CTS"
            cfg = load_env_cfg(task).commands["arm_motion"]
            self.assertEqual(cfg.resampling_time_range, (2.0, 4.0))
            self.assertEqual(cfg.max_target_speed, 1.0)
            configs.append(asdict(cfg))
            self.assertNotIn("arm_motion", load_env_cfg(task, play=True).commands)
        self.assertEqual(configs[0], configs[1])
        for params in (
            {"max_target_speed": 0}, {"max_target_speed": float("nan")},
            {"resampling_time_range": (0, 2)}, {"resampling_time_range": (4, 2)},
            {"range_fraction": 1.1}, {"nominal_probability": -1},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                RandomArmMotionCfg(**params)

    def test_continuous_motion_speed_bound_and_target_resampling(self):
        torch.manual_seed(1)
        command, env, robot = self.make_command()
        env.episode_length_buf[:] = 1
        initial = command.command.clone()
        moved = torch.zeros(8, dtype=torch.bool)
        for _ in range(650):
            previous = command.command.clone()
            command.compute(0.02)
            self.assertLessEqual((command.command - previous).abs().max().item(), 0.02001)
            self.assertTrue((command.command.abs() <= 1.60001).all())
            torch.testing.assert_close(robot.data.joint_pos_target[:, 10:], command.command)
            torch.testing.assert_close(robot.data.joint_pos_target[:, :10], torch.full((8, 10), 0.3))
            moved |= (command.command - initial).abs().amax(dim=1) > 0.1
        self.assertTrue(moved.all())
        self.assertTrue((command.command_counter >= 3).all())
        self.assertGreater(command.command.std(dim=0).max().item(), 0.1)

    def test_long_moves_extend_duration_and_end_without_jumps(self):
        cfg = RandomArmMotionCfg(resampling_time_range=(0.1, 0.1), max_target_speed=0.25)
        command, env, robot = self.make_command(cfg)
        robot.data.joint_pos_target[:, 10:] = -1.6
        command.time_left[:] = 0.1
        with patch("tron2_mjlab.arm_motion.sample_arm_pose", return_value=torch.full((8, 6), 1.6)):
            command._resample_command(torch.arange(8))
        torch.testing.assert_close(command.duration, torch.full((8,), 1.875 * 3.2 / 0.25))
        start = command.command.clone()
        command._update_command()
        torch.testing.assert_close(command.command, start)
        command.time_left[:] = command.duration / 2
        command._update_command()
        torch.testing.assert_close(command.command, torch.zeros_like(command.command), atol=1e-6, rtol=0)
        command.time_left.zero_()
        command._update_command()
        torch.testing.assert_close(command.command, torch.full((8, 6), 1.6))
        before = command.command.clone()
        command._resample(torch.arange(8))
        command._update_command()
        torch.testing.assert_close(command.command, before)

    def test_partial_reset_and_no_simulation_state_writes(self):
        command, env, robot = self.make_command()
        before = command.command.clone()
        command.compute(0.02)
        torch.testing.assert_close(command.command, before)
        env.episode_length_buf[:] = 1
        for _ in range(20):
            command.compute(0.02)
        old_command = command.command.clone()
        old_goal = command.goal.clone()
        old_time = command.time_left.clone()
        robot.data.joint_pos_target[2, 10:] = 0.75
        command.reset(torch.tensor([2]))
        torch.testing.assert_close(command.command[2], torch.full((6,), 0.75))
        keep = torch.tensor([0, 1, 3, 4, 5, 6, 7])
        torch.testing.assert_close(command.command[keep], old_command[keep])
        torch.testing.assert_close(command.goal[keep], old_goal[keep])
        torch.testing.assert_close(command.time_left[keep], old_time[keep])
        env.episode_length_buf[2] = 0
        command.compute(0.02)
        torch.testing.assert_close(command.command[2], torch.full((6,), 0.75))
        robot.write_joint_state_to_sim.assert_not_called()
        for dt in (0, -0.02, float("nan")):
            with self.assertRaises(ValueError):
                command.compute(dt)
