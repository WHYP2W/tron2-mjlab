"""CPU tests for alternating contact rewards and their policy clock."""

import unittest
from math import pi
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from mjlab.envs import ManagerBasedRlEnv

from tron2_mjlab.gait import (
    GaitClock, GaitClockCfg, GaitVelocityCommand, alternating_contacts, gait_phase,
)


class GaitTests(unittest.TestCase):
    def setUp(self):
        self.env = Mock(spec=ManagerBasedRlEnv)
        self.env.step_dt = 0.02
        self.env.num_envs = 4
        self.command = torch.tensor([[1.0, 0.0, 0.0]]).repeat(4, 1)
        self.clock = GaitClock(GaitClockCfg(), 4, "cpu")
        self.clock.phase[:] = torch.tensor([0.0, 0.2, 0.5, 0.7])
        self.term = Mock(spec=GaitVelocityCommand)
        self.term.command = self.command
        self.term.gait_clock = self.clock
        self.env.command_manager = Mock()
        self.env.command_manager.get_term.return_value = self.term
        self.contacts = SimpleNamespace(found=torch.tensor([[1, 1], [1, 0], [1, 1], [0, 1]]))
        self.env.scene = {"feet_ground_contact": SimpleNamespace(data=self.contacts)}
        self.clock_params = {"command_name": "twist"}
        self.reward_params = {
            **self.clock_params, "sensor_name": "feet_ground_contact", "stance_ratio": 0.6,
        }

    def test_alternation_and_clock(self):
        angle = torch.tensor([0.0, 0.4 * pi, pi, 1.4 * pi])
        torch.testing.assert_close(
            gait_phase(self.env, **self.clock_params),
            torch.stack((angle.sin(), angle.cos()), dim=-1),
        )
        torch.testing.assert_close(alternating_contacts(self.env, **self.reward_params), torch.ones(4))
        self.contacts.found = 1 - self.contacts.found
        torch.testing.assert_close(alternating_contacts(self.env, **self.reward_params), torch.zeros(4))

    def test_standing_and_deadband(self):
        self.command[:] = torch.tensor(
            [[0.0, 0.0, 0.0], [0.03, -0.03, 0.03], [0.05, 0.0, 0.0], [0.0, 0.0, -0.05]],
        )
        self.contacts.found = torch.tensor([[1, 1], [1, 1], [1, 0], [0, 0]])
        torch.testing.assert_close(gait_phase(self.env, **self.clock_params), torch.zeros((4, 2)))
        torch.testing.assert_close(
            alternating_contacts(self.env, **self.reward_params),
            torch.tensor([1.0, 1.0, 0.5, 0.0]),
        )

    def test_backward_lateral_and_turning_commands(self):
        self.command[:] = torch.tensor(
            [[-1.0, 0.0, 0.0], [0.0, -0.6, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
        )
        torch.testing.assert_close(
            gait_phase(self.env, **self.clock_params).square().sum(dim=1), torch.ones(4),
        )
        torch.testing.assert_close(alternating_contacts(self.env, **self.reward_params), torch.ones(4))

    def test_period_wrap_and_per_environment_reset(self):
        self.clock.phase[:] = 0.99
        self.clock.advance(self.command, 0.02, torch.ones(4, dtype=torch.bool))
        self.assertTrue((self.clock.phase < 0.03).all())
        before = gait_phase(self.env, **self.clock_params)
        self.clock.reset(torch.tensor([1]))
        after = gait_phase(self.env, **self.clock_params)
        torch.testing.assert_close(after[1], torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(after[[0, 2, 3]], before[[0, 2, 3]])
        self.assertAlmostEqual(self.clock.frequency[1].item(), 1 / 0.9, places=6)
        self.command[1] = 0.0
        old_phase = self.clock.phase.clone()
        self.clock.advance(self.command, 0.02, torch.tensor([False, True, True, True]))
        torch.testing.assert_close(self.clock.phase[:2], old_phase[:2])
        torch.testing.assert_close(gait_phase(self.env, **self.clock_params)[1], torch.zeros(2))

    def test_static_contact_patterns_do_not_get_full_cycle_reward(self):
        self.env.num_envs = 30
        self.term.command = torch.tensor([[1.0, 0.0, 0.0]]).repeat(30, 1)
        self.clock.phase = torch.arange(30) / 30
        for contact in ([1, 1], [0, 0], [1, 0], [0, 1]):
            with self.subTest(contact=contact):
                self.contacts.found = torch.tensor([contact]).repeat(30, 1)
                score = alternating_contacts(self.env, **self.reward_params)
                self.assertLess(score.mean().item(), 0.7)

    def test_invalid_parameters_and_missing_inputs(self):
        for params in (
            {"fast_period": 0.0}, {"slow_period": float("nan")}, {"linear_threshold": -1.0},
            {"angular_threshold": float("inf")}, {"smoothing_time": 0.0},
            {"fast_speed": 0.1}, {"fast_yaw_speed": 0.01}, {"fast_period": 1.0},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                GaitClockCfg(**params)
        for dt in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                self.clock.advance(self.command, dt, torch.ones(4, dtype=torch.bool))
        for ratio in (0.49, 1.0, float("nan")):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                alternating_contacts(self.env, **{**self.reward_params, "stance_ratio": ratio})
        for found in (None, torch.zeros((4, 3))):
            self.contacts.found = found
            with self.assertRaises(ValueError):
                alternating_contacts(self.env, **self.reward_params)
        self.env.command_manager.get_term.return_value = None
        with self.assertRaises(ValueError):
            gait_phase(self.env, **self.clock_params)

    def test_speed_to_period_mapping(self):
        for command, expected in (
            ([0, 0, 0], 0.9), ([0.2, 0, 0], 0.9), ([0.85, 0, 0], 0.75),
            ([1.5, 0, 0], 0.6), ([2, 0, 0], 0.6), ([-1.5, 0, 0], 0.6),
            ([0, -1.5, 0], 0.6), ([0.9, 1.2, 0], 0.6),
            ([0, 0, 0.525], 0.75), ([0, 0, -1], 0.6), ([0.2, 0, 1], 0.6),
        ):
            with self.subTest(command=command):
                actual = self.clock.target_period(torch.tensor([command], dtype=torch.float32))
                self.assertAlmostEqual(actual.item(), expected, places=6)

    def test_smooth_frequency_and_continuous_phase(self):
        self.clock.phase[:] = 0.2
        self.command[:] = torch.tensor([2.0, 0, 0])
        active = torch.ones(4, dtype=torch.bool)
        self.clock.advance(self.command, 0.02, active)
        self.assertTrue((self.clock.frequency > 1 / 0.9).all())
        self.assertTrue((self.clock.frequency < 1 / 0.6).all())
        torch.testing.assert_close(self.clock.phase, 0.2 + 0.02 * self.clock.frequency)
        for _ in range(200):
            old_phase = self.clock.phase.clone()
            self.clock.advance(self.command, 0.02, active)
            torch.testing.assert_close(
                self.clock.phase, torch.remainder(old_phase + 0.02 * self.clock.frequency, 1.0),
            )
        torch.testing.assert_close(self.clock.frequency, torch.full((4,), 1 / 0.6))
        old_phase = self.clock.phase.clone()
        self.command[:] = torch.tensor([0.1, 0, 0])
        self.clock.advance(self.command, 0.02, active)
        self.assertTrue((self.clock.frequency < 1 / 0.6).all())
        torch.testing.assert_close(
            self.clock.phase, torch.remainder(old_phase + 0.02 * self.clock.frequency, 1.0),
        )
        self.command.zero_()
        stopped_phase = self.clock.phase.clone()
        for _ in range(200):
            self.clock.advance(self.command, 0.02, active)
        torch.testing.assert_close(self.clock.phase, stopped_phase)
        torch.testing.assert_close(self.clock.frequency, torch.full((4,), 1 / 0.9))
        self.command[:] = torch.tensor([0, 0, 1.0])
        self.clock.advance(self.command, 0.02, active)
        torch.testing.assert_close(
            self.clock.phase, torch.remainder(stopped_phase + 0.02 * self.clock.frequency, 1.0),
        )

    def test_observation_and_reward_reads_do_not_advance_clock(self):
        before = self.clock.phase.clone()
        for _ in range(5):
            gait_phase(self.env, **self.clock_params)
            alternating_contacts(self.env, **self.reward_params)
        torch.testing.assert_close(self.clock.phase, before)
