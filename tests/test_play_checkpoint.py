"""Reject wrong CTS checkpoints before constructing the simulator."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from tron2_mjlab.cts import validate_actor_checkpoint
from tron2_mjlab.play import main


def actor_state(width, frames=5):
    return {
        "obs_normalizer._mean": torch.zeros(1, width),
        "student_encoder.0.0.weight": torch.zeros(32, frames * width),
    }


class PlaybackCheckpointTests(unittest.TestCase):
    def test_matching_and_unnormalized_interfaces(self):
        for width in (44, 56):
            state = actor_state(width)
            validate_actor_checkpoint(state, width, 5)
            del state["obs_normalizer._mean"]
            validate_actor_checkpoint(state, width, 5)

    def test_invalid_history_or_checkpoint(self):
        for state in ({}, {"student_encoder.0.0.weight": None}, actor_state(56, frames=4)):
            with self.subTest(keys=list(state)), self.assertRaises(ValueError):
                validate_actor_checkpoint(state, 56, 5)

    def test_wrong_task_is_rejected_before_simulation(self):
        for width, baseline, hint in ((56, True, "remove --baseline"), (44, False, "add --baseline")):
            with self.subTest(width=width), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.pt"
                torch.save({"actor_state_dict": actor_state(width)}, path)
                argv = ["tron2-play", "SFYG", "--checkpoint-file", str(path)]
                if baseline:
                    argv.append("--baseline")
                stderr = io.StringIO()
                with patch("sys.argv", argv), patch("tron2_mjlab.play.ManagerBasedRlEnv") as env_cls:
                    with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                        main()
                    self.assertEqual(error.exception.code, 2)
                    env_cls.assert_not_called()
                self.assertIn(hint, stderr.getvalue())
                self.assertIn("Do not mix checkpoints", stderr.getvalue())

    def test_correct_task_continues_to_strict_load(self):
        for width, baseline in ((56, False), (44, True)):
            with self.subTest(width=width), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.pt"
                torch.save({"actor_state_dict": actor_state(width)}, path)
                argv = ["tron2-play", "SFYG", "--checkpoint-file", str(path), "--device", "cpu"]
                if baseline:
                    argv.append("--baseline")
                runner_class = Mock()
                with (
                    patch("sys.argv", argv),
                    patch("tron2_mjlab.play.ManagerBasedRlEnv") as env_cls,
                    patch("tron2_mjlab.play.RslRlVecEnvWrapper"),
                    patch("tron2_mjlab.play.load_runner_cls", return_value=runner_class),
                    patch("tron2_mjlab.play.VelocityViewer") as viewer,
                ):
                    env_cls.return_value.device = "cpu"
                    main()
                    runner_class.return_value.load.assert_called_once_with(
                        str(path), load_cfg={"actor": True}, strict=True, map_location="cpu",
                    )
                    viewer.return_value.run.assert_called_once()
                    env_cls.return_value.close.assert_called_once()

    def test_invalid_file_has_explicit_cli_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.pt"
            for contents in (b"", b"not a checkpoint"):
                path.write_bytes(contents)
                with (
                    patch("sys.argv", ["tron2-play", "SFYG", "--checkpoint-file", str(path)]),
                    patch("tron2_mjlab.play.ManagerBasedRlEnv") as env_cls,
                    contextlib.redirect_stderr(io.StringIO()) as stderr,
                    self.assertRaises(SystemExit) as error,
                ):
                    main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn("Cannot load checkpoint", stderr.getvalue())
                env_cls.assert_not_called()
