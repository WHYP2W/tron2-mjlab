"""TRON2A SFYG LBC-CTS and Baseline-CTS tasks."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import tron2_flat_env_cfg
from .robot import VARIANTS
from .cts import CtsActorCfg, CtsPpoCfg

for variant in VARIANTS:
    for baseline in (False, True):
        suffix = "-Baseline-CTS" if baseline else "-LBC-CTS"
        register_mjlab_task(
            task_id=f"Mjlab-Velocity-Flat-TRON2A-{variant}{suffix}",
            env_cfg=tron2_flat_env_cfg(variant, observe_arm=not baseline),
            play_env_cfg=tron2_flat_env_cfg(variant, play=True, observe_arm=not baseline),
            rl_cfg=RslRlOnPolicyRunnerCfg(
                actor=CtsActorCfg(
                    hidden_dims=(512, 256, 128), stochastic=True,
                ),
                critic=RslRlModelCfg(hidden_dims=(512, 256, 128), obs_normalization=True),
                algorithm=CtsPpoCfg(entropy_coef=0.01),
                experiment_name=f"tron2a_{variant.lower()}_velocity"
                + ("_baseline_cts" if baseline else "_lbc_cts"),
                logger="tensorboard",
                max_iterations=10_000,
            ),
            runner_cls=VelocityOnPolicyRunner,
        )
