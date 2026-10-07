"""TRON2A SFYG LBC and Baseline tasks."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import tron2_flat_env_cfg
from .robot import VARIANTS

for variant in VARIANTS:
    for baseline in (False, True):
        suffix = "-Baseline" if baseline else "-LBC"
        register_mjlab_task(
            task_id=f"Mjlab-Velocity-Flat-TRON2A-{variant}{suffix}",
            env_cfg=tron2_flat_env_cfg(variant, observe_arm=not baseline),
            play_env_cfg=tron2_flat_env_cfg(variant, play=True, observe_arm=not baseline),
            rl_cfg=RslRlOnPolicyRunnerCfg(
                actor=RslRlModelCfg(hidden_dims=(512, 256, 128), stochastic=True),
                critic=RslRlModelCfg(hidden_dims=(512, 256, 128)),
                algorithm=RslRlPpoAlgorithmCfg(entropy_coef=0.01),
                experiment_name=f"tron2a_{variant.lower()}_velocity"
                + ("_baseline" if baseline else "_lbc"),
                logger="tensorboard",
                max_iterations=10_000,
            ),
            runner_cls=VelocityOnPolicyRunner,
        )
