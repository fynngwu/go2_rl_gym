import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry, Logger
from legged_gym.utils.exporter import export_policy_as_jit, export_policy_as_onnx, export_policy_as_pkl

import numpy as np
import torch

def configure_play_terrain(env_cfg, terrain_name, difficulty):
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    terrain_name = terrain_name.lower()
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = True
    env_cfg.terrain.max_init_terrain_level = 0

    if terrain_name == "flat":
        env_cfg.terrain.terrain_kwargs = {
            'type': 'terrain_utils.random_uniform_terrain',
            'terrain_kwargs': {'min_height': 0.0, 'max_height': 0.0, 'step': 0.005, 'downsampled_scale': 0.2},
        }
    elif terrain_name in ["stairs_up", "stairs_down"]:
        step_height = 0.05 + 0.23 * difficulty
        if terrain_name == "stairs_up":
            step_height *= -1
        env_cfg.terrain.terrain_kwargs = {
            'type': 'terrain_utils.pyramid_stairs_terrain',
            'terrain_kwargs': {'step_width': 0.31, 'step_height': step_height, 'platform_size': 3.0},
        }
    elif terrain_name in ["slope_up", "slope_down"]:
        slope = 0.1 + 0.52 * difficulty
        if terrain_name == "slope_down":
            slope *= -1
        env_cfg.terrain.terrain_kwargs = {
            'type': 'terrain_utils.pyramid_sloped_terrain',
            'terrain_kwargs': {'slope': slope, 'platform_size': 3.0},
        }
    elif terrain_name == "wave":
        env_cfg.terrain.terrain_kwargs = {
            'type': 'terrain_utils.wave_terrain',
            'terrain_kwargs': {'num_waves': 5, 'amplitude': 0.1 + 0.2 * difficulty},
        }
    elif terrain_name == "obstacles":
        env_cfg.terrain.terrain_kwargs = {
            'type': 'terrain_utils.discrete_obstacles_terrain',
            'terrain_kwargs': {
                'max_height': 0.05 + 0.25 * difficulty,
                'min_size': 1.0,
                'max_size': 2.0,
                'num_rects': 20,
                'platform_size': 3.0,
            },
        }
    else:
        raise ValueError(f"Unsupported play terrain: {terrain_name}")

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    # env_cfg.terrain.mesh_type = 'plane'
    configure_play_terrain(env_cfg, args.play_terrain, args.play_difficulty)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_pd_gains = False
    env_cfg.domain_rand.randomize_motor_zero_offset = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        if hasattr(runner.alg, 'actor_critic'):
            model = runner.alg.actor_critic
        else:
            model = runner.alg.model
        export_policy_as_jit(model, path)
        export_policy_as_onnx(model, path)
        export_policy_as_pkl(model, path)
        print('Exported policy as jit script / onnx to: ', path)

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())

        if FIX_COMMAND:
            env.commands[:, 0] = 1.0
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0

        obs, _, rews, dones, infos = env.step(actions.detach())

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    FIX_COMMAND = True
    args = get_args()
    play(args)
