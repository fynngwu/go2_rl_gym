import sys
from types import MethodType

import isaacgym
from isaacgym import gymapi
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


class KeyboardCommand:
    def __init__(self, lin_speed=1.0, yaw_speed=1.5):
        self.command = torch.tensor([[0.0, 0.0, 0.0]])
        self.lin_speed = lin_speed
        self.yaw_speed = yaw_speed
        self.reset_requested = False

    def handle(self, action):
        if action == "cmd_forward":
            self.command[0, 0] = self.lin_speed
        elif action == "cmd_back":
            self.command[0, 0] = -self.lin_speed
        elif action == "cmd_left":
            self.command[0, 1] = self.lin_speed
        elif action == "cmd_right":
            self.command[0, 1] = -self.lin_speed
        elif action == "cmd_yaw_left":
            self.command[0, 2] = self.yaw_speed
        elif action == "cmd_yaw_right":
            self.command[0, 2] = -self.yaw_speed
        elif action == "cmd_stop":
            self.command[:] = 0.0
        elif action == "reset_robot":
            self.reset_requested = True
        c = self.command[0]
        print(f"\rcmd x={c[0]: .1f} y={c[1]: .1f} yaw={c[2]: .1f}", end="", flush=True)

    def apply(self, env):
        env.commands[:, :3] = self.command.to(env.device)
        env.commands_resampling_step[:] = env.cfg.commands.resampling_time / env.dt
        env.stop_heading[:] = False


def install_keyboard_render(env, keyboard_command):
    if env.viewer is None:
        return

    key_actions = {
        gymapi.KEY_W: "cmd_forward",
        gymapi.KEY_S: "cmd_back",
        gymapi.KEY_Q: "cmd_left",
        gymapi.KEY_E: "cmd_right",
        gymapi.KEY_A: "cmd_yaw_left",
        gymapi.KEY_D: "cmd_yaw_right",
        gymapi.KEY_SPACE: "cmd_stop",
        gymapi.KEY_R: "reset_robot",
        gymapi.KEY_UP: "cmd_forward",
        gymapi.KEY_DOWN: "cmd_back",
        gymapi.KEY_LEFT: "cmd_yaw_left",
        gymapi.KEY_RIGHT: "cmd_yaw_right",
    }
    for key, action in key_actions.items():
        env.gym.subscribe_viewer_keyboard_event(env.viewer, key, action)

    def render_with_keyboard(self, sync_frame_time=True):
        if self.viewer:
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.value > 0:
                    keyboard_command.handle(evt.action)

            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)

            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    env.render = MethodType(render_with_keyboard, env)


def configure_play_env(env_cfg):
    env_cfg.env.num_envs = 1
    env_cfg.env.test = True
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = True
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.terrain.terrain_kwargs = {
        'type': 'terrain_utils.stairs_terrain',
        'terrain_kwargs': {'step_width': 0.31, 'step_height': -0.08},
    }
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_pd_gains = False
    env_cfg.domain_rand.randomize_motor_zero_offset = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.randomize_action_delay = False
    env_cfg.commands.resampling_time = 100000.0
    env_cfg.commands.zero_command_curriculum = None
    env_cfg.commands.limit_vel_prob = 0.0
    env_cfg.commands.limit_ang_vel_at_zero_command_prob = 0.0


def play(args):
    args.num_envs = 1
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    configure_play_env(env_cfg)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    keyboard_command = KeyboardCommand()
    install_keyboard_render(env, keyboard_command)

    if env.viewer is not None:
        origin = env.env_origins[0].detach().cpu().numpy()
        env.set_camera([origin[0] - 3.0, origin[1] - 5.0, origin[2] + 2.2],
                       [origin[0] + 2.0, origin[1], origin[2] + 0.6])

    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)

    print("Keyboard command control:")
    print("  W/S or Up/Down: lin_vel_x +/-")
    print("  Q/E: lin_vel_y +/-")
    print("  A/D or Left/Right: yaw +/-")
    print("  Space: stop, R: reset, V: toggle viewer sync, Esc: quit")

    while True:
        if keyboard_command.reset_requested:
            env.reset_idx(torch.arange(env.num_envs, device=env.device))
            keyboard_command.reset_requested = False
        keyboard_command.apply(env)
        env.compute_observations()
        obs = env.get_observations()
        with torch.no_grad():
            actions = policy(obs.detach())
        env.step(actions.detach())


if __name__ == "__main__":
    args = get_args()
    play(args)
