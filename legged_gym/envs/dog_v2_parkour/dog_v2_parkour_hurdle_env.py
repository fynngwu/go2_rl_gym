import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float

from legged_gym.envs.dog_v2.dog_v2_env import DogV2Robot
from legged_gym.envs.dog_v2_parkour.parkour_debug_viz import draw_edge_mask, draw_goal_marker
from legged_gym.utils.math import wrap_to_pi

from legged_gym.utils.parkour_hurdle_terrain import ParkourHurdleTerrain


class DogV2ParkourHurdleRobot(DogV2Robot):
    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.terrain = ParkourHurdleTerrain(self.cfg.terrain, self.num_envs)
        self._create_trimesh()
        self._create_envs()

    def _init_buffers(self):
        super()._init_buffers()
        self.reach_goal_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reached_goal_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.target_pos_rel = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)
        self.next_target_pos_rel = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float)
        self.target_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.next_target_yaw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask, device=self.device)
        self.last_torques = torch.zeros_like(self.torques)
        self.hurdle_sensor_offsets = torch.arange(
            self.cfg.terrain.horizontal_scale,
            self.cfg.env.hurdle_sensor_range + 1e-6,
            self.cfg.terrain.horizontal_scale,
            device=self.device,
        )
        self._update_goals()
        assert self.num_dof == 12
        assert self.num_actions == 12
        assert len(self.feet_indices) == 4
        assert self.obs_buf.shape[1] == self.cfg.env.num_observations

    def _post_physics_step_callback(self):
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        self._update_goals()
        if self.cfg.env.test and self.viewer and self.enable_viewer_sync:
            self.gym.clear_lines(self.viewer)
            draw_goal_marker(self)
            if getattr(self.cfg.terrain, "draw_edge_mask", False):
                draw_edge_mask(self)
        if self.cfg.env.test and self.common_step_counter % 25 == 0:
            self._print_play_goal_command()

    def _refresh_env_goals(self):
        temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
        last_col = temp[:, -1].unsqueeze(1)
        future = last_col.repeat(1, self.cfg.env.num_future_goal_obs, 1)
        self.env_goals[:] = torch.cat((temp, future), dim=1)
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)

    def _get_env_origins(self):
        self.custom_origins = True
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float)
        self.env_class = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        max_init_level = self.cfg.terrain.max_init_terrain_level
        if not self.cfg.terrain.curriculum:
            max_init_level = self.cfg.terrain.num_rows - 1
        self.terrain_levels = torch.randint(0, max_init_level + 1, (self.num_envs,), device=self.device)
        per_col = self.num_envs / self.cfg.terrain.num_cols
        self.terrain_types = torch.div(
            torch.arange(self.num_envs, device=self.device),
            per_col,
            rounding_mode="floor",
        ).to(torch.long)
        self.max_terrain_level = self.cfg.terrain.num_rows
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
        self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.long)
        self.terrain_goals = torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float)
        self.env_goals = torch.zeros(self.num_envs, self.cfg.terrain.num_goals + self.cfg.env.num_future_goal_obs, 3, device=self.device, dtype=torch.float)
        self.cur_goal_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]
        self._refresh_env_goals()

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if self.cfg.env.randomize_start_y:
            self.root_states[env_ids, 1] += self.cfg.env.rand_y_range * torch_rand_float(
                -1.0, 1.0, (len(env_ids), 1), device=self.device
            ).squeeze(1)
        self.root_states[env_ids, 7:13] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super().reset_idx(env_ids)
        self.cur_goal_idx[env_ids] = 0
        self.reach_goal_timer[env_ids] = 0.0
        self.reached_goal_ids[env_ids] = False
        self.last_torques[env_ids] = 0.0
        self._refresh_env_goals()
        self._update_goals()

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        self.commands_resampling_step[env_ids] = self.cfg.commands.resampling_time / self.dt
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1:4] = 0.0

    def _update_goals(self):
        next_flag = self.reach_goal_timer > self.cfg.env.reach_goal_delay / self.dt
        self.cur_goal_idx[next_flag] += 1
        self.reach_goal_timer[next_flag] = 0.0
        self.cur_goals = self._gather_cur_goals()
        self.next_goals = self._gather_cur_goals(future=1)
        self.reached_goal_ids = torch.norm(self.root_states[:, :2] - self.cur_goals[:, :2], dim=1) < self.cfg.env.next_goal_threshold
        self.reach_goal_timer[self.reached_goal_ids] += 1
        self.target_pos_rel = self.cur_goals[:, :2] - self.root_states[:, :2]
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_states[:, :2]
        self.target_yaw = torch.atan2(self.target_pos_rel[:, 1], self.target_pos_rel[:, 0])
        self.next_target_yaw = torch.atan2(self.next_target_pos_rel[:, 1], self.next_target_pos_rel[:, 0])

    def _gather_cur_goals(self, future=0):
        goal_ids = (self.cur_goal_idx[:, None, None] + future).expand(-1, -1, self.env_goals.shape[-1])
        return self.env_goals.gather(1, goal_ids).squeeze(1)

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done:
            return
        goal_counts = self.cur_goal_idx[env_ids]
        move_up = goal_counts >= self.cfg.terrain.num_goals
        move_down = goal_counts < max(1, self.cfg.terrain.num_goals // 3)
        self.terrain_levels[env_ids] += move_up.long() - move_down.long()
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clamp(self.terrain_levels[env_ids], min=0),
        )
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.env_class[env_ids] = self.terrain_class[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def check_termination(self):
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        roll_cutoff = torch.abs(self.rpy[:, 0]) > 1.5
        pitch_cutoff = torch.abs(self.rpy[:, 1]) > 1.5
        height_cutoff = self.root_states[:, 2] < 0.12

        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        reach_goal_cutoff = self.cur_goal_idx >= self.cfg.terrain.num_goals
        self.time_out_buf |= reach_goal_cutoff
        self.reset_buf |= self.time_out_buf | roll_cutoff | pitch_cutoff | height_cutoff

    def compute_observations(self):
        goal_cmd_obs = self._get_goal_cmd_obs()
        dof_pos = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dof_vel = self.dof_vel * self.obs_scales.dof_vel
        self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel, self.projected_gravity, goal_cmd_obs, dof_pos, dof_vel, self.actions), dim=-1)
        heights = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
            -1.0,
            1.0,
        ) * self.obs_scales.height_measurements
        self.privileged_obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                goal_cmd_obs,
                dof_pos,
                dof_vel,
                self.actions,
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) * 1e-3,
                self.torques / self.torque_limits,
                (self.last_dof_vel - self.dof_vel) / self.dt * 1e-4,
                heights,
            ),
            dim=-1,
        )
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _reward_tracking_goal_vel(self):
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        cur_vel_global = self.root_states[:, 7:9]
        forward_vel = torch.sum(target_vec_norm * cur_vel_global, dim=-1)
        target_speed = self.commands[:, 0]
        return torch.minimum(forward_vel, target_speed) / (target_speed + 1e-5)

    def _reward_tracking_yaw(self):
        return torch.exp(-torch.abs(wrap_to_pi(self.target_yaw - self.rpy[:, 2])))

    def _reward_feet_stumble(self):
        stumble = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
        support = 4 * torch.abs(self.contact_forces[:, self.feet_indices, 2])
        return torch.any(stumble > support, dim=1).float()

    def _reward_feet_edge(self):
        rigid_body_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        feet_pos_xy = ((rigid_body_states[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()
        feet_pos_xy[..., 0] = torch.clamp(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0] - 1)
        feet_pos_xy[..., 1] = torch.clamp(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1] - 1)
        feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        self.feet_at_edge = contact_filt & feet_at_edge
        return (self.terrain_levels > 3).float() * torch.sum(self.feet_at_edge, dim=-1).float()

    def _reward_delta_torques(self):
        rew = torch.sum(torch.square(self.torques - self.last_torques), dim=1)
        self.last_torques[:] = self.torques[:]
        return rew

    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_indices] - self.default_dof_pos[:, self.hip_indices]), dim=1)

    def _reward_dof_error(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _get_goal_cmd_obs(self):
        """Goal command: heading error, forward hurdle distance, target speed."""
        delta_yaw = wrap_to_pi(self.target_yaw - self.rpy[:, 2])
        distance = self._get_forward_hurdle_distance() / self.cfg.env.hurdle_sensor_range
        speed = self.commands[:, 0] * self.obs_scales.lin_vel

        return torch.stack((delta_yaw, distance, speed), dim=-1)

    def _get_forward_hurdle_distance(self):
        ray_x = self.root_states[:, 0:1] + self.hurdle_sensor_offsets.unsqueeze(0)
        ray_y = self.root_states[:, 1:2].expand_as(ray_x)
        scale = self.cfg.terrain.horizontal_scale
        border = self.cfg.terrain.border_size
        x_ids = ((ray_x + border) / scale).round().long().clamp(0, self.height_samples.shape[0] - 1)
        y_ids = ((ray_y + border) / scale).round().long().clamp(0, self.height_samples.shape[1] - 1)
        heights = self.height_samples[x_ids, y_ids] * self.cfg.terrain.vertical_scale
        hit = heights > self.cfg.env.hurdle_sensor_height_threshold
        hit_ids = torch.argmax(hit.float(), dim=1)
        has_hit = hit.any(dim=1)
        distances = self.hurdle_sensor_offsets[hit_ids]
        max_dist = torch.full_like(distances, self.cfg.env.hurdle_sensor_range)
        return torch.where(has_hit, distances, max_dist)

    def _print_play_goal_command(self):
        goal_cmd_obs = self._get_goal_cmd_obs()[0]
        goal_dist = torch.norm(self.target_pos_rel[0]).item()
        print(
            "[parkour play] env0 command: "
            f"target_speed={self.commands[0, 0].item():.3f} m/s, "
            f"delta_yaw={goal_cmd_obs[0].item():.3f} rad, "
            f"goal_dist={goal_dist:.3f} m, "
            f"hurdle_sensor={goal_cmd_obs[1].item() * self.cfg.env.hurdle_sensor_range:.3f} m, "
            f"hurdle_sensor_obs={goal_cmd_obs[1].item():.3f}",
            flush=True,
        )
