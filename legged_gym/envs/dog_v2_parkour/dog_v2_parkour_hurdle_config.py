from legged_gym.envs.dog_v2.dog_v2_config import DogV2Cfg, DogV2CfgCTS


class DogV2ParkourHurdleCfg(DogV2Cfg):
    class env(DogV2Cfg.env):
        num_envs = 4096
        num_observations = 45
        num_privileged_obs = 45 + 3 + 4 + 12 + 12 + 187
        episode_length_s = 20
        randomize_start_pos = False
        randomize_start_yaw = False
        randomize_start_vel = False
        randomize_start_y = True
        rand_y_range = 0.1
        next_goal_threshold = 0.2
        reach_goal_delay = 0.1
        num_future_goal_obs = 2
        goal_distance_range = 4.0

    class depth:
        use_camera = False

    class viewer(DogV2Cfg.viewer):
        pos = [40.0, -12.0, 45.0]
        lookat = [40.0, 2.0, 0.0]

    class terrain(DogV2Cfg.terrain):
        mesh_type = "trimesh"
        curriculum = True
        terrain_length = 8.0
        terrain_width = 4.0
        num_rows = 10
        num_cols = 32
        max_init_terrain_level = 2
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 5.0
        measure_heights = True
        num_goals = 2
        y_range = [-0.4, 0.4]
        edge_width_thresh = 0.05
        draw_edge_mask = True
        edge_mask_draw_stride = 2
        origin_zero_z = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

    class commands(DogV2Cfg.commands):
        curriculum = False
        heading_command = False
        resampling_time = 6.0
        zero_command_curriculum = None
        limit_vel_prob = 0.0
        limit_ang_vel_at_zero_command_prob = 0.0
        dynamic_resample_commands = False
        command_range_curriculum = []
        terrain_max_command_ranges = []

        class ranges:
            lin_vel_x = [0.3, 0.8]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class domain_rand(DogV2Cfg.domain_rand):
        randomize_friction = True
        friction_range = [0.6, 2.0]

        randomize_base_mass = True
        added_mass_range = [0.0, 3.0]

        randomize_link_mass = False

        randomize_base_com = True
        added_base_com_range = [-0.2, 0.2]

        randomize_restitution = False

        randomize_pd_gains = False
        randomize_motor_zero_offset = False

        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]

        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.5
        max_push_ang_vel = 0.0

        randomize_action_delay = False

    class rewards(DogV2Cfg.rewards):
        only_positive_rewards = True
        curriculum_rewards = []
        dynamic_sigma = None
        tracking_sigma = 0.2

        class scales:
            tracking_goal_vel = 1.5
            tracking_yaw = 0.5
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.0
            dof_acc = -2.5e-7
            collision = -10.0
            action_rate = -0.1
            delta_torques = -1e-7
            torques = -1e-5
            hip_pos = -0.5
            dof_error = -0.04
            feet_stumble = -1.0
            feet_edge = -1.0
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            feet_air_time = 0.0
            stand_still = 0.0
            hip_to_default = 0.0
            correct_base_height = 0.0
            action_smoothness = 0.0
            dof_power = 0.0
            dof_pos_limits = 0.0
            feet_regulation = 0.0


class DogV2ParkourHurdleCfgCTS(DogV2CfgCTS):
    class runner(DogV2CfgCTS.runner):
        experiment_name = "dog_v2_parkour_hurdle_cts"
        max_iterations = 150000
        save_interval = 500
        num_steps_per_env = 24

    class policy(DogV2CfgCTS.policy):
        latent_dim = 32
        norm_type = "l2norm"
