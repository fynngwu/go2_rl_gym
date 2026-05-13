from legged_gym.envs.go2.go2_config import (
    GO2Cfg,
    GO2CfgCTS,
    GO2CfgMoECTS,
    GO2CfgMoENGCTS,
    GO2CfgMCPCTS,
    GO2CfgACMoECTS,
    GO2CfgDualMoECTS,
)


class DogV2Cfg(GO2Cfg):
    class init_state(GO2Cfg.init_state):
        default_joint_angles = {
            'FL_hip_joint': 0.0,
            'FL_thigh_joint': 0.0,
            'FL_calf_joint': 0.0,
            'FR_hip_joint': 0.0,
            'FR_thigh_joint': 0.0,
            'FR_calf_joint': 0.0,
            'RL_hip_joint': 0.0,
            'RL_thigh_joint': 0.0,
            'RL_calf_joint': 0.0,
            'RR_hip_joint': 0.0,
            'RR_thigh_joint': 0.0,
            'RR_calf_joint': 0.0,
        }

    class asset(GO2Cfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/dog_v2/urdf/dog_v2_2_4.urdf'
        name = "dog_v2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        flip_visual_attachments = False

    # class rewards(GO2Cfg.rewards):
    #     base_height_target = 0.34  # inherit GO2's 0.38


class DogV2CfgCTS(GO2CfgCTS):
    class runner(GO2CfgCTS.runner):
        experiment_name = 'dog_v2_cts'


class DogV2CfgMoENGCTS(GO2CfgMoENGCTS):
    class runner(GO2CfgMoENGCTS.runner):
        experiment_name = 'dog_v2_moe_no_goal_cts'


class DogV2CfgMCPCTS(GO2CfgMCPCTS):
    class runner(GO2CfgMCPCTS.runner):
        experiment_name = 'dog_v2_mcp_cts'


class DogV2CfgACMoECTS(GO2CfgACMoECTS):
    class runner(GO2CfgACMoECTS.runner):
        experiment_name = 'dog_v2_ac_moe_cts'


class DogV2CfgDualMoECTS(GO2CfgDualMoECTS):
    class runner(GO2CfgDualMoECTS.runner):
        experiment_name = 'dog_v2_dual_moe_cts'


class DogV2CfgMoECTS(GO2CfgMoECTS):
    class policy(GO2CfgMoECTS.policy):
        expert_num = 4
        student_encoder_hidden_dims = [256, 128]

    class runner(GO2CfgMoECTS.runner):
        experiment_name = 'dog_v2_moe_cts'
