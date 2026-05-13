from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_env import Go2Robot
from legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO, GO2CfgCTS, GO2CfgMoECTS, GO2CfgMoENGCTS, GO2CfgMCPCTS, GO2CfgACMoECTS, GO2CfgDualMoECTS
from legged_gym.envs.dog_v2.dog_v2_env import DogV2Robot
from legged_gym.envs.dog_v2.dog_v2_config import DogV2Cfg, DogV2CfgCTS, DogV2CfgMoECTS, DogV2CfgMoENGCTS, DogV2CfgMCPCTS, DogV2CfgACMoECTS, DogV2CfgDualMoECTS
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register("go2", Go2Robot, GO2Cfg(), GO2CfgPPO())
task_registry.register("go2_cts", Go2Robot, GO2Cfg(), GO2CfgCTS())
task_registry.register("go2_moe_cts", Go2Robot, GO2Cfg(), GO2CfgMoECTS())
task_registry.register("go2_moe_ng_cts", Go2Robot, GO2Cfg(), GO2CfgMoENGCTS())
task_registry.register("go2_mcp_cts", Go2Robot, GO2Cfg(), GO2CfgMCPCTS())
task_registry.register("go2_ac_moe_cts", Go2Robot, GO2Cfg(), GO2CfgACMoECTS())
task_registry.register("go2_dual_moe_cts", Go2Robot, GO2Cfg(), GO2CfgDualMoECTS())
task_registry.register("dog_v2_cts", DogV2Robot, DogV2Cfg(), DogV2CfgCTS())
task_registry.register("dog_v2_moe_cts", DogV2Robot, DogV2Cfg(), DogV2CfgMoECTS())
task_registry.register("dog_v2_moe_ng_cts", DogV2Robot, DogV2Cfg(), DogV2CfgMoENGCTS())
task_registry.register("dog_v2_mcp_cts", DogV2Robot, DogV2Cfg(), DogV2CfgMCPCTS())
task_registry.register("dog_v2_ac_moe_cts", DogV2Robot, DogV2Cfg(), DogV2CfgACMoECTS())
task_registry.register("dog_v2_dual_moe_cts", DogV2Robot, DogV2Cfg(), DogV2CfgDualMoECTS())
