# AGENTS.md
go2_rl_gym 当前目标：dog_v2 复用 Go2 CTS 训练链路，并验证 Isaac Gym 到 MuJoCo/ONNX 部署。

## 项目规则
- 只做与当前机器人/训练目标直接相关的改动；避免无关重构。
- 修改 dog_v2 资产、奖励或训练参数后，必须做一次 env 加载或短训练验证。
- 重要状态变化后更新本文件；保持 70 行以内。

## 当前状态
- 当前训练：tmux session `dogv2_cts_to50k` 正在跑 `dog_v2_cts --resume --checkpoint 7000 --num_envs=6200 --max_iterations=43000`。
- 训练 stdout：`logs/dog_v2_cts_to50k.log`；查看 `tmux capture-pane -t dogv2_cts_to50k -p -S -120`。
- 当前训练 run：`logs/dog_v2_cts/May07_23-57-46_`，从 `May07_17-18-02_/model_7000.pt` 继续到约 `50001`。
- TensorBoard：tmux session `dogv2_tensorboard` 跑 `--logdir logs/dog_v2_cts --port 6006 --load_fast=false`，用 `tools/tensorboard_compat/sitecustomize.py` 兼容 protobuf 5。
- 当前 ONNX：`logs/dog_v2_cts/exported/policies/policy.onnx`，输入 `1x225=5x45 history`，输出 `1x12 actions`。
- MuJoCo obs：history 顺序是 old->new；base ang vel 必须用 `quat_rotate_inverse(qpos[3:7], qvel[3:6])`，不要用 `mj_objectVelocity(local=1)`。
- MuJoCo init：dog_v2 deploy 使用 `init_base_pos=[0,0,0.42]`，与训练 run config 对齐。
- GPU：RTX 4060 Laptop 8GB；默认 `8192`、`7004`、`6912`、`6656`、`6528`、`6464` 均 OOM。
- 已验证可跑：`4096`、`6144`、`6400` 能完成 `max_iterations=1`。
- 当前经验上限：`6400` 可过一次迭代；`6464` 会 OOM，实训建议 `6200` 留余量。
- 需确认：当前 `DogV2Cfg.rewards.base_height_target` 实际继承 Go2 `0.38`；此前目标文档写过 `0.34`。
- dog_v2 默认关节角：12 个 Go2 风格关节名全部 `0.0`。
- dog_v2 env：已简化为直接继承 `Go2Robot`，无额外索引覆盖。
- Go2 command：非 heading 模式；初始 `x[-0.5,1.0] y[-0.5,0.5] yaw[-0.5,0.5]`。
- Go2 command curriculum：2000 轮到 `x[-1,2] y[-1,1] yaw[-1.5,1.5]`；5000 轮到 `x[-2,2.5] y[-1.5,1.5] yaw[-2,2]`。
- terrain command cap：stairs up/down 的 `lin_vel_x` 限制 `[-1.0,1.5]`；所有地形 yaw cap `[-2,2]`。
- URDF：`dog_v2_2_4.urdf` 的 DOF 顺序已对齐 Go2：`FL, FR, RL, RR`。
- foot fixed joints：已加 `dont_collapse="true"`，保证 `foot_name="foot"` 能找到 foot body。

## 重要文件

### 训练
- `legged_gym/scripts/train.py`: 训练入口。
- `legged_gym/utils/task_registry.py`: task 注册、env/runner 创建。
- `rsl_rl/rsl_rl/runners/on_policy_runner_cts.py`: CTS runner。
- `rsl_rl/rsl_rl/algorithms/cts.py`: CTS update/OOM 热点。

### dog_v2
- `legged_gym/envs/dog_v2/dog_v2_env.py`: dog_v2 env，当前仅继承 Go2。
- `legged_gym/envs/dog_v2/dog_v2_config.py`: dog_v2 asset、base height、默认角和 CTS experiment names。
- `resources/robots/dog_v2/urdf/dog_v2_2_4.urdf`: 当前训练用 URDF。
- `resources/robots/dog_v2/meshes/`: dog_v2 mesh assets。
- `deploy/deploy_mujoco/configs/dog_v2.yaml`: dog_v2 ONNX/MuJoCo 键盘部署配置。
- `deploy/deploy_mujoco/deploy_go2.py`: ONNX history obs 和键盘控制实现。

### Go2 继承链
- `legged_gym/envs/go2/go2_env.py`: observation 和 hip reward 逻辑。
- `legged_gym/envs/go2/go2_config.py`: dog_v2 继承的 CTS/terrain/reward/control 默认配置。
- `legged_gym/envs/base/legged_robot.py`: feet/hip/body/dof indices、reward 实现。
