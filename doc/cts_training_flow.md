# CTS 训练流程全解

> CTS = Concurrent Teacher-Student，参考论文 [arXiv:2405.10830](https://arxiv.org/abs/2405.10830)。
> Teacher 编码器处理特权观测，Student 编码器处理观测历史，两者共享 Actor/Critic，同时训练并在线蒸馏。

---

## 目录

1. [整体数据流](#1-整体数据流)
2. [训练入口](#2-训练入口)
3. [CTS 算法](#3-cts-算法)
4. [CTS Runner](#4-cts-runner)
5. [观测 (Observations)](#5-观测-observations)
6. [奖励 (Rewards)](#6-奖励-rewards)
7. [域随机化 (Domain Randomization)](#7-域随机化-domain-randomization)
8. [课程学习 (Curriculum)](#8-课程学习-curriculum)
9. [dog_v2 与 Go2 的差异](#9-dog_v2-与-go2-的差异)

---

## 1. 整体数据流

```
train.py
  └─ task_registry.make_env("dog_v2_cts")
     └─ DogV2Robot(DogV2Cfg)     [继承 Go2Robot → LeggedRobot → BaseTask]
        ├─ _parse_cfg()           [解析 reward scales、command ranges、timing]
        ├─ create_sim()           [创建地形、环境、加载 URDF、设置 DR props]
        ├─ _init_buffers()        [GPU tensors、PD gains、history]
        └─ _prepare_reward_function()  [根据 scales 构建 reward 函数列表]

  └─ task_registry.make_alg_runner("dog_v2_cts")
     └─ OnPolicyRunnerCTS(env, train_cfg)
        ├─ ActorCriticCTS(obs=45, critic_obs=263, actions=12, history=5)
        │   ├─ teacher_encoder:  privileged_obs(263) → latent(32)
        │   ├─ student_encoder:  history(5×45=225) → latent(32)
        │   ├─ actor:   [latent(32) + obs(45)] → actions(12)
        │   └─ critic:  [latent(32) + privileged_obs(263)] → value(1)
        ├─ CTS(model, num_envs)
        │   ├─ optimizer1: teacher_encoder + critic + actor + std
        │   └─ optimizer2: student_encoder only
        └─ RolloutStorageCTS(teacher_envs, student_envs, history_length=5, steps=24)

  └─ runner.learn(max_iterations)
     每轮迭代：
       1. 收集 24 步 rollout:
          - alg.act(obs, priv_obs, history)  [teacher 或 student 编码器]
          - env.step(actions)                [力矩计算、仿真、reward、新 obs]
          - alg.process_env_step(rewards, dones, infos)
       2. alg.compute_returns()
       3. alg.update():
          - Phase 1: PPO clipped surrogate + value loss + entropy (optimizer1)
          - Phase 2: Student 蒸馏 MSE loss (optimizer2)
       4. 日志记录 + 定期保存模型
```

---

## 2. 训练入口

**`legged_gym/scripts/train.py`** (20 行)

```
L12: env, env_cfg = task_registry.make_env(name=args.task, args=args)
L13: runner, train_cfg = task_registry.make_alg_runner(env, name=args.task, args=args)
L14: env.common_step_counter = runner.current_learning_iteration * env.num_steps_per_env
L15: env.update_reward_curriculum(force_update=True)  # 同步 reward curriculum
L16: runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
```

**`legged_gym/utils/task_registry.py`** (129 行)

- `make_env()` (L36-73): 按名称查找 env 类，应用 CLI 参数覆盖，创建仿真环境
- `make_alg_runner()` (L75-126): 创建日志目录 `logs/<experiment_name>/<date>_<run_name>`，实例化 Runner，可选从 checkpoint 恢复

---

## 3. CTS 算法

**`rsl_rl/rsl_rl/algorithms/cts.py`** (286 行)

### 3.1 核心思想

- 75% 环境 (teacher_env_ratio=0.75) 使用 teacher 编码器，25% 使用 student 编码器
- 两个编码器输出 latent(32)，共享同一个 actor 和 critic
- 每轮 update 分两阶段：先做 PPO 更新 (optimizer1)，再做 student 蒸馏 (optimizer2)

### 3.2 关键方法

| 方法 | 行号 | 功能 |
|------|------|------|
| `__init__` | L41-101 | 初始化双编码器、双优化器，按交错索引划分 teacher/student 环境 |
| `act()` | L112-142 | 分别计算 teacher/student 的动作，合并后重排回原始环境顺序 |
| `process_env_step()` | L144-157 | 重排 reward/done 到 teacher-first 格式，处理 time-out bootstrap |
| `compute_returns()` | L159-165 | 分别计算 teacher/student 的 last values |
| `update()` | L167-286 | **Phase 1** (L176-257): PPO 更新 teacher_encoder + actor + critic; **Phase 2** (L259-277): MSE 蒸馏 student_encoder → teacher_encoder |

### 3.3 环境划分逻辑 (L93-97)

```python
teacher_num_envs = max(int(num_envs * 0.75), 1)
student_num_envs = num_envs - teacher_num_envs
# 交错索引: i % 4 != 0 → teacher, i % 4 == 0 → student
```

### 3.4 蒸馏损失 (L266-275)

```python
student_latent = student_encoder(history_batch[teacher_samples:])    # student 部分
teacher_latent = teacher_encoder(privileged_obs_batch[teacher_samples:])  # teacher 部分 (no_grad)
latent_loss = (teacher_latent - student_latent).pow(2).mean()        # MSE
optimizer2.step()  # 只更新 student_encoder
```

### 3.5 推理

**`on_policy_runner_cts.py` L369-373**: `get_inference_policy()` 返回 `model.act_inference`，只使用 student 编码器 + 历史观测。

---

## 4. CTS Runner

**`rsl_rl/rsl_rl/runners/on_policy_runner_cts.py`** (373 行)

| 方法 | 行号 | 功能 |
|------|------|------|
| `__init__` | L65-121 | 创建 ActorCriticCTS 模型、CTS 算法、RolloutStorage、history buffer `(num_envs, history_length, num_obs)` |
| `learn()` | L123-202 | 主训练循环：rollout → compute_returns → update → log → save |
| `log()` | L204-285 | TensorBoard 日志：value/surrogate/entropy/latent loss，分 teacher/student 统计 reward 和 episode length |
| `save()`/`load()` | L287-367 | 保存 model、optimizer1、optimizer2、iteration、infos |

**History 管理** (L98, L133, L155-156):

```python
self.history = torch.zeros((num_envs, history_length, num_obs))  # 初始化
# 每步: 滑动窗口 + 新 obs
self.history = cat([history[:, 1:], obs.unsqueeze(1)], dim=1)
# done 时: 重置 history
self.history[dones > 0] = 0.0
```

---

## 5. 观测 (Observations)

### 5.1 Go2/dog_v2 观测结构

**`legged_gym/envs/go2/go2_env.py` L23-53** `compute_observations()`

#### Actor 观测 (45 维)

| 索引 | 维度 | 内容 | 缩放 |
|------|------|------|------|
| 0:3 | 3 | `base_ang_vel` | ×0.25 |
| 3:6 | 3 | `projected_gravity` | 1.0 |
| 6:9 | 3 | `commands[:, :3]` (vx, vy, yaw) | commands_scale |
| 9:21 | 12 | `dof_pos - default_dof_pos` | 1.0 |
| 21:33 | 12 | `dof_vel` | ×0.05 |
| 33:45 | 12 | `actions` (上一步动作) | 1.0 |

#### Critic 特权观测 (263 维)

| 索引 | 维度 | 内容 | 说明 |
|------|------|------|------|
| 0:3 | 3 | `base_lin_vel` | Actor 看不到！ |
| 3:48 | 45 | 同 Actor 观测 | ang_vel, gravity, commands, dof_pos, dof_vel, actions |
| 48:52 | 4 | 足端接触力 (norm × 1e-3) | |
| 52:64 | 12 | 关节力矩 / torque_limits | |
| 64:76 | 12 | 关节加速度 `(last_dof_vel - dof_vel)/dt × 1e-4` | |
| 76:263 | 187 | 身高扫描 (相对高度测量) | `clip(root_z - 0.5 - heights, -1, 1)` |

### 5.2 观测噪声

**`legged_gym/envs/go2/go2_env.py` L9-21** `_get_noise_scale_vec()`

| 观测 | 噪声比例 |
|------|----------|
| ang_vel | `0.2 × noise_level × 0.25` |
| gravity | `0.05 × noise_level` |
| commands | 0 (无噪声) |
| dof_pos | `0.01 × noise_level` |
| dof_vel | `1.5 × noise_level × 0.05` |
| actions | 0 (无噪声) |

噪声配置: `go2_config.py` L207 `add_noise=True`, `noise_level=1.0`

噪声实现: `go2_env.py` L52-53 `(2 * rand - 1) * noise_scale_vec`

### 5.3 关键源码位置

| 内容 | 文件 | 行号 |
|------|------|------|
| Actor obs 定义 | `legged_gym/envs/go2/go2_env.py` | L23-53 |
| 噪声向量 | `legged_gym/envs/go2/go2_env.py` | L9-21 |
| 观测维度配置 | `legged_gym/envs/go2/go2_config.py` | L32-38 |
| 噪声参数 | `legged_gym/envs/base/legged_robot_config.py` | L226-232 |

---

## 6. 奖励 (Rewards)

### 6.1 奖励机制

**`legged_gym/envs/base/legged_robot.py`**

1. `_prepare_reward_function()` (L897-928): 遍历 `reward_scales` 字典，对非零项查找对应的 `_reward_<name>` 方法
2. `compute_reward()` (L249-276): `reward = raw_reward × scale × curriculum_multiplier`，可选 `only_positive_rewards` 裁剪

### 6.2 Go2/dog_v2 奖励项及权重

**`legged_gym/envs/go2/go2_config.py` L177-205**

| 奖励名 | 权重 | 类型 | 源码位置 (legged_robot.py) |
|--------|------|------|---------------------------|
| `tracking_lin_vel` | 1.0 | 追踪 | L1310 高斯 `exp(-err²/σ²)` |
| `tracking_ang_vel` | 0.5 | 追踪 | L1324 高斯 `exp(-err²/σ²)` |
| `lin_vel_z` | -2.0 | 惩罚 | L1216 z 轴速度² |
| `ang_vel_xy` | -0.05 | 惩罚 | L1220 xy 角速度² |
| `dof_acc` | -2.5e-7 | 惩罚 | L1257 关节加速度² |
| `dof_power` | -2e-5 | 惩罚 | L1366 `abs(torque × vel)` |
| `torques` | -1e-4 | 惩罚 | L1249 力矩² |
| `correct_base_height` | -1.0 | 惩罚 | L1384 基座高度偏差 (用 height scan) |
| `action_rate` | -0.01 | 惩罚 | L1261 动作变化率² |
| `action_smoothness` | -0.01 | 惩罚 | L1361 jerk: `a_t - 2a_{t-1} + a_{t-2}` |
| `collision` | -1.0 | 惩罚 | L1265 大腿/小腿碰撞 |
| `dof_pos_limits` | -2.0 | 惩罚 | L1273 超出关节软限位 |
| `feet_regulation` | -0.05 | 惩罚 | L1389 足端近地时 xy 速度 |
| `hip_to_default` | -0.05 | 惩罚 | `go2_env.py` L55 髋关节偏离默认 |

### 6.3 奖励课程 (Reward Curriculum)

**`legged_gym/envs/base/legged_robot.py` L148-172** `update_reward_curriculum()`

**Go2 配置** (`go2_config.py` L160-165):

| 奖励 | 迭代区间 | 权重变化 | 效果 |
|------|----------|----------|------|
| `lin_vel_z` | 0 → 1500 | 1.0 → 0.0 | z 速度惩罚逐渐消退 |
| `correct_base_height` | 0 → 5000 | 1.0 → 10.0 | 高度约束逐渐增强 |

线性插值: `scale = (1 - t) × start + t × end`，`t = clip((iter - start) / (end - start), 0, 1)`

### 6.4 动态 Sigma

**`legged_gym/envs/base/legged_robot.py` L1288-1308** `_get_dynamic_sigma()`
**配置**: `go2_config.py` L166-174

根据**指令速度大小**和**地形等级**动态调整 tracking reward 的 sigma (宽容度)。

#### 公式

```
final_sigma = default_sigma + level_scale × (velocity_sigma - default_sigma)
```

**速度维度** (`default_sigma = 0.25`):

| 条件 | velocity_sigma |
|------|----------------|
| `|vel| < min_vel` | `default_sigma` (0.25) |
| `min_vel ≤ |vel| < max_vel` | 线性插值: `default + ratio × (max_sigma - default)` |
| `|vel| ≥ max_vel` | `max_sigma[terrain_type]` |

**地形维度** (L1306):

```python
level_scale = clamp(exp((level + 1) / 10) - 1, max=1.0)
```

| level | level_scale |
|-------|-------------|
| 0 | 0.10 |
| 2 | 0.22 |
| 5 | 0.45 |
| 9 | 1.00 |

**max_sigma 按地形类型** (`go2_config.py` L173):

| 地形 | max_sigma | 含义 |
|------|-----------|------|
| wave | 5/12 ≈ 0.42 | 较宽容 |
| slope / rough_slope | 0.25 | 不变 (同 default) |
| stairs_up / stairs_down | 0.50 | 宽容 |
| obstacles | 0.75 | 很宽容 |
| flat | 0.25 | 严格 |

**效果**: 速度越快、地形越难 → sigma 越大 → `exp(-err²/sigma)` 对误差越宽容 → 机器人不必精确追踪高速指令，专注于不摔倒。

**示例**: level=5 的 stairs_up，`|vel_x|=1.5`:
```
velocity_sigma = 0.25 + 1.0 × (0.50 - 0.25) = 0.50  (达到 max_vel)
level_scale = exp(6/10) - 1 = 0.45
final_sigma = 0.25 + 0.45 × (0.50 - 0.25) = 0.36
```

---

## 7. 域随机化 (Domain Randomization)

### 7.1 DR 参数一览

**`legged_gym/envs/go2/go2_config.py` L41-76**

| 参数 | 范围 | 时机 | 说明 |
|------|------|------|------|
| `randomize_friction` | [0.0, 2.0] | 环境创建 | 64 级摩擦力桶 |
| `randomize_base_mass` | [-1., 1.] kg | 环境创建 | 基座质量偏移 |
| `randomize_link_mass` | [0.9, 1.1] | 环境创建 | 连杆质量乘数 |
| `randomize_base_com` | [-0.03, 0.03] m | 环境创建 | 质心偏移 |
| `randomize_restitution` | [0.0, 0.5] | 环境创建 | 恢复系数 |
| `randomize_pd_gains` | Kp/Kd [0.9, 1.1] | 环境重置 | PD 增益乘数 |
| `randomize_motor_zero_offset` | [-0.035, 0.035] rad | 环境重置 | 电机零点偏移 |
| `randomize_motor_strength` | [0.8, 1.2] | 环境重置 | 电机力矩强度 |
| `push_robots` | xy_vel≤0.4, ang_vel≤0.6 | 每 4s | 外力推扰 |
| `randomize_action_delay` | 0~20ms (4 decimation) | 每步 | 动作延迟随机化 |

### 7.2 DR 实现位置

**`legged_gym/envs/base/legged_robot.py`**

| 功能 | 行号 | 方法 |
|------|------|------|
| 摩擦/恢复系数 | L322-350 | `_process_rigid_shape_props()` |
| 质量/质心 | L381-404 | `_process_rigid_body_props()` |
| 关节限位 | L352-379 | `_process_dof_props()` |
| 电机强度/零点/PD增益 | L197-210 | `reset_idx()` |
| 动作延迟 | L71-78 | `step()` |
| 电机强度缩放力矩 | L80-81 | `step()` |
| 外力推扰 | L710-725 | `_push_robots()` |
| 力矩计算 (含 DR) | L595-619 | `_compute_torques()` |

### 7.3 力矩计算中的 DR (L609-612)

```python
p_gains = self.p_gains * self.p_gains_multiplier[env_ids]
d_gains = self.d_gains * self.d_gains_multiplier[env_ids]
torques = p_gains * (action_scaled + default_pos - dof_pos + motor_zero_offsets) - d_gains * dof_vel
torques *= self.motor_strength[env_ids]  # 电机强度
```

---

## 8. 课程学习 (Curriculum)

### 8.1 地形课程 (Terrain Curriculum)

**`legged_gym/envs/base/legged_robot.py` L1131-1157** `_update_terrain_curriculum()`

- **升级条件**: `max_move_distance > terrain_length / 2` (机器人走得足够远)
- **降级条件**: `actual_distance < expected_distance` (来自累积指令或绝对距离)
- 每次升/降 1 级，最低为 0

**配置** (`go2_config.py` L87-97):
- `max_init_terrain_level = 5`
- `curriculum = True`
- 地形比例: `[0.05, 0.20, 0.05, 0.25, 0.10, 0.20, 0.0, 0.0, 0.15]`
  (wave, slope, rough_slope, stairs_up, stairs_down, obstacles, stepping_stones, gap, flat)

### 8.2 指令采样机制 (Command Sampling)

**核心方法**: `_resample_commands()` (`legged_robot.py` L413-593)
**配置**: `go2_config.py` L99-145
**辅助函数**: `sample_disjoint_intervals()` (`isaacgym_utils.py` L32-47)

#### 8.2.1 动态采样 (`dynamic_resample_commands=True`)

开启后，速度采样使用 `sample_disjoint_intervals`，**排除接近零的速度**：

```
采样区间 = [cfg_min, -lower_bound] ∪ [lower_bound, cfg_max]
```

`lower_bound` (L437-443) 计算为"剩余需要走的距离 / 剩余 episode 时间"，保证机器人始终有足够的前进速度到达地形边界。

```python
remaining_dist = clip(0.625 * terrain_length - norm(xy_accumulation) * resampling_time, 0)
vel_low_bound = clip(remaining_dist / ((max_ep_len - ep_len) * dt), 0)
```

**源码**: `isaacgym_utils.py` L32-47

```python
def sample_disjoint_intervals(env_ids, limit_bound, cfg_min, cfg_max, device):
    """从 [cfg_min, -limit_bound] ∪ [limit_bound, cfg_max] 采样"""
    width_neg = relu(-limit_bound - cfg_min)
    width_pos = relu(cfg_max - limit_bound)
    total_width = width_neg + width_pos
    u = rand * total_width
    samples = where(u < width_neg, cfg_min + u, cfg_max - width_pos + (u - width_neg))
```

#### 8.2.2 极限速度机制 (`limit_vel_prob=0.2`)

**配置**: `go2_config.py` L106-109

20% 概率将速度设为**边界值** (最大/最小/零)，强制直线行走：

```python
limit_vel = {"lin_vel_x": [-1, 1], "lin_vel_y": [-1, 1], "ang_vel_yaw": [-1, 0, 1]}
# 含义: -1=取最小, 0=取零, 1=取最大
```

- `limit_vel_invert_when_continuous=True`: 上次是极限速度则反向 (L508-514)
- 角速度被设为 -1 或 +1，即**最大正/负角速度** → 强制转向但方向不固定

**源码**: `legged_robot.py` L502-543

#### 8.2.3 累积位移计算 (`commands_xy_accumulation`)

**初始化**: `legged_robot.py` L803 `zeros(num_envs, 2)`

每次采样新指令时累加 (L582):

```python
self.commands_xy_accumulation[env_ids] += self.commands[env_ids, :2]
```

这记录的是**整个 episode 内所有指令速度的累积和** (乘以 resampling_time 即近似总位移)。地形课程用它判断降级 (L1146):

```python
move_down = (distance < norm(xy_accumulation) * resampling_time * (1 - zero_proba) * 0.5)
```

含义：如果实际移动距离 < 累积指令位移的一半 → 表现不好 → 地形降级。

#### 8.2.4 为什么不用 heading 模式也行

当前配置 `heading_command=False`，直接采样 `ang_vel_yaw`。不绕圈的原因：

1. **每 5 秒重新采样** (`resampling_time=5.0`)，角速度正负随机
2. **动态采样排除低速**，保证有前进分量
3. **limit_vel_prob=0.2** 强制极限速度且反向翻转
4. **零指令概率 10%** 让机器人学会静止
5. **累积位移判据** 不依赖绝对位移方向，即使转向也能累积

heading 模式 (`heading_command=True`) 的好处是角速度由目标朝向误差计算，更精确地控制方向，但非 heading 模式通过上述随机+课程机制同样能学会在地形上行走。

### 8.3 指令范围课程 (Command Range Curriculum)

**`legged_gym/envs/base/legged_robot.py` L423-436** (在 `_resample_commands()` 中)

**Go2 配置** (`go2_config.py` L111-123):

| 迭代 | lin_vel_x | lin_vel_y | ang_vel_yaw |
|------|-----------|-----------|-------------|
| 初始 | [-0.5, 1.0] | [-0.5, 0.5] | [-0.5, 0.5] |
| 2000 | [-1.0, 2.0] | [-1.0, 1.0] | [-1.5, 1.5] |
| 5000 | [-2.0, 2.5] | [-1.5, 1.5] | [-2.0, 2.0] |

### 8.4 零指令课程 (Zero Command Curriculum)

**Go2 配置** (`go2_config.py` L104):

```python
zero_command_curriculum = {start_iter: 0, end_iter: 1500, start_value: 0.0, end_value: 0.1}
```

前 1500 次迭代中，零速度指令概率从 0% 线性增长到 10%，让机器人学会静止。
额外有 `limit_ang_vel_at_zero_command_prob=0.2` (L560-572)，零速度时 20% 概率给最大角速度，让机器人学会原地转向。

### 8.5 奖励课程 (见 §6.3)

---

## 9. dog_v2 与 Go2 的差异

| 项目 | Go2 | dog_v2 | 源码 |
|------|-----|--------|------|
| URDF | `go2.urdf` | `dog_v2_2_4.urdf` | `dog_v2_config.py` L30 |
| 默认关节角 | 非零 (Go2 风格) | 全部 0.0 | `dog_v2_config.py` L13-27 |
| 基座高度目标 | 0.38 m | 0.34 m | `dog_v2_config.py` L37 |
| 实验名称 | `go2_cts` | `dog_v2_cts` | `dog_v2_config.py` L42 |
| Env 类 | `Go2Robot` | `DogV2Robot(Go2Robot)` — 空继承 | `dog_v2_env.py` |

其余所有配置 (奖励权重、DR、观测、地形、控制参数、课程) 完全继承 Go2。
