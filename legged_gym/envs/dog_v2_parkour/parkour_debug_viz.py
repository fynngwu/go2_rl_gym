import numpy as np

from isaacgym import gymapi, gymutil


def draw_goal_marker(env):
    base_sphere = gymutil.WireframeSphereGeometry(0.08, 24, 24, None, color=(1.0, 0.0, 0.0))
    cur_sphere = gymutil.WireframeSphereGeometry(0.12, 32, 32, None, color=(0.0, 0.25, 1.0))
    next_sphere = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(0.0, 1.0, 0.35))
    for goals in env.terrain.goals.reshape(-1, env.cfg.terrain.num_goals, 3):
        for goal in goals:
            draw_one_goal(env, goal, base_sphere)
    for env_id in range(env.num_envs):
        cur_idx = int(env.cur_goal_idx[env_id].detach().cpu().item())
        cur_idx = min(cur_idx, env.cfg.terrain.num_goals - 1)
        next_idx = min(cur_idx + 1, env.cfg.terrain.num_goals - 1)
        goals = env.env_goals[env_id].detach().cpu().numpy()
        draw_one_goal(env, goals[cur_idx], cur_sphere)
        draw_one_goal(env, goals[next_idx], next_sphere)


def draw_one_goal(env, goal, sphere):
    goal_xy = goal[:2] + env.terrain.cfg.border_size
    pts = (goal_xy / env.terrain.cfg.horizontal_scale).astype(np.int64)
    pts[0] = np.clip(pts[0], 0, env.height_samples.shape[0] - 1)
    pts[1] = np.clip(pts[1], 0, env.height_samples.shape[1] - 1)
    goal_z = env.height_samples[pts[0], pts[1]].detach().cpu().item() * env.terrain.cfg.vertical_scale
    pose = gymapi.Transform(gymapi.Vec3(goal[0], goal[1], goal_z + 0.08), r=None)
    gymutil.draw_lines(sphere, env.gym, env.viewer, env.envs[0], pose)


def draw_edge_mask(env):
    mask = env.terrain.x_edge_mask
    rows = np.where(mask.any(axis=1))[0]
    stride = max(1, int(getattr(env.cfg.terrain, "edge_mask_draw_stride", 1)))
    vertices, colors = [], []
    for row in range(env.cfg.terrain.num_rows):
        x0 = row * env.cfg.terrain.terrain_length
        x1 = x0 + env.cfg.terrain.terrain_length
        y0, y1, z = 0.0, env.cfg.terrain.terrain_width, 0.06
        vertices.extend([[x0, y0, z], [x1, y0, z], [x1, y0, z], [x1, y1, z]])
        vertices.extend([[x1, y1, z], [x0, y1, z], [x0, y1, z], [x0, y0, z]])
        colors.extend([[0.0, 0.9, 1.0]] * 4)
    for x_id in rows[::stride]:
        y_ids = np.where(mask[x_id])[0]
        x = x_id * env.terrain.cfg.horizontal_scale - env.terrain.cfg.border_size
        for seg in np.split(y_ids, np.where(np.diff(y_ids) > 1)[0] + 1):
            if len(seg) < 2:
                continue
            y0 = seg[0] * env.terrain.cfg.horizontal_scale - env.terrain.cfg.border_size
            y1 = seg[-1] * env.terrain.cfg.horizontal_scale - env.terrain.cfg.border_size
            z = env.height_samples[x_id, seg].max().detach().cpu().item() * env.terrain.cfg.vertical_scale + 0.04
            vertices.extend([[x, y0, z], [x, y1, z]])
            colors.append([1.0, 0.0, 1.0])
    if vertices:
        env.gym.add_lines(
            env.viewer,
            env.envs[0],
            len(colors),
            np.asarray(vertices, dtype=np.float32),
            np.asarray(colors, dtype=np.float32),
        )
