import numpy as np
from collections import defaultdict

from isaacgym import terrain_utils


class ParkourHurdleTerrain:
    def __init__(self, cfg, num_robots):
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", "plane"]:
            return

        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)
        self.border = int(cfg.border_size / cfg.horizontal_scale)
        self.tot_cols = cfg.num_cols * self.width_per_env_pixels + 2 * self.border
        self.tot_rows = cfg.num_rows * self.length_per_env_pixels + 2 * self.border

        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3), dtype=np.float32)
        self.terrain_type = np.zeros((cfg.num_rows, cfg.num_cols), dtype=np.int64)
        self.goals = np.zeros((cfg.num_rows, cfg.num_cols, cfg.num_goals, 3), dtype=np.float32)
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.name2cols = defaultdict(set)
        self.cols2id = []

        self._build_map()
        self.heightsamples = self.height_field_raw

        if cfg.mesh_type != "trimesh":
            raise ValueError("Parkour hurdle terrain only supports trimesh")
        self.vertices, self.triangles, self.x_edge_mask = convert_heightfield_to_trimesh(
            self.height_field_raw,
            cfg.horizontal_scale,
            cfg.vertical_scale,
            cfg.slope_threshold,
        )

    def _build_map(self):
        difficulty_den = max(self.cfg.num_rows - 1, 1)
        for col in range(self.cfg.num_cols):
            for row in range(self.cfg.num_rows):
                difficulty = row / difficulty_den
                terrain = terrain_utils.SubTerrain(
                    "parkour_hurdle",
                    width=self.length_per_env_pixels,
                    length=self.width_per_env_pixels,
                    vertical_scale=self.cfg.vertical_scale,
                    horizontal_scale=self.cfg.horizontal_scale,
                )
                self._make_hurdle_track(terrain, difficulty)
                self._add_terrain_to_map(terrain, row, col)
            self.name2cols["parkour_hurdle"].add(col)
            self.cols2id.append(16)

    def _make_hurdle_track(self, terrain, difficulty):
        cfg = self.cfg
        goals = np.zeros((cfg.num_goals, 2), dtype=np.float32)

        mid_y = terrain.length // 2
        platform_len = round(2.5 / terrain.horizontal_scale)
        wall_len = max(1, round(1.0 / terrain.horizontal_scale))
        wall_width = max(1, round(0.05 / terrain.horizontal_scale))
        wall_height_base = (0.15 + 0.25 * difficulty) / terrain.vertical_scale
        first_hurdle_x = round(4.0 / terrain.horizontal_scale)
        hurdle_spacing = round(2.0 / terrain.horizontal_scale)
        goal_after_hurdle = round(0.6 / terrain.horizontal_scale)
        pad_width = int(0.1 / terrain.horizontal_scale)
        pad_height = 0

        terrain.height_field_raw[0:platform_len, :] = 0
        for idx in range(cfg.num_goals):
            dis_x = first_hurdle_x + idx * hurdle_spacing
            x0 = max(dis_x - wall_width // 2, 0)
            x1 = min(x0 + wall_width, terrain.width)
            y0 = max(mid_y - wall_len // 2, 0)
            y1 = min(y0 + wall_len, terrain.length)
            noise = round(np.random.normal(0, 0.03 / terrain.vertical_scale))
            wall_height = max(round(wall_height_base + noise), 0)
            terrain.height_field_raw[x0:x1, y0:y1] = wall_height
            goals[idx] = [min(dis_x + goal_after_hurdle, terrain.width - 1), mid_y]

        # smooth roughness noise on ground
        terrain_utils.random_uniform_terrain(terrain, min_height=-0.02, max_height=0.02, step=0.005, downsampled_scale=0.3)

        terrain.goals = goals * terrain.horizontal_scale

        terrain.height_field_raw[:, :pad_width] = pad_height
        terrain.height_field_raw[:, -pad_width:] = pad_height
        terrain.height_field_raw[:pad_width, :] = pad_height
        terrain.height_field_raw[-pad_width:, :] = pad_height
        terrain.idx = 16

    def _add_terrain_to_map(self, terrain, row, col):
        start_x = self.border + row * self.length_per_env_pixels
        end_x = start_x + self.length_per_env_pixels
        start_y = self.border + col * self.width_per_env_pixels
        end_y = start_y + self.width_per_env_pixels
        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = row * self.env_length + 1.0
        env_origin_y = (col + 0.5) * self.env_width
        self.env_origins[row, col] = [env_origin_x, env_origin_y, 0.0]
        self.terrain_type[row, col] = terrain.idx
        self.goals[row, col, :, :2] = terrain.goals + [row * self.env_length, col * self.env_width]


def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    hf = height_field_raw
    num_rows, num_cols = hf.shape
    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    move_x = np.zeros((num_rows, num_cols), dtype=np.int8)
    if slope_threshold is not None:
        slope_threshold *= horizontal_scale / vertical_scale
        move_y = np.zeros((num_rows, num_cols), dtype=np.int8)
        move_corners = np.zeros((num_rows, num_cols), dtype=np.int8)
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > slope_threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        )
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale

    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for row in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + row * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * row * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3
    return vertices, triangles, move_x != 0
