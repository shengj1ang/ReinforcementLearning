import numpy as np
import random


class Snake3DEnv:
    """
    3D Snake environment with:
    - Normalized head position
    - Normalized fruit position
    - Direction vector to fruit
    - Current movement direction
    - Manhattan distance shaping reward
    - Strong death penalty
    - Strong fruit reward
    """

    def __init__(self, grid_size=8, max_steps_factor=4):
        self.grid_size = grid_size
        self.max_steps = max_steps_factor * (grid_size ** 3)
        self.reset()

    # ---------------------------
    #         RESET
    # ---------------------------
    def reset(self):
        g = self.grid_size
        mid = g // 2

        # Initial snake: length 3 in the center, pointing +x
        self.snake = [
            [mid - 1, mid, mid],
            [mid, mid, mid],
            [mid + 1, mid, mid],
        ]
        self.direction = [1, 0, 0]

        self.score = 0
        self.steps = 0
        self.done = False

        self._spawn_fruit()
        return self._obs()

    # ---------------------------
    #         FRUIT
    # ---------------------------
    def _spawn_fruit(self):
        g = self.grid_size
        occupied = {tuple(p) for p in self.snake}

        while True:
            x = np.random.randint(0, g)
            y = np.random.randint(0, g)
            z = np.random.randint(0, g)

            if (x, y, z) not in occupied:
                self.fruit = [x, y, z]
                return

    # ---------------------------
    #       WRAP-AROUND
    # ---------------------------
    def _wrap(self, v):
        """Wrap coordinate into the valid range [0, grid_size-1]."""
        if v >= self.grid_size:
            return 0
        elif v < 0:
            return self.grid_size - 1
        return v

    # ---------------------------
    #   CHECK REVERSE MOVEMENT
    # ---------------------------
    def _is_reverse(self, new_dir):
        """Disallow 180° turn."""
        dx, dy, dz = self.direction
        ndx, ndy, ndz = new_dir
        return (ndx == -dx and ndy == -dy and ndz == -dz)

    # ---------------------------
    #        NORMALIZATION
    # ---------------------------
    def _norm(self, v):
        """Normalize grid coordinate to [-1, 1]."""
        return (v / (self.grid_size - 1)) * 2 - 1

    def _direction_to_fruit(self):
        """Return normalized direction vector to the fruit."""
        hx, hy, hz = self.snake[-1]
        fx, fy, fz = self.fruit

        # Continuous normalized vector (unit vector)
        vec = np.array([fx - hx, fy - hy, fz - hz], dtype=np.float32)
        norm = np.linalg.norm(vec) + 1e-6
        vec = vec / norm
        return vec  # shape (3,)

    # ---------------------------
    #       OBSERVATION
    # ---------------------------
    def _obs(self):
        """
        Observation vector:
        [head_x_norm, head_y_norm, head_z_norm,
         fruit_x_norm, fruit_y_norm, fruit_z_norm,
         dir_to_fruit_x, dir_to_fruit_y, dir_to_fruit_z,
         move_dx, move_dy, move_dz]
        Total = 12 dims
        """
        hx, hy, hz = self.snake[-1]
        fx, fy, fz = self.fruit
        dx, dy, dz = self.direction
        fruit_dir = self._direction_to_fruit()

        obs = np.array([
            self._norm(hx), self._norm(hy), self._norm(hz),
            self._norm(fx), self._norm(fy), self._norm(fz),
            fruit_dir[0], fruit_dir[1], fruit_dir[2],
            dx, dy, dz
        ], dtype=np.float32)

        return obs

    # ---------------------------
    #      REWARD SHAPING
    # ---------------------------
    def _manhattan(self, a, b):
        """Manhattan distance between 3D points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    # ---------------------------
    #           STEP
    # ---------------------------
    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset() before stepping again.")

        # Strong step penalty to prevent wandering
        reward = -0.01

        # 6 actions = +/- x,y,z
        act_map = {
            0: [1, 0, 0],
            1: [-1, 0, 0],
            2: [0, 1, 0],
            3: [0, -1, 0],
            4: [0, 0, 1],
            5: [0, 0, -1],
        }
        new_dir = act_map[int(action)]

        # Prevent reverse movement
        if not self._is_reverse(new_dir):
            self.direction = new_dir

        # Move head
        hx, hy, hz = self.snake[-1]
        dx, dy, dz = self.direction

        nx = self._wrap(hx + dx)
        ny = self._wrap(hy + dy)
        nz = self._wrap(hz + dz)

        new_head = [nx, ny, nz]

        # Collision check: cannot hit body (except the tail if it moves)
        body_to_check = self.snake[1:]

        # -------------------------
        #      DEATH
        # -------------------------
        if new_head in body_to_check:
            reward = -5.0 - 0.03 * len(self.snake)   # strong penalty
            self.done = True
            return self._obs(), reward, True, {"score": self.score, "length": len(self.snake)}

        # -------------------------
        #   DISTANCE SHAPING
        # -------------------------
        old_dist = self._manhattan([hx, hy, hz], self.fruit)

        # Move snake body
        self.snake.append(new_head)

        # -------------------------
        #        FRUIT
        # -------------------------
        if new_head == self.fruit:
            # Very strong reward to encourage eating
            reward = 2.0 + 0.1 * len(self.snake)
            self.score += 1
            self._spawn_fruit()

        else:
            # Continue normal movement
            new_dist = self._manhattan(new_head, self.fruit)
            reward += 0.2 * (old_dist - new_dist)  # stronger reward
            self.snake.pop(0)  # remove tail

        # -------------------------
        #      MAX STEPS
        # -------------------------
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        return self._obs(), reward, self.done, {
            "score": self.score,
            "length": len(self.snake)
        }
