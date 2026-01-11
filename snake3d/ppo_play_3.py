import os
import numpy as np
import torch
import torch.nn as nn
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


# ============================================================
#                Rendering helpers (same style as before)
# ============================================================

edges = (
    (0, 1),
    (2, 1),
    (2, 3),
    (2, 7),
    (6, 3),
    (6, 4),
    (6, 7),
    (5, 1),
    (5, 4),
    (5, 7),
    (0, 3),
    (0, 4),
)

surfaces = (
    (0, 1, 2, 3),
    (3, 2, 7, 6),
    (6, 7, 5, 4),
    (4, 5, 1, 0),
    (1, 5, 7, 2),
    (4, 0, 3, 6),
)

colors = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0, 0, 0),
    (1, 1, 1),
    (0, 1, 1),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0, 0, 0),
    (1, 1, 1),
    (0, 1, 1),
)


def cube_solid(x, y, z, shade):
    """Draw a solid cube with wireframe edges at position (x, y, z)."""
    verticies = (
        (x + 1, y - 1, z - 1),
        (x + 1, y + 1, z - 1),
        (x - 1, y + 1, z - 1),
        (x - 1, y - 1, z - 1),
        (x + 1, y - 1, z + 1),
        (x + 1, y + 1, z + 1),
        (x - 1, y - 1, z + 1),
        (x - 1, y + 1, z + 1),
    )

    glBegin(GL_QUADS)
    for surface in surfaces:
        for vertex in surface:
            glColor3fv(colors[shade])
            glVertex3fv(verticies[vertex])
    glEnd()

    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glColor3fv(colors[10])
            glVertex3fv(verticies[vertex])
    glEnd()


def cube_wire(x, y, z):
    """Draw only wireframe cube at position (x, y, z)."""
    verticies = (
        (x + 1, y - 1, z - 1),
        (x + 1, y + 1, z - 1),
        (x - 1, y + 1, z - 1),
        (x - 1, y - 1, z - 1),
        (x + 1, y - 1, z + 1),
        (x + 1, y + 1, z + 1),
        (x - 1, y - 1, z + 1),
        (x - 1, y + 1, z + 1),
    )

    glBegin(GL_LINES)
    glColor3f(0.8, 0.8, 0.8)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()


# ============================================================
#              CNN training environment (same as train)
# ============================================================

class Snake3DConvEnv:
    """
    3D Snake environment with a full occupancy volume.

    Grid coordinates are integer indices in [0, grid_size-1].
    The snake moves with wrap-around at the borders.

    Observation = flattened 3-channel grid + 6 global features:
        grid channels:
            0: body (excluding head)
            1: head
            2: fruit
        global features:
            [fruit_dir_x, fruit_dir_y, fruit_dir_z, dir_x, dir_y, dir_z]
    """

    def __init__(self, grid_size=8, max_steps_factor=4):
        self.grid_size = grid_size
        self.max_steps_factor = max_steps_factor
        self.max_steps = max_steps_factor * (grid_size ** 3)
        self.reset()

    def reset(self):
        g = self.grid_size
        mid = g // 2
        # Simple initial snake in the center (length 3) pointing +x
        self.snake = [
            [mid - 1, mid, mid],
            [mid,     mid, mid],
            [mid + 1, mid, mid],
        ]
        self.direction = [1, 0, 0]
        self.score = 0
        self.steps = 0
        self.done = False
        self._spawn_fruit()
        return self._obs()

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

    def _wrap(self, v):
        if v >= self.grid_size:
            return 0
        elif v < 0:
            return self.grid_size - 1
        return v

    def _is_reverse(self, nd):
        dx, dy, dz = self.direction
        ndx, ndy, ndz = nd
        return ndx == -dx and ndy == -dy and ndz == -dz

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    def _build_grid_channels(self):
        """Build (3, G, G, G) occupancy grid: body, head, fruit."""
        g = self.grid_size
        grid = np.zeros((3, g, g, g), dtype=np.float32)

        # Body channel
        for x, y, z in self.snake[:-1]:
            grid[0, x, y, z] = 1.0

        # Head channel
        hx, hy, hz = self.snake[-1]
        grid[1, hx, hy, hz] = 1.0

        # Fruit channel
        fx, fy, fz = self.fruit
        grid[2, fx, fy, fz] = 1.0

        return grid

    def _fruit_dir_sign(self):
        """Sign(-1,0,1) of fruit position relative to head."""
        hx, hy, hz = self.snake[-1]
        fx, fy, fz = self.fruit

        def sgn(v):
            if v > 0:
                return 1.0
            elif v < 0:
                return -1.0
            return 0.0

        return np.array([sgn(fx - hx), sgn(fy - hy), sgn(fz - hz)],
                        dtype=np.float32)

    def _obs(self):
        """Flattened grid + global features."""
        grid = self._build_grid_channels().reshape(-1)
        fruit_dir = self._fruit_dir_sign()
        dx, dy, dz = self.direction
        dir_vec = np.array([dx, dy, dz], dtype=np.float32)
        obs = np.concatenate([grid, fruit_dir, dir_vec], axis=0)
        return obs

    def step(self, action):
        """
        Environment dynamics are the same as in training.
        The reward is computed but not actually used in play mode.
        """
        if self.done:
            raise RuntimeError("Call reset() before stepping a finished env.")

        reward = -0.001  # small step penalty

        actions = {
            0: [1, 0, 0],
            1: [-1, 0, 0],
            2: [0, 1, 0],
            3: [0, -1, 0],
            4: [0, 0, 1],
            5: [0, 0, -1],
        }
        nd = actions.get(int(action), self.direction)

        if not self._is_reverse(nd):
            self.direction = nd

        hx, hy, hz = self.snake[-1]
        dx, dy, dz = self.direction
        nx = self._wrap(hx + dx)
        ny = self._wrap(hy + dy)
        nz = self._wrap(hz + dz)
        new_head = [nx, ny, nz]

        eating = (new_head == self.fruit)
        body_to_check = self.snake[1:]

        done = False

        if new_head in body_to_check:
            reward = -2.0 - 0.02 * len(self.snake)
            done = True
        else:
            old_dist = self._manhattan(self.snake[-1], self.fruit)
            self.snake.append(new_head)

            if eating:
                max_len = self.grid_size ** 3
                length_bonus = len(self.snake) / max_len
                reward += 1.0 + 0.5 * length_bonus
                self.score += 1
                self._spawn_fruit()
            else:
                new_dist = self._manhattan(new_head, self.fruit)
                reward += 0.01 * (old_dist - new_dist)
                self.snake.pop(0)

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        self.done = done
        obs = self._obs()
        info = {"score": self.score, "length": len(self.snake)}
        return obs, reward, done, info


# ============================================================
#                  CNN-based Actor-Critic (same as train)
# ============================================================

class ActorCriticCNN(nn.Module):
    """
    CNN-based Actor-Critic for 3D Snake.

    Input x: (batch, obs_dim) = [grid_flat, global_feats]
    Internally:
        - reshape grid_flat -> (B, C, G, G, G)
        - apply 3D CNN + AdaptiveAvgPool3d(1)
        - concat with global_feats
        - feed through MLP -> policy logits and value
    """

    def __init__(self, channels=3, global_dim=6, act_dim=6):
        super().__init__()
        self.channels = channels
        self.global_dim = global_dim

        self.cnn = nn.Sequential(
            nn.Conv3d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(output_size=1),
        )

        self.fc = nn.Sequential(
            nn.Linear(32 + global_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.policy = nn.Linear(128, act_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        B = x.shape[0]
        global_feats = x[:, -self.global_dim:]
        grid_flat = x[:, :-self.global_dim]

        # Recover grid size from the flat part
        num_voxels = grid_flat.shape[1] // self.channels
        g = int(round(num_voxels ** (1.0 / 3.0)))
        grid = grid_flat.view(B, self.channels, g, g, g)

        feat = self.cnn(grid).view(B, -1)
        feat = torch.cat([feat, global_feats], dim=1)
        feat = self.fc(feat)

        logits = self.policy(feat)
        value = self.value(feat)
        return logits, value


def load_model(path, device="cpu"):
    """Load CNN actor-critic weights from disk."""
    model = ActorCriticCNN().to(device)
    # Use weights_only=True as recommended by PyTorch
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"[loaded model] {path}")
    return model


# ============================================================
#                           Main loop
# ============================================================

def main():
    pygame.init()

    grid_size = 8
    env = Snake3DConvEnv(grid_size=grid_size)

    display = (800, 800)
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(60, display[0] / display[1], 0.1, 100.0)
    camera_distance = grid_size * 4 + 10
    glTranslatef(-grid_size, -grid_size, -camera_distance)
    glEnable(GL_DEPTH_TEST)

    dragging = False
    last_mouse_pos = (0, 0)
    clock = pygame.time.Clock()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "snake3d_ppo_v3_cnn.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Train first with snake3d_ppo_cnn_train.py."
        )

    model = load_model(model_path, device=device)
    obs = env.reset()

    running = True
    while running:
        # Handle window / mouse events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            if event.type == MOUSEBUTTONDOWN and event.button == 1:
                dragging = True
                last_mouse_pos = event.pos

            if event.type == MOUSEBUTTONUP and event.button == 1:
                dragging = False

            if event.type == MOUSEMOTION and dragging:
                dx = event.pos[0] - last_mouse_pos[0]
                dy = event.pos[1] - last_mouse_pos[1]
                last_mouse_pos = event.pos
                glMatrixMode(GL_MODELVIEW)
                glRotatef(dx * 0.3, 0, 1, 0)
                glRotatef(dy * 0.3, 1, 0, 0)

        # One PPO action step (greedy policy: argmax over logits)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(obs_t)
            action = torch.argmax(logits, dim=-1).item()

        obs, reward, done, info = env.step(action)

        if done:
            pygame.display.set_caption(
                f"Snake3D CNN PPO - Episode End | Score={info['score']} Len={info['length']}"
            )
            obs = env.reset()
        else:
            pygame.display.set_caption(
                f"Snake3D CNN PPO - Score={info['score']} Len={info['length']}"
            )

        # ------------ OpenGL rendering ------------
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Draw full grid (wireframe)
        for gx in range(grid_size):
            for gy in range(grid_size):
                for gz in range(grid_size):
                    # Multiply by 2 to match cube size in renderer
                    cube_wire(2 * gx, 2 * gy, 2 * gz)

        # Draw fruit
        fx, fy, fz = env.fruit
        cube_solid(2 * fx, 2 * fy, 2 * fz, 6)

        # Draw snake segments
        for sx, sy, sz in env.snake:
            cube_solid(2 * sx, 2 * sy, 2 * sz, 1)

        pygame.display.flip()
        clock.tick(30)  # 30 FPS

    pygame.quit()


if __name__ == "__main__":
    main()
