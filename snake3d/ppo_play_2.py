import os
import random
import numpy as np
import torch
import torch.nn as nn
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


# ======================================================================
#                     渲染用的立方体工具（和你原来一样）
# ======================================================================

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


# ======================================================================
#                           环境（和训练版一致）
# ======================================================================

class Snake3DGame:
    def __init__(self, grid_size=8, grow_per_fruit=1, max_steps=2000):
        self.grid_size = grid_size
        self.grow_per_fruit = grow_per_fruit
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.snake = [[0, 0, 0], [2, 0, 0], [4, 0, 0]]
        self.direction = [1, 0, 0]
        self.score = 0
        self.steps = 0
        self.growth = 0
        self.game_over = False
        self._spawn()
        return self._obs()

    def _spawn(self):
        while True:
            x = random.randint(0, self.grid_size-1) * 2
            y = random.randint(0, self.grid_size-1) * 2
            z = random.randint(0, self.grid_size-1) * 2
            if [x, y, z] not in self.snake:
                self.fruit = [x, y, z]
                return

    def _wrap(self, v):
        m = self.grid_size * 2
        if v >= m:
            return 0
        elif v < 0:
            return (self.grid_size - 1) * 2
        return v

    def _is_reverse(self, nd):
        dx, dy, dz = self.direction
        ndx, ndy, ndz = nd
        return ndx == -dx and ndy == -dy and ndz == -dz

    def _dist(self, p):
        x, y, z = p
        fx, fy, fz = self.fruit
        return abs(x - fx) + abs(y - fy) + abs(z - fz)

    def _get_occupancy_patch(self):
        hx, hy, hz = self.snake[-1]
        patch = []
        body_set = set(tuple(seg) for seg in self.snake[:-1])

        for dz in [-2, 0, 2]:
            for dy in [-2, 0, 2]:
                for dx in [-2, 0, 2]:
                    x = self._wrap(hx + dx)
                    y = self._wrap(hy + dy)
                    z = self._wrap(hz + dz)
                    val = 0.0
                    if [x, y, z] == self.fruit:
                        val = 1.0
                    elif (x, y, z) in body_set:
                        val = -1.0
                    patch.append(val)
        return np.array(patch, dtype=np.float32)

    def _fruit_dir_sign(self):
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
        occ = self._get_occupancy_patch()          # 27
        fruit_dir = self._fruit_dir_sign()         # 3
        dx, dy, dz = self.direction                # 3
        dir_vec = np.array([dx, dy, dz], dtype=np.float32)
        obs = np.concatenate([occ, fruit_dir, dir_vec], axis=0)
        return obs

    def step(self, action):
        reward = -0.002

        actions = {
            0: [1, 0, 0],
            1: [-1, 0, 0],
            2: [0, 1, 0],
            3: [0, -1, 0],
            4: [0, 0, 1],
            5: [0, 0, -1],
        }
        nd = actions.get(action, self.direction)

        if not self._is_reverse(nd):
            self.direction = nd

        x, y, z = self.snake[-1]
        dx, dy, dz = self.direction
        nx = self._wrap(x + dx * 2)
        ny = self._wrap(y + dy * 2)
        nz = self._wrap(z + dz * 2)
        head = [nx, ny, nz]

        eating = (head == self.fruit)
        will_pop = (self.growth == 0 and not eating)
        body = self.snake[1:] if will_pop else self.snake

        done = False

        if head in body:
            reward = -2.0 - 0.02 * len(self.snake)
            done = True
        else:
            old_dist = self._dist(self.snake[-1])
            self.snake.append(head)

            if eating:
                reward += 1.0 + 0.03 * len(self.snake)
                self.growth += self.grow_per_fruit
                self.score += 1
                self._spawn()
            else:
                new_dist = self._dist(head)
                reward += (old_dist - new_dist) * 0.01

                if self.growth > 0:
                    self.growth -= 1
                else:
                    self.snake.pop(0)

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        return self._obs(), reward, done, {
            "score": self.score,
            "length": len(self.snake),
        }


# ======================================================================
#                           PPO MODEL（33维）
# ======================================================================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=33, act_dim=6):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.policy = nn.Linear(128, act_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)


def load_model(model_path, device="cpu"):
    model = ActorCritic(obs_dim=33, act_dim=6).to(device)
    # 这里保持 weights_only=False 就行，只是个 warning
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[loaded model] {model_path}")
    return model


# ======================================================================
#                           主循环：用 PPO 玩游戏
# ======================================================================

def main():
    pygame.init()

    grid_size = 8
    env = Snake3DGame(grid_size=grid_size)

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
    model = load_model("snake3d_ppo_v2.pt", device=device)

    obs = env.reset()

    running = True
    step_counter = 0

    while running:
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

        # 每一帧让 PPO 走一步（或你可以改成隔几帧走一步）
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(obs_t)
            action = torch.argmax(logits, dim=-1).item()

        obs, reward, done, info = env.step(action)
        step_counter += 1

        if done:
            # 重置一局
            pygame.display.set_caption(
                f"3D Snake PPO - Episode End | Score={info['score']} Len={info['length']}"
            )
            obs = env.reset()

        else:
            pygame.display.set_caption(
                f"3D Snake PPO - Score={info['score']} Len={info['length']}"
            )

        # ----------- OpenGL 渲染 -----------
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 网格
        for gx in range(grid_size):
            for gy in range(grid_size):
                for gz in range(grid_size):
                    cube_wire(2 * gx, 2 * gy, 2 * gz)

        # 果子
        fx, fy, fz = env.fruit
        cube_solid(fx, fy, fz, 6)

        # 蛇
        for seg in env.snake:
            cube_solid(seg[0], seg[1], seg[2], 1)

        pygame.display.flip()
        clock.tick(30)   # 控制可视化速度，30 FPS 左右

    pygame.quit()


if __name__ == "__main__":
    main()