import os
import random
import numpy as np
import torch
import torch.nn as nn
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


# ============================================================
#                  3D Snake 逻辑环境 (和训练版一致)
# ============================================================

class Snake3DGame:
    def __init__(self, grid_size=8, grow_per_fruit=1, max_steps=600):
        self.grid_size = grid_size
        self.grow_per_fruit = grow_per_fruit
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        # 和你原来的初始形状一样
        self.snake = [[0, 0, 0], [2, 0, 0], [4, 0, 0]]
        self.direction = [1, 0, 0]   # 初始 +X
        self.score = 0
        self.steps = 0
        self.growth = 0
        self.game_over = False
        self._spawn()
        return self._obs()

    def _spawn(self):
        while True:
            x = random.randint(0, self.grid_size - 1) * 2
            y = random.randint(0, self.grid_size - 1) * 2
            z = random.randint(0, self.grid_size - 1) * 2
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

    def _norm(self, v):
        maxv = (self.grid_size - 1) * 2
        if maxv == 0:
            return 0.0
        return (v / maxv) * 2.0 - 1.0

    def _obs(self):
        hx, hy, hz = self.snake[-1]
        fx, fy, fz = self.fruit
        dx, dy, dz = self.direction
        return np.array(
            [
                self._norm(hx),
                self._norm(hy),
                self._norm(hz),
                self._norm(fx),
                self._norm(fy),
                self._norm(fz),
                dx,
                dy,
                dz,
            ],
            dtype=np.float32,
        )

    def _dist(self, p):
        x, y, z = p
        fx, fy, fz = self.fruit
        return abs(x - fx) + abs(y - fy) + abs(z - fz)

    def step(self, action):
        """
        action: 0..5
            0: +X (1,0,0)
            1: -X (-1,0,0)
            2: +Y (0,1,0)
            3: -Y (0,-1,0)
            4: +Z (0,0,1)
            5: -Z (0,0,-1)
        """
        actions = {
            0: [1, 0, 0],
            1: [-1, 0, 0],
            2: [0, 1, 0],
            3: [0, -1, 0],
            4: [0, 0, 1],
            5: [0, 0, -1],
        }
        nd = actions.get(action, self.direction)

        # 禁止直接 180° 掉头
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

        if will_pop:
            body = self.snake[1:]
        else:
            body = self.snake

        reward = 0.0
        done = False

        # 撞到身体
        if head in body:
            reward = -1.0
            done = True
            self.game_over = True
        else:
            old_dist = self._dist(self.snake[-1])
            self.snake.append(head)

            if eating:
                reward += 1.0
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

        return self._obs(), reward, done, {"score": self.score, "length": len(self.snake)}


# ============================================================
#                   PPO Actor-Critic 模型
# ============================================================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim=9, act_dim=6):
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


def load_model(model_path="snake3d_ppo_v1.pt", device="cpu"):
    model = ActorCritic().to(device)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[loaded model] {model_path}")
    else:
        raise FileNotFoundError(
            f"找不到模型文件 {model_path}，请先用训练脚本生成 snake3d_ppo_v1.pt"
        )
    model.eval()
    return model


# ============================================================
#                 OpenGL 绘制函数（3D 立方体）
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


# ============================================================
#                主程序：PPO 控制 pygame+OpenGL
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("snake3d_ppo_v1.pt", device=device)

    env = Snake3DGame(grid_size=8, grow_per_fruit=1, max_steps=600)
    obs = env.reset()

    pygame.init()
    GridSize = env.grid_size

    SPEED_MS = 200   # 控制蛇移动速度（毫秒）

    display = (800, 800)
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(60, display[0] / display[1], 0.1, 200.0)

    camera_distance = GridSize * 4 + 10
    glTranslatef(-GridSize, -GridSize, -camera_distance)
    glEnable(GL_DEPTH_TEST)

    clock = pygame.time.Clock()
    running = True

    MOVEEVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(MOVEEVENT, SPEED_MS)

    dragging = False
    last_mouse_pos = (0, 0)

    def update_caption(score, length):
        pygame.display.set_caption(f"3D Snake PPO - Score: {score}  Length: {length}")

    update_caption(env.score, len(env.snake))

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

            if event.type == MOVEEVENT:
                # 用 PPO 模型选择动作
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    logits, value = model(obs_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    # 你可以用 sample() 更随机，也可以 argmax 贪婪
                    action = torch.argmax(dist.probs, dim=-1).item()
                    # action = dist.sample().item()

                next_obs, reward, done, info = env.step(action)
                obs = next_obs

                update_caption(info["score"], info["length"])

                if done:
                    # 简单处理：重置一局继续
                    obs = env.reset()
                    update_caption(env.score, len(env.snake))

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 绘制网格
        for gx in range(GridSize):
            for gy in range(GridSize):
                for gz in range(GridSize):
                    cube_wire(2 * gx, 2 * gy, 2 * gz)

        # 果子
        fx, fy, fz = env.fruit
        cube_solid(fx, fy, fz, 6)

        # 蛇
        for seg in env.snake:
            cube_solid(seg[0], seg[1], seg[2], 1)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()