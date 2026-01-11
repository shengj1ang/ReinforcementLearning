import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ======================================================================
#                           3D SNAKE ENV
# ======================================================================

class Snake3DGame:
    def __init__(self, grid_size=8, grow_per_fruit=1, max_steps=2000):
        self.grid_size = grid_size
        self.grow_per_fruit = grow_per_fruit
        self.max_steps = max_steps
        self.reset()

    # ---------------- 基本工具 ----------------

    def reset(self):
        # 和你原来的初始形状一样（0,2,4）
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

    # ---------------- 观测：3x3x3 occupancy + 辅助特征 ----------------

    def _get_occupancy_patch(self):
        """
        3x3x3 Patch（以蛇头为中心，步长=2）:
        空: 0.0，身体: -1.0，果子: +1.0
        顺序: z(-2,0,2) 里嵌 y(-2,0,2) 里嵌 x(-2,0,2)
        """
        hx, hy, hz = self.snake[-1]
        patch = []

        body_set = set(tuple(seg) for seg in self.snake[:-1])  # 不含头

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
        """
        果子相对方向（简单符号）：-1 / 0 / 1
        不考虑 wrap 的最短路径，先用简单版本即可。
        """
        hx, hy, hz = self.snake[-1]
        fx, fy, fz = self.fruit

        def sgn(v):
            if v > 0:
                return 1.0
            elif v < 0:
                return -1.0
            return 0.0

        return np.array(
            [sgn(fx - hx), sgn(fy - hy), sgn(fz - hz)],
            dtype=np.float32,
        )

    def _obs(self):
        occ = self._get_occupancy_patch()           # 27
        fruit_dir = self._fruit_dir_sign()          # 3
        dx, dy, dz = self.direction                 # 3
        dir_vec = np.array([dx, dy, dz], dtype=np.float32)
        obs = np.concatenate([occ, fruit_dir, dir_vec], axis=0)
        return obs

    # ---------------- step ----------------

    def step(self, action):
        # 每步轻微惩罚，鼓励快去吃果子
        reward = -0.002

        # 6 个方向映射
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

        body = self.snake[1:] if will_pop else self.snake

        done = False

        # 撞到自己：惩罚随长度增加
        if head in body:
            reward = -2.0 - 0.02 * len(self.snake)
            done = True
        else:
            old_dist = self._dist(self.snake[-1])
            self.snake.append(head)

            if eating:
                # 蛇越长，吃果子奖励越高
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

        return self._obs(), reward, done, {"score": self.score, "length": len(self.snake)}


# ======================================================================
#                           PPO MODEL
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


# ======================================================================
#                           PPO TRAINING
# ======================================================================

SAVE_PATH = "snake3d_ppo_v2.pt"
USE_OLD_MODEL = False 


def save_model(model):
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"[saved] {SAVE_PATH}")


def maybe_load_model(model):
    if USE_OLD_MODEL and os.path.exists(SAVE_PATH):
        model.load_state_dict(torch.load(SAVE_PATH))
        print(f"[loaded existing model] {SAVE_PATH}")
    else:
        print("[training from scratch]")


def train_ppo(total_steps=300_000, rollout=2048, lr=1e-4):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = Snake3DGame()
    obs_dim = 33
    model = ActorCritic(obs_dim=obs_dim, act_dim=6).to(device)
    maybe_load_model(model)

    optim_ = optim.Adam(model.parameters(), lr=lr)

    obs = env.reset()

    step_i = 0

    # ---- chache for graph drawing ----
    steps_log = []
    mean_return_log = []
    mean_len_log = []
    policy_loss_log = []
    value_loss_log = []
    entropy_log = []

    #  episode statistics return / length
    episode_returns = []
    episode_lengths = []
    current_ret = 0.0
    current_len = 0

    while step_i < total_steps:

        # ------------ collect rollout ------------
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [],[],[],[],[],[]
        last_info = {"score": 0, "length": 0}

        for _ in range(rollout):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = model(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()

            logp = dist.log_prob(action).item()
            value = value.item()

            next_obs, reward, done, info = env.step(action.item())
            last_info = info

            obs_buf.append(obs)
            act_buf.append(action.item())
            logp_buf.append(logp)
            rew_buf.append(reward)
            val_buf.append(value)
            done_buf.append(float(done))

            # episode Statistics
            current_ret += reward
            current_len += 1
            if done:
                episode_returns.append(current_ret)
                episode_lengths.append(current_len)
                current_ret = 0.0
                current_len = 0
                next_obs = env.reset()

            obs = next_obs
            step_i += 1

            if step_i >= total_steps:
                break

        # ------------ GAE ------------
        obs_t_last = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        _, last_value = model(obs_t_last)
        last_value = last_value.item()

        vals = np.array(val_buf + [last_value], dtype=np.float32)
        rews = np.array(rew_buf, dtype=np.float32)
        dones = np.array(done_buf, dtype=np.float32)

        adv = np.zeros_like(rews)
        gae = 0.0
        gamma = 0.99
        lam = 0.95

        for t in reversed(range(len(rews))):
            delta = rews[t] + gamma * vals[t+1] * (1 - dones[t]) - vals[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            adv[t] = gae
        ret = adv + vals[:-1]

        # ------------ PPO Update ------------
        obs_t     = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        act_t     = torch.tensor(act_buf, dtype=torch.int64, device=device)
        old_logp_t= torch.tensor(logp_buf, dtype=torch.float32, device=device)
        adv_t     = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t     = torch.tensor(ret, dtype=torch.float32, device=device)

        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_pl, total_vl, total_ent = 0.0, 0.0, 0.0
        mb_count = 0

        for _ in range(10):  # epochs
            logits, value = model(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act_t)
            ratio = torch.exp(logp - old_logp_t)

            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1-0.3, 1+0.3) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss  = (ret_t - value.squeeze()).pow(2).mean()
            entropy     = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.02 * entropy

            optim_.zero_grad()
            loss.backward()
            optim_.step()

            total_pl += policy_loss.item()
            total_vl += value_loss.item()
            total_ent += entropy.item()
            mb_count += 1

        avg_pl = total_pl / mb_count
        avg_vl = total_vl / mb_count
        avg_ent = total_ent / mb_count

        if len(episode_returns) > 0:
            mean_ret = float(np.mean(episode_returns[-10:]))
            mean_len = float(np.mean(episode_lengths[-10:]))
        else:
            mean_ret = 0.0
            mean_len = 0.0

        steps_log.append(step_i)
        mean_return_log.append(mean_ret)
        mean_len_log.append(mean_len)
        policy_loss_log.append(avg_pl)
        value_loss_log.append(avg_vl)
        entropy_log.append(avg_ent)

        save_model(model)
        print(
            f"steps={step_i} "
            f" last_score={last_info['score']} length={last_info['length']} "
            f" mean_ret(10ep)={mean_ret:.3f} "
            f" ploss={avg_pl:.4f} vloss={avg_vl:.4f} ent={avg_ent:.4f}"
        )

    # =================== Draw Graph ===================
    plt.figure()
    plt.plot(steps_log, mean_return_log)
    plt.xlabel("steps")
    plt.ylabel("mean episode return (last 10)")
    plt.title("Snake3D PPO - Mean Return")
    plt.grid(True)

    plt.figure()
    plt.plot(steps_log, policy_loss_log, label="policy loss")
    plt.plot(steps_log, value_loss_log, label="value loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.title("Snake3D PPO - Loss")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(steps_log, entropy_log)
    plt.xlabel("steps")
    plt.ylabel("entropy")
    plt.title("Snake3D PPO - Entropy")
    plt.grid(True)

    plt.show()

    return model


# ======================================================================
#                                MAIN
# ======================================================================

if __name__ == "__main__":
    train_ppo(total_steps=3000000)