import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ============================================================
#                 3D Snake environment (CNN-friendly)
# ============================================================

class Snake3DConvEnv:
    """
    3D Snake environment with a full occupancy volume.

    Grid coordinates are integer indices in [0, grid_size-1].
    The snake moves with wrap-around at the borders.
    Observation: flattened 3-channel grid + 6 global features.

    Grid channels:
        channel 0: snake body (excluding head), 1.0 where body occupies, else 0
        channel 1: snake head, 1.0 at head position, else 0
        channel 2: fruit, 1.0 at fruit position, else 0

    Global features (length = 6):
        [fruit_dir_x, fruit_dir_y, fruit_dir_z, dir_x, dir_y, dir_z]
        where fruit_dir_* are sign(-1, 0, 1) of fruit - head in each axis.
    """

    def __init__(self, grid_size=8, max_steps_factor=4):
        self.grid_size = grid_size
        self.max_steps_factor = max_steps_factor
        self.max_steps = max_steps_factor * (grid_size ** 3)
        self.reset()

    def reset(self):
        g = self.grid_size
        mid = g // 2
        # A short snake along +x direction in the middle of the grid
        self.snake = [
            [mid - 1, mid, mid],
            [mid,     mid, mid],
            [mid + 1, mid, mid],
        ]
        self.direction = [1, 0, 0]  # +x as initial direction
        self.score = 0
        self.steps = 0
        self.done = False
        self._spawn_fruit()
        return self._obs()

    def _spawn_fruit(self):
        """Spawn a fruit at a random empty cell."""
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
        """Periodic boundary conditions."""
        if v >= self.grid_size:
            return 0
        elif v < 0:
            return self.grid_size - 1
        return v

    def _is_reverse(self, nd):
        """Check if the new direction is a 180-degree turn."""
        dx, dy, dz = self.direction
        ndx, ndy, ndz = nd
        return ndx == -dx and ndy == -dy and ndz == -dz

    def _manhattan(self, a, b):
        """Manhattan distance in 3D."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    def _build_grid_channels(self):
        """
        Build a (3, G, G, G) float32 occupancy grid:
        body, head, fruit.
        """
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
        """
        Sign of fruit direction relative to the head:
        -1 / 0 / 1 per axis.
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
        """
        Build the observation vector:
        flatten(grid_channels) + fruit_dir_sign + direction.
        """
        grid = self._build_grid_channels().reshape(-1)
        fruit_dir = self._fruit_dir_sign()
        dx, dy, dz = self.direction
        dir_vec = np.array([dx, dy, dz], dtype=np.float32)
        obs = np.concatenate([grid, fruit_dir, dir_vec], axis=0)
        return obs

    def step(self, action):
        """
        One step in the environment.

        Reward shaping:
            - small negative step penalty to encourage efficiency
            - strong negative reward when hitting yourself
            - positive reward for eating fruit (scaled by current length)
            - small shaped reward for moving closer to the fruit
        """
        if self.done:
            raise RuntimeError("Call reset() before stepping a finished env.")

        # Base step penalty
        reward = -0.001

        # 6 discrete actions
        actions = {
            0: [1, 0, 0],
            1: [-1, 0, 0],
            2: [0, 1, 0],
            3: [0, -1, 0],
            4: [0, 0, 1],
            5: [0, 0, -1],
        }
        nd = actions.get(int(action), self.direction)

        # Prevent direct 180-degree reversal
        if not self._is_reverse(nd):
            self.direction = nd

        hx, hy, hz = self.snake[-1]
        dx, dy, dz = self.direction
        nx = self._wrap(hx + dx)
        ny = self._wrap(hy + dy)
        nz = self._wrap(hz + dz)
        new_head = [nx, ny, nz]

        eating = (new_head == self.fruit)

        # For classical snake, when eating we simply do not remove tail in this step.
        # Collision is always checked against the full body (excluding head).
        body_to_check = self.snake[1:]

        done = False

        if new_head in body_to_check:
            # Strong penalty for self-collision (scaled by length)
            reward = -2.0 - 0.02 * len(self.snake)
            done = True
        else:
            # Shaped distance reward
            old_dist = self._manhattan(self.snake[-1], self.fruit)
            self.snake.append(new_head)

            if eating:
                # Do not pop tail => length + 1
                max_len = self.grid_size ** 3
                length_bonus = len(self.snake) / max_len
                reward += 1.0 + 0.5 * length_bonus
                self.score += 1
                self._spawn_fruit()
            else:
                # Move closer / farther from fruit
                new_dist = self._manhattan(new_head, self.fruit)
                reward += 0.01 * (old_dist - new_dist)
                # Pop tail: keep length constant
                self.snake.pop(0)

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        self.done = done
        obs = self._obs()
        info = {"score": self.score, "length": len(self.snake)}
        return obs, reward, done, info


# ============================================================
#                   CNN-based PPO Actor-Critic
# ============================================================

class ActorCriticCNN(nn.Module):
    """
    CNN-based Actor-Critic for 3D Snake.

    Input x: (batch, obs_dim)
        x = [grid_flat, global_feats]
        where grid_flat encodes (channels, D, H, W) flattened,
        and global_feats has fixed length global_dim.

    Internally it:
        - reshapes the grid part into (B, C, D, H, W)
        - applies 3D CNN + AdaptiveAvgPool3d(1) -> (B, 32)
        - concatenates with global features -> (B, 32 + global_dim)
        - passes through an MLP to produce policy logits + value.
    """

    def __init__(self, channels=3, global_dim=6, act_dim=6):
        super().__init__()
        self.channels = channels
        self.global_dim = global_dim

        # 3D CNN encoder (grid-size agnostic thanks to AdaptiveAvgPool3d)
        self.cnn = nn.Sequential(
            nn.Conv3d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(output_size=1),  # -> (B, 32, 1, 1, 1)
        )

        # Fully-connected trunk that combines CNN features + global features
        self.fc = nn.Sequential(
            nn.Linear(32 + global_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Policy and value heads
        self.policy = nn.Linear(128, act_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        """
        x: (B, obs_dim) = [grid_flatten, global_feats]
        """
        B = x.shape[0]
        global_feats = x[:, -self.global_dim:]
        grid_flat = x[:, :-self.global_dim]

        # Recover grid size from the flat part
        num_voxels = grid_flat.shape[1] // self.channels
        g = int(round(num_voxels ** (1.0 / 3.0)))
        grid = grid_flat.view(B, self.channels, g, g, g)

        # CNN encoder
        feat = self.cnn(grid).view(B, -1)

        # Concatenate with global features
        feat = torch.cat([feat, global_feats], dim=1)
        feat = self.fc(feat)

        logits = self.policy(feat)
        value = self.value(feat)
        return logits, value


# ============================================================
#                         PPO training
# ============================================================

SAVE_PATH = "snake3d_ppo_v3_cnn.pt"
USE_OLD_MODEL = False   # set True if you want to continue training


def save_model(model):
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"[saved] {SAVE_PATH}")


def maybe_load_model(model, device):
    if USE_OLD_MODEL and os.path.exists(SAVE_PATH):
        # weights_only=True is the recommended pattern from PyTorch
        state = torch.load(SAVE_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[loaded existing model] {SAVE_PATH}")
    else:
        print("[training from scratch]")


def train_ppo_cnn(
    grid_size=8,
    total_steps=3_000_000,
    rollout=4096,
    lr=1e-4,
    gamma=0.99,
    lam=0.95,
    clip_ratio=0.2,
    epochs=10,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = Snake3DConvEnv(grid_size=grid_size)
    obs = env.reset()
    obs_dim = obs.shape[0]

    model = ActorCriticCNN().to(device)
    maybe_load_model(model, device)
    optim_ = optim.Adam(model.parameters(), lr=lr)

    step_i = 0

    # Logging buffers for plotting
    steps_log = []
    mean_return_log = []
    mean_len_log = []
    policy_loss_log = []
    value_loss_log = []
    entropy_log = []

    # Episode statistics
    episode_returns = []
    episode_lengths = []
    current_ret = 0.0
    current_len = 0

    while step_i < total_steps:

        # ---------------- Collect a rollout ----------------
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        last_info = {"score": 0, "length": len(env.snake)}

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

        # ---------------- Compute GAE advantages ----------------
        obs_t_last = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        _, last_value = model(obs_t_last)
        last_value = last_value.item()

        vals = np.array(val_buf + [last_value], dtype=np.float32)
        rews = np.array(rew_buf, dtype=np.float32)
        dones = np.array(done_buf, dtype=np.float32)

        adv = np.zeros_like(rews, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rews))):
            delta = rews[t] + gamma * vals[t + 1] * (1 - dones[t]) - vals[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            adv[t] = gae
        ret = adv + vals[:-1]

        # ---------------- PPO update ----------------
        obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
        act_t = torch.tensor(act_buf, dtype=torch.int64, device=device)
        old_logp_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret, dtype=torch.float32, device=device)

        # Normalize advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_pl, total_vl, total_ent = 0.0, 0.0, 0.0
        mb_count = 0

        for _ in range(epochs):
            logits, value = model(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act_t)
            ratio = torch.exp(logp - old_logp_t)

            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (ret_t - value.squeeze()).pow(2).mean()
            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

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

        # Recent episode statistics (last 10 episodes)
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
            f" last_score={last_info['score']} len={last_info['length']} "
            f" mean_ret(10ep)={mean_ret:.2f} mean_len(10ep)={mean_len:.2f} "
            f" ploss={avg_pl:.4f} vloss={avg_vl:.4f} ent={avg_ent:.4f}"
        )

    # ================== Plot training curves ==================
    plt.figure()
    plt.plot(steps_log, mean_return_log)
    plt.xlabel("steps")
    plt.ylabel("mean episode return (last 10)")
    plt.title("Snake3D CNN PPO - Mean Return")
    plt.grid(True)

    plt.figure()
    plt.plot(steps_log, mean_len_log)
    plt.xlabel("steps")
    plt.ylabel("mean episode length (last 10)")
    plt.title("Snake3D CNN PPO - Mean Episode Length")
    plt.grid(True)

    plt.figure()
    plt.plot(steps_log, policy_loss_log, label="policy loss")
    plt.plot(steps_log, value_loss_log, label="value loss")
    plt.xlabel("steps")
    plt.ylabel("loss")
    plt.title("Snake3D CNN PPO - Loss")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(steps_log, entropy_log)
    plt.xlabel("steps")
    plt.ylabel("entropy")
    plt.title("Snake3D CNN PPO - Entropy")
    plt.grid(True)

    plt.show()

    return model


# ============================================================
#                            Main
# ============================================================

if __name__ == "__main__":
    # Ultimate training run: full 3D occupancy + CNN encoder
    train_ppo_cnn(
        grid_size=8,
        total_steps=3_000_000,
        rollout=4096,
        lr=1e-4,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        epochs=10,
    )