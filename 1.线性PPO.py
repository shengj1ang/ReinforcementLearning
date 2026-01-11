import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# =============================
# 1. 环境：DrugVesselEnv
# =============================
class DrugVesselEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.target = 1.0          # 目标血药浓度
        self.decay = 0.9           # 药物消除因子

        # 动作空间 0~2（给人看的），真正执行时还是会 clip 到这个范围
        self.action_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([2.0], dtype=np.float32),
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([5.0], dtype=np.float32),
        )
        self.state = None

    def reset(self):
        # 初始浓度
        self.state = np.array([0.5], dtype=np.float32)
        return self.state

    def step(self, action):
        # env_action: 已经是 numpy 的动作，这里再做一次 clip
        action = np.clip(action, self.action_space.low, self.action_space.high)
        injection = float(action[0])

        # 一室模型：浓度衰减 + 当前注射
        conc = self.decay * self.state + 0.1 * injection
        self.state = conc.astype(np.float32)

        # 误差
        error = conc[0] - self.target
        # B 方案：强化偏离惩罚
        reward = -(6.0 * (error ** 2)) - 0.01 * (injection ** 2)

        done = False
        info = {}
        return self.state, reward, done, info


# =============================
# 2. Actor-Critic 网络
# =============================
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.mu = nn.Linear(32, 1)
        # 初始 std 适中一点，保证有探索
        self.log_std = nn.Parameter(torch.tensor([-0.7]))  # std ≈ 0.5

    def forward(self, x):
        x = self.net(x)
        mu = self.mu(x)           # 未限制范围的均值
        std = self.log_std.exp()
        return mu, std


class Value(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# =============================
# 3. PPO Agent
# =============================
class PPOAgent:
    def __init__(self, lr=1e-3, eps_clip=0.2, gamma=0.99, ppo_epochs=10):
        self.policy = Policy()
        self.value = Value()

        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr
        )

        self.eps_clip = eps_clip
        self.gamma = gamma
        self.ppo_epochs = ppo_epochs

    def select_action(self, state):
        """
        返回：
          env_action: 送进 env 的动作 (np.array([a_clipped]))
          raw_action: 采样前未 clip 的动作 (tensor)
          logp      : 对 raw_action 的 log_prob (tensor)
        """
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # [1,1]
        mu, std = self.policy(s)
        dist = torch.distributions.Normal(mu, std)

        raw_a = dist.sample()              # [1,1] 未 clip
        logp = dist.log_prob(raw_a)        # [1,1]

        # 送进环境的动作：clip 到 0~2
        env_a = torch.clamp(raw_a, 0.0, 2.0)

        return env_a.detach().numpy()[0], raw_a.detach()[0], logp.detach()[0]

    def compute_returns_and_advantages(self, rewards, states):
        """
        rewards: list[float], 长度 T
        states:  tensor [T,1]
        """
        # 1) discounted returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # 2) value baseline
        with torch.no_grad():
            values = self.value(states).squeeze(-1)

        # 3) advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def update(self, states, raw_actions, logp_old, rewards):
        """
        用当前 episode 的数据做多次 PPO 更新
        states:      [T,1]
        raw_actions: [T,1]   （未 clip 动作）
        logp_old:    [T,1]   （旧策略在 raw_actions 上的 log_prob）
        rewards:     list[float]
        """
        states = states.detach()
        raw_actions = raw_actions.detach()
        logp_old = logp_old.detach().squeeze(-1)
        returns, advantages = self.compute_returns_and_advantages(rewards, states)

        total_loss = 0.0

        for _ in range(self.ppo_epochs):
            mu, std = self.policy(states)
            dist = torch.distributions.Normal(mu, std)

            new_logp = dist.log_prob(raw_actions).squeeze(-1)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - logp_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            values = self.value(states).squeeze(-1)
            value_loss = F.mse_loss(values, returns)

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / self.ppo_epochs


# =============================
# 4. Training Loop
# =============================
env = DrugVesselEnv()
agent = PPOAgent(lr=1e-3, eps_clip=0.2, gamma=0.99, ppo_epochs=10)

episodes = 800        # 可以再加大，比如 1000
time_steps = 200      # 每个 episode 的时间步数

all_rewards = []
all_concentrations = []
all_losses = []

for ep in range(episodes):
    state = env.reset()
    states, raw_actions, logp_old, rewards = [], [], [], []
    conc_history = []

    for t in range(time_steps):
        conc_history.append(state[0])

        env_action, raw_action, logp = agent.select_action(state)
        next_state, reward, done, _ = env.step(env_action)

        states.append(torch.tensor(state, dtype=torch.float32))
        raw_actions.append(raw_action)
        logp_old.append(logp)
        rewards.append(reward)

        state = next_state
        if done:
            break

    states = torch.stack(states)          # [T,1]
    raw_actions = torch.stack(raw_actions)  # [T,1]
    logp_old = torch.stack(logp_old)      # [T,1]

    all_concentrations.append(conc_history)
    avg_return = np.mean(rewards)
    all_rewards.append(avg_return)

    loss = agent.update(states, raw_actions, logp_old, rewards)
    all_losses.append(loss)

    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1}/{episodes}, AvgReward: {avg_return:.3f}, Loss: {loss:.4f}")

# =============================
# 5. 可视化
# =============================

# 1) Reward per Episode
plt.figure(figsize=(8, 5))
plt.plot(all_rewards)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Reward per Episode')
plt.grid(True)
plt.show()

# 2) Concentration of Last Episode
plt.figure(figsize=(8, 5))
plt.plot(all_concentrations[-1], label='Concentration')
plt.axhline(y=env.target, color='red', linestyle='--', label='Target')
plt.xlabel('Time Step')
plt.ylabel('Concentration')
plt.title('Drug Concentration Over Time (Last Episode)')
plt.legend()
plt.grid(True)
plt.show()

# 3) PPO Loss
plt.figure(figsize=(8, 5))
plt.plot(all_losses)
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('PPO Loss per Episode')
plt.grid(True)
plt.show()