import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# =============================
# 1. Environment: DrugVesselEnv
# 线性 + 随机噪声版 DrugVesselEnv
# =============================
class DrugVesselEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.target = 1.0          # target drug concentration
        self.decay = 0.9           # elimination factor

        # ----- Noise parameters (you can play with these) -----  # <<<
        self.process_noise_std = 0.01   # std of process noise in dynamics  # <<<
        self.obs_noise_std = 0.05       # std of observation noise          # <<<

        # Action space: 0~2 (human interpretable), will be clipped anyway
        self.action_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([2.0], dtype=np.float32),
        )
        # Observation is still drug concentration, but noisy version will be returned
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([5.0], dtype=np.float32),
        )
        self.state = None  # this will store the "true" concentration

    def reset(self):
        # Initial true concentration
        self.state = np.array([0.5], dtype=np.float32)

        # Add observation noise to simulate sensor measurement           # <<<
        obs_noise = np.random.normal(0.0, self.obs_noise_std, size=self.state.shape)  # <<<
        obs = self.state + obs_noise                                       # <<<
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)  # <<<
        return obs  # return noisy observation instead of true state       # <<<

    def step(self, action):
        # env_action: already numpy, clip it again to be safe
        action = np.clip(action, self.action_space.low, self.action_space.high)
        injection = float(action[0])

        # One-compartment model: decay + current injection (true dynamics)
        conc = self.decay * self.state + 0.1 * injection

        # Add process noise to the true concentration (model uncertainty)      # <<<
        process_noise = np.random.normal(0.0, self.process_noise_std, size=conc.shape)  # <<<
        conc = conc + process_noise                                            # <<<

        # Keep internal true state
        self.state = conc.astype(np.float32)

        # Error is computed on the true concentration
        error = conc[0] - self.target

        # Reward: penalize squared deviation + small penalty on injection
        reward = -(6.0 * (error ** 2)) - 0.01 * (injection ** 2)

        # Build noisy observation for the agent                               # <<<
        obs_noise = np.random.normal(0.0, self.obs_noise_std, size=self.state.shape)  # <<<
        obs = self.state + obs_noise                                          # <<<
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)  # <<<

        done = False
        info = {
            "true_concentration": self.state.copy()   # you can log this if needed   # <<<
        }

        return obs, reward, done, info  # return noisy observation


# =============================
# 2. Actor-Critic Networks
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
        # Initialize log_std with a moderate value to ensure some exploration
        self.log_std = nn.Parameter(torch.tensor([-0.7]))  # std ≈ 0.5

    def forward(self, x):
        x = self.net(x)
        mu = self.mu(x)           # unbounded mean
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
        Returns:
          env_action: action sent into the environment (np.array([a_clipped]))
          raw_action: un-clipped action sampled from the Normal policy (tensor)
          logp      : log_prob of raw_action under current policy (tensor)
        """
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # [1,1]
        mu, std = self.policy(s)
        dist = torch.distributions.Normal(mu, std)

        raw_a = dist.sample()              # [1,1] un-clipped action
        logp = dist.log_prob(raw_a)        # [1,1]

        # The action sent into env is clipped to [0, 2]
        env_a = torch.clamp(raw_a, 0.0, 2.0)

        return env_a.detach().numpy()[0], raw_a.detach()[0], logp.detach()[0]

    def compute_returns_and_advantages(self, rewards, states):
        """
        rewards: list[float], length T
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
        Perform multiple PPO updates using data from one episode.

        states:      [T,1]     noisy observations
        raw_actions: [T,1]     un-clipped actions
        logp_old:    [T,1]     old log_prob of raw_actions
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

            # Combined PPO loss
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

episodes = 200        # you can increase this if needed
time_steps = 500      # time steps per episode

all_rewards = []
all_concentrations = []   # will store noisy observations for the last episode or all episodes
all_losses = []

for ep in range(episodes):
    state = env.reset()
    states, raw_actions, logp_old, rewards = [], [], [], []
    conc_history = []

    for t in range(time_steps):
        # Here we record the *observed* concentration (noisy)                 # <<<
        conc_history.append(state[0])

        env_action, raw_action, logp = agent.select_action(state)
        next_state, reward, done, info = env.step(env_action)

        states.append(torch.tensor(state, dtype=torch.float32))
        raw_actions.append(raw_action)
        logp_old.append(logp)
        rewards.append(reward)

        state = next_state
        if done:
            break

    states = torch.stack(states)              # [T,1]
    raw_actions = torch.stack(raw_actions)    # [T,1]
    logp_old = torch.stack(logp_old)          # [T,1]

    all_concentrations.append(conc_history)
    avg_return = np.mean(rewards)
    all_rewards.append(avg_return)

    loss = agent.update(states, raw_actions, logp_old, rewards)
    all_losses.append(loss)

    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1}/{episodes}, AvgReward: {avg_return:.3f}, Loss: {loss:.4f}")

# =============================
# 5. Visualization
# =============================

# 1) Reward per Episode
plt.figure(figsize=(8, 5))
plt.plot(all_rewards)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Reward per Episode (Noisy Environment)')
plt.grid(True)
plt.show()

# 2) Concentration of Last Episode (noisy observation)
plt.figure(figsize=(8, 5))
plt.plot(all_concentrations[-1], label='Observed Concentration (noisy)')
plt.axhline(y=env.target, color='red', linestyle='--', label='Target')
plt.xlabel('Time Step')
plt.ylabel('Concentration')
plt.title('Drug Concentration Over Time (Last Episode, Noisy Observations)')
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