import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =============================
# 1. Environment: DrugVesselEnv
# 可切换非线性 & 噪声 & 毒性惩罚的 DrugVesselEnv
# 仍然是一室模型（one-compartment），但动力学更复杂
# use_nonlinear=False 时：退化为线性版本（类似旧代码）
# use_nonlinear=True 时：你在更新里加入了 非线性项（例如吸收饱和/清除非线性），
# =============================
class DrugVesselEnv(gym.Env):
    """
    One-compartment drug model with optional:
    - process / observation noise
    - nonlinear absorption and toxicity penalty

    Args:
        use_nonlinear (bool): if True, use nonlinear dynamics and toxicity term.
                              if False, use purely linear dynamics.
        use_noise     (bool): if True, add process + observation noise.
                              if False, dynamics and observations are noise-free.
    """
    def __init__(self, use_nonlinear: bool = True, use_noise: bool = False):
        super().__init__()

        # ----- basic settings -----
        self.target = 1.0          # target drug concentration
        self.decay = 0.9           # elimination factor (linear part)

        # ----- nonlinear parameters (used only if use_nonlinear=True) -----
        # saturation: high concentration reduces effective injection
        self.saturation_alpha = 0.7
        # toxicity: extra penalty if concentration too high
        self.toxic_threshold = 2.0
        self.toxic_penalty_coef = 8.0

        # ----- noise parameters -----
        self.process_noise_std = 0.01   # std of process noise in dynamics
        self.obs_noise_std = 0.05       # std of observation noise

        # switches
        self.use_nonlinear = use_nonlinear
        self.use_noise = use_noise

        # Action space: injection amount in [0, 2]
        self.action_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([2.0], dtype=np.float32),
        )

        # Observation: concentration (or noisy version), clipped to [0, 5]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([5.0], dtype=np.float32),
        )

        # This holds the true concentration
        self.state = None

    # ----------------- reset -----------------
    def reset(self, seed=None, options=None):
        """
        Reset the environment:
        - set true concentration to 0.5
        - return either noisy or noise-free observation
        """
        super().reset(seed=seed)
        self.state = np.array([0.5], dtype=np.float32)

        if self.use_noise:
            obs_noise = np.random.normal(
                0.0, self.obs_noise_std, size=self.state.shape
            )
        else:
            obs_noise = 0.0

        obs = self.state + obs_noise
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        return obs

    # ----------------- step -----------------
    def step(self, action):
        """
        Step the environment:
        - apply either linear or nonlinear dynamics
        - optionally add process noise
        - compute reward (tracking + control cost, plus toxicity if nonlinear)
        - optionally add observation noise to the returned state
        """
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        injection = float(action[0])
        current_conc = float(self.state[0])

        # -------- 1) dynamics: linear or nonlinear --------
        if not self.use_nonlinear:
            # ----- purely linear dynamics -----
            conc = self.decay * self.state + 0.1 * injection
            conc = float(conc[0])
        else:
            # ----- nonlinear version -----
            # linear decay on current concentration
            conc = self.decay * current_conc

            # nonlinear absorption (saturation):
            # when current concentration is above target, effective injection is reduced
            over_target = max(0.0, current_conc - self.target)
            effective_injection = 0.1 * injection / (1.0 + self.saturation_alpha * over_target)

            conc = conc + effective_injection

        # -------- 2) process noise (if enabled) --------
        if self.use_noise:
            process_noise = np.random.normal(0.0, self.process_noise_std)
            conc = conc + float(process_noise)

        # update true state
        self.state = np.array([conc], dtype=np.float32)

        # -------- 3) reward computation --------
        error = conc - self.target

        # base tracking + control cost (used in both linear and nonlinear cases)
        reward = -(6.0 * (error ** 2)) - 0.01 * (injection ** 2)

        # extra toxicity penalty only if nonlinear is enabled
        if self.use_nonlinear and conc > self.toxic_threshold:
            toxic_excess = conc - self.toxic_threshold
            reward -= self.toxic_penalty_coef * (toxic_excess ** 2)

        # -------- 4) observation: noisy or clean --------
        if self.use_noise:
            obs_noise = np.random.normal(0.0, self.obs_noise_std, size=self.state.shape)
        else:
            obs_noise = 0.0

        obs = self.state + obs_noise
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        done = False  # fixed-length episodes
        info = {
            "true_concentration": self.state.copy()
        }

        return obs, reward, done, info

# =============================
# 2. Actor-Critic Networks
# =============================
class Policy(nn.Module):
    """
    Policy network:
    - Input: noisy concentration (1D)
    - Output: Gaussian parameters (mu, std) for an unconstrained action z
    - Environment action is: a = 1 + tanh(z)  in [0, 2]
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.mu = nn.Linear(64, 1)

        # Global log_std parameter, clamped during forward for stability
        self.log_std = nn.Parameter(torch.tensor([-0.7]))  # around std ≈ 0.5

    def forward(self, x):
        """
        Forward pass:
        - x: [batch, 1]
        Returns:
        - mu: [batch, 1]
        - std: [batch, 1]
        """
        x = self.net(x)
        mu = self.mu(x)
        # Clamp log_std to avoid extremely small/large std
        log_std = torch.clamp(self.log_std, -2.0, 2.0)
        std = log_std.exp()
        return mu, std


class Value(nn.Module):
    """
    Value network:
    - Input: noisy concentration (1D)
    - Output: scalar state value V(s)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# =============================
# 3. PPO Agent with GAE + tanh action
# =============================
class PPOAgent:
    def __init__(
        self,
        lr=1e-3,
        eps_clip=0.2,
        gamma=0.99,
        lam=0.95,
        ppo_epochs=10,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5
    ):
        self.policy = Policy()
        self.value = Value()

        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr
        )

        self.eps_clip = eps_clip
        self.gamma = gamma
        self.lam = lam
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

    # --------- Action selection ---------
    def select_action(self, state):
        """
        Select an action given current (noisy) observation.

        Args:
            state: np.array([concentration])

        Returns:
            env_action: np.array([a]) in [0, 2], used in environment
            raw_action: tensor([z]), unconstrained Gaussian sample (before tanh)
            logp:       tensor scalar, log_prob of raw_action (over z)
        """
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # [1,1]
        mu, std = self.policy(s)
        dist = torch.distributions.Normal(mu, std)

        # z is unconstrained action
        z = dist.rsample()  # [1,1], reparameterization trick
        # Squash to [-1, 1]
        a_tanh = torch.tanh(z)
        # Map to [0, 2]
        env_a = 1.0 + a_tanh  # center at 1.0, radius 1.0

        # Log probability of z under the Gaussian (jacobian term is ignored,
        # which is common in tanh-Gaussian PPO implementations as it cancels in the ratio)
        logp = dist.log_prob(z).sum(dim=-1)  # [1]

        return env_a.detach().numpy()[0], z.detach()[0], logp.detach()[0]

    # --------- GAE computation ---------
    def compute_returns_and_advantages(self, rewards, states, last_state):
        """
        Use GAE(λ) to compute returns and advantages.

        Args:
            rewards   : list of floats, length T
            states    : tensor [T, 1]
            last_state: np.array([obs]) of the final state of the episode

        Returns:
            returns    : tensor [T]
            advantages : tensor [T]
        """
        with torch.no_grad():
            # values for all states in the episode: [T, 1] -> [T]
            values = self.value(states).squeeze(-1)

            # bootstrap value for the final state
            last_state_t = torch.tensor(last_state, dtype=torch.float32).unsqueeze(0)  # [1,1]
            last_value = self.value(last_state_t).squeeze()  # scalar (0D tensor)

        # Now we make an extended value vector of shape [T+1]:
        # values: [T]
        # last_value: scalar -> [1]
        last_value = last_value.view(1)  # [1]
        values_ext = torch.cat([values, last_value], dim=0)  # [T+1]

        T = len(rewards)
        advantages = torch.zeros(T)
        gae = 0.0

        # GAE(λ) backward recursion
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values_ext[t + 1] - values_ext[t]
            gae = delta + self.gamma * self.lam * gae
            advantages[t] = gae

        returns = advantages + values  # V(s) + A(s) = return
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    # --------- PPO update ---------
    def update(self, states, raw_actions, logp_old, rewards, last_state):
        """
        Perform PPO update using data from one episode.

        Args:
            states     : tensor [T, 1], noisy observations
            raw_actions: tensor [T, 1], unconstrained z samples before tanh
            logp_old   : tensor [T], old log_prob of raw_actions
            rewards    : list[float] length T
            last_state : np.array([obs]) final observation of this episode
        """
        # Detach to avoid accidental backprop through old graph
        states = states.detach()
        raw_actions = raw_actions.detach()
        logp_old = logp_old.detach()

        # Compute returns and advantages via GAE
        returns, advantages = self.compute_returns_and_advantages(
            rewards, states, last_state
        )

        total_loss = 0.0

        for _ in range(self.ppo_epochs):
            mu, std = self.policy(states)
            dist = torch.distributions.Normal(mu, std)

            # New log_prob of the same raw actions z
            new_logp = dist.log_prob(raw_actions).sum(dim=-1)  # [T]
            entropy = dist.entropy().sum(dim=-1).mean()        # scalar

            # Probability ratio for PPO clipping
            ratio = torch.exp(new_logp - logp_old)  # [T]

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            values = self.value(states).squeeze(-1)
            value_loss = F.mse_loss(values, returns)

            # Total PPO loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value.parameters()),
                self.max_grad_norm
            )

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / self.ppo_epochs


# =============================
# 4. Training Loop
# =============================
env = DrugVesselEnv()
agent = PPOAgent(
    lr=1e-3,
    eps_clip=0.2,
    gamma=0.99,
    lam=0.95,
    ppo_epochs=10,
    entropy_coef=0.01,
    value_coef=0.5,
    max_grad_norm=0.5
)

episodes = 1800       # number of episodes
time_steps = 200      # time steps per episode

all_rewards = []          # average reward per episode
all_concentrations = []   # noisy observations per episode
all_losses = []           # PPO loss per episode

for ep in range(episodes):
    state = env.reset()
    states, raw_actions, logp_old, rewards = [], [], [], []
    conc_history = []

    for t in range(time_steps):
        # Record noisy concentration
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

    states = torch.stack(states)               # [T, 1]
    raw_actions = torch.stack(raw_actions)     # [T, 1]
    logp_old = torch.stack(logp_old).squeeze(-1)  # [T]

    all_concentrations.append(conc_history)
    avg_return = np.mean(rewards)
    all_rewards.append(avg_return)

    # Use the final state of this episode for bootstrapping in GAE
    loss = agent.update(states, raw_actions, logp_old, rewards, last_state=state)
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
plt.title('Reward per Episode (Noisy Environment, PPO with GAE & tanh)')
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