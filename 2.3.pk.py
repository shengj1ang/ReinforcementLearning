import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class PKEnv(gym.Env):
    """
    Pharmacokinetic (PK) environment for RL-controlled drug dosing.

    Features:
    - ODE-based one- or two-compartment PK model
    - Optional nonlinear elimination (Michaelis-Menten-like)
    - Optional process noise and observation noise
    - Observation is always the central compartment concentration (1D),
      so it is compatible with a 1D PPO policy network.

    Args:
        model_type (str): 'one_compartment' or 'two_compartment'.
        use_nonlinear (bool): if True, use nonlinear elimination on central compartment.
        use_noise (bool): if True, add process noise and observation noise.
        dt (float): physical time (e.g. hours) per RL step.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        model_type: str = "one_compartment",
        use_nonlinear: bool = False,
        use_noise: bool = True,
        dt: float = 0.1,
    ):
        super().__init__()

        assert model_type in ("one_compartment", "two_compartment")
        self.model_type = model_type
        self.use_nonlinear = use_nonlinear
        self.use_noise = use_noise
        self.dt = dt

        # ----- Target concentration -----
        self.target = 1.0

        # ----- Linear PK parameters -----
        # for one-compartment: dc/dt = -k_elim * c + u
        self.k_elim = 0.5  # elimination rate from central compartment

        # for two-compartment:
        # dc1/dt = -(k10 + k12) c1 + k21 c2 + u
        # dc2/dt = k12 c1 - k21 c2
        self.k10 = 0.3   # elimination from central
        self.k12 = 0.2   # central -> peripheral
        self.k21 = 0.1   # peripheral -> central

        # ----- Nonlinear elimination parameters (Michaelis-Menten-like) -----
        # For central compartment:
        # elim_rate = Vmax * c / (Km + c)
        self.Vmax = 0.8
        self.Km = 1.0

        # ----- Noise parameters -----
        self.process_noise_std = 0.01
        self.obs_noise_std = 0.05

        # ----- Action space: infusion rate / dose per unit time -----
        # Keep the same action range as your old environment: [0, 2]
        self.action_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([2.0], dtype=np.float32),
        )

        # ----- Observation space: always central concentration only -----
        # This keeps your PPO input dimension = 1
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([5.0], dtype=np.float32),
        )

        # True internal state:
        # - one_compartment: [c1]
        # - two_compartment: [c1, c2]
        self.state = None

    # ----------------- ODE dynamics -----------------
    def _ode_one_compartment(self, t, y, u):
        """
        ODE for one-compartment model:
        dc/dt = -elim(c) + u

        elim(c) is either linear (k_elim * c) or nonlinear (Vmax * c / (Km + c)).
        """
        c = y[0]

        if self.use_nonlinear:
            # nonlinear elimination
            elim = self.Vmax * c / (self.Km + c + 1e-8)
        else:
            # linear elimination
            elim = self.k_elim * c

        dc_dt = -elim + u
        return [dc_dt]

    def _ode_two_compartment(self, t, y, u):
        """
        ODE for two-compartment model:
        c1: central compartment
        c2: peripheral compartment

        Linear part:
            dc1/dt = -(k10 + k12) c1 + k21 c2 + u
            dc2/dt = k12 c1 - k21 c2

        Nonlinear elimination (if enabled) replaces k10 * c1 by
            elim(c1) = Vmax * c1 / (Km + c1).
        """
        c1, c2 = y

        if self.use_nonlinear:
            elim = self.Vmax * c1 / (self.Km + c1 + 1e-8)
            dc1_dt = -(elim + self.k12 * c1) + self.k21 * c2 + u
        else:
            dc1_dt = -(self.k10 + self.k12) * c1 + self.k21 * c2 + u

        dc2_dt = self.k12 * c1 - self.k21 * c2

        return [dc1_dt, dc2_dt]

    # ----------------- reset -----------------
    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        For one_compartment:
            state = [0.5]
        For two_compartment:
            state = [0.5, 0.3]

        Returns:
            obs: noisy or clean central concentration, shape (1,)
        """
        super().reset(seed=seed)
        
        
         # 病人类型：0 = clearance 慢，1 = clearance 快
        self.patient_type = np.random.choice([0, 1])

        if self.patient_type == 0:
            # 慢代谢患者
            self.k10 = 0.15
            self.k12 = 0.1
            self.k21 = 0.05
        else:
            # 快代谢患者
            self.k10 = 0.8
            self.k12 = 0.4
            self.k21 = 0.2


            
        if self.model_type == "one_compartment":
            self.state = np.array([0.5], dtype=np.float32)
        else:
            # central starts at 0.5, peripheral at 0.3 (arbitrary but reasonable)
            self.state = np.array([0.5, 0.3], dtype=np.float32)

        # Build observation from central compartment
        c1 = float(self.state[0])

        if self.use_noise:
            obs_noise = np.random.normal(0.0, self.obs_noise_std)
        else:
            obs_noise = 0.0

        obs = np.array([c1 + obs_noise], dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        return obs

    # ----------------- step -----------------
    def step(self, action):
        """
        Step the environment by dt using ODE integration.

        Args:
            action: np.array([u]), infusion rate / dose per unit time

        Returns:
            obs: noisy or clean central concentration, shape (1,)
            reward: scalar
            done: always False (fixed-length episodes)
            info: dict with full true state and central concentration
        """
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        u = float(action[0])

        # Solve ODE over [0, dt] starting from current true state
        if self.model_type == "one_compartment":
            fun = lambda t, y: self._ode_one_compartment(t, y, u)
        else:
            fun = lambda t, y: self._ode_two_compartment(t, y, u)

        # Use a small integration window [0, dt]
        sol = solve_ivp(
            fun=fun,
            t_span=(0.0, self.dt),
            y0=self.state.astype(float),
            method="RK45",
            max_step=self.dt / 5.0,  # more steps for accuracy
        )

        # Take the last point as the new true state
        new_state = sol.y[:, -1]

        # Add process noise to central compartment (and possibly to others)
        if self.use_noise:
            if self.model_type == "one_compartment":
                new_state[0] += np.random.normal(0.0, self.process_noise_std)
            else:
                # add small noise to both compartments
                new_state += np.random.normal(
                    0.0, self.process_noise_std, size=new_state.shape
                )

        self.state = new_state.astype(np.float32)

        # Central compartment concentration
        c1 = float(self.state[0])

        # ----- Reward: tracking + control cost -----
        error = c1 - self.target
        # Quadratic tracking penalty + quadratic control cost
        reward = -(6.0 * (error ** 2)) - 0.01 * (u ** 2)

        # ----- Observation: central concentration with optional noise -----
        if self.use_noise:
            obs_noise = np.random.normal(0.0, self.obs_noise_std)
        else:
            obs_noise = 0.0

        obs = np.array([c1 + obs_noise], dtype=np.float32)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        done = False

        info = {
            "true_state": self.state.copy(),      # full state (1D or 2D)
            "central_concentration": c1,
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



env = PKEnv(
    model_type="two_compartment",
    use_nonlinear=True,
    use_noise=True,
    dt=0.1,
)

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

episodes = 300       # number of episodes
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