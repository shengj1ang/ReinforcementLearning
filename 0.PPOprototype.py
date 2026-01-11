import gym # for the first prototype, i use old version of gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ===== 1. env =====
class DrugVesselEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.target = 1.0
        self.decay = 0.9
        self.action_space = gym.spaces.Box(low=0, high=2, shape=(1,))
        self.observation_space = gym.spaces.Box(low=0, high=5, shape=(1,))
        self.state = None

    def reset(self):
        self.state = np.array([0.5])
        return self.state

    def step(self, action):
        injection = float(action[0])
        conc = self.decay * self.state + 0.1 * injection
        self.state = conc
        reward = -abs(conc - self.target) - 0.05 * (injection**2)
        done = False
        return self.state, reward, done, {}

# ===== 2. PPO network =====
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1,32), nn.ReLU(), nn.Linear(32,32), nn.ReLU())
        self.mu = nn.Linear(32,1)
        self.log_std = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        x = self.net(x)
        return self.mu(x), self.log_std.exp()

# ===== 3. PPO Agent =====
class PPOAgent:
    def __init__(self, lr=1e-3, eps=0.2):
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.eps = eps

    def select_action(self, state):
        s = torch.tensor(state, dtype=torch.float32)
        mu, std = self.policy(s)
        dist = torch.distributions.Normal(mu, std)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a.detach().numpy(), logp.detach()

    def update(self, states, actions, old_logp, returns, adv):
        mu, std = self.policy(states)
        dist = torch.distributions.Normal(mu, std)
        new_logp = dist.log_prob(actions)
        ratio = torch.exp(new_logp - old_logp)
        clip_obj = torch.min(ratio*adv, torch.clamp(ratio,1-self.eps,1+self.eps)*adv)
        loss = -clip_obj.mean() + 0.5*((returns - returns.mean())**2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ===== 4. training loop =====
env = DrugVesselEnv()
agent = PPOAgent()
episodes = 800
time_steps = 200

all_rewards = []
all_concentrations = []
all_losses = []  

for ep in range(episodes):
    state = env.reset()
    states, actions, logp_old, rewards = [], [], [], []
    conc_history = []

    for t in range(time_steps):
        conc_history.append(state[0])
        action, logp = agent.select_action(state)
        next_state, reward, _, _ = env.step(action)

        states.append(torch.tensor(state, dtype=torch.float32))
        actions.append(torch.tensor(action, dtype=torch.float32))
        logp_old.append(logp)
        rewards.append(reward)
        state = next_state

    all_concentrations.append(conc_history)
    returns = torch.tensor(rewards, dtype=torch.float32)
    adv = returns - returns.mean()

    # loss
    mu, std = agent.policy(torch.stack(states))
    dist = torch.distributions.Normal(mu, std)
    new_logp = dist.log_prob(torch.stack(actions))
    ratio = torch.exp(new_logp - torch.stack(logp_old))
    clip_obj = torch.min(ratio*adv, torch.clamp(ratio,1-agent.eps,1+agent.eps)*adv)
    loss = -clip_obj.mean() + 0.5*((returns - returns.mean())**2).mean()
    all_losses.append(loss.item())

    # update
    agent.update(torch.stack(states), torch.stack(actions), torch.stack(logp_old), returns, adv)
    all_rewards.append(returns.mean().item())

# ===== visulization =====
import matplotlib.pyplot as plt
#plt.style.use('seaborn-darkgrid') 

# ===== 1. Reward =====
plt.figure(figsize=(8,5))
plt.plot(all_rewards, color='tab:blue', lw=2)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Average Reward', fontsize=12)
plt.title('Reward per Episode', fontsize=14)
plt.grid(True)
plt.show()

# ===== 2. Blood Concentration =====
plt.figure(figsize=(8,5))
plt.plot(all_concentrations[-1], color='tab:green', lw=2, label='Concentration')
plt.axhline(y=env.target, color='red', linestyle='--', lw=2, label='Target')
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Concentration', fontsize=12)
plt.title('Drug Concentration Over Time (Last Episode)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()

# ===== 3. PPO Loss =====
plt.figure(figsize=(8,5))
plt.plot(all_losses, color='tab:orange', lw=2)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('PPO Loss per Episode', fontsize=14)
plt.grid(True)
plt.show()