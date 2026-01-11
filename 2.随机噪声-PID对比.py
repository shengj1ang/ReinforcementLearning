import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# =============================
# 1. Environment: DrugVesselEnv
#    (same logic as you provided)
# =============================
class DrugVesselEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.target = 1.0          # target drug concentration
        self.decay = 0.9           # elimination factor

        # ----- Noise parameters (you can play with these) -----
        self.process_noise_std = 0.01   # std of process noise in dynamics
        self.obs_noise_std = 0.05       # std of observation noise

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

        # Add observation noise to simulate sensor measurement
        obs_noise = np.random.normal(0.0, self.obs_noise_std, size=self.state.shape)
        obs = self.state + obs_noise
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs  # return noisy observation instead of true state

    def step(self, action):
        # env_action: already numpy-like, clip it again to be safe
        action = np.clip(action, self.action_space.low, self.action_space.high)
        injection = float(action[0])

        # One-compartment model: decay + current injection (true dynamics)
        conc = self.decay * self.state + 0.1 * injection

        # Add process noise to the true concentration (model uncertainty)
        process_noise = np.random.normal(0.0, self.process_noise_std, size=conc.shape)
        conc = conc + process_noise

        # Keep internal true state
        self.state = conc.astype(np.float32)

        # Error is computed on the true concentration
        error = conc[0] - self.target

        # Reward: penalize squared deviation + small penalty on injection
        reward = -(6.0 * (error ** 2)) - 0.01 * (injection ** 2)

        # Build noisy observation for the agent
        obs_noise = np.random.normal(0.0, self.obs_noise_std, size=self.state.shape)
        obs = self.state + obs_noise
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)

        done = False
        info = {
            "true_concentration": self.state.copy()   # you can log this if needed
        }

        return obs, reward, done, info  # return noisy observation


# =============================
# 2. PID Controller
# =============================
class PIDController:
    def __init__(self, Kp=2.0, Ki=0.05, Kd=0.3,
                 action_low=0.0, action_high=2.0):
        """
        Simple PID controller for continuous control.
        The output will be clipped to [action_low, action_high].

        Kp, Ki, Kd: PID gains.
        action_low, action_high: bounds of the actuator (same as env.action_space).
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integral = 0.0
        self.prev_error = 0.0

        self.action_low = action_low
        self.action_high = action_high

    def reset(self):
        """Reset PID internal state at the beginning of each episode."""
        self.integral = 0.0
        self.prev_error = 0.0

    def compute_action(self, measurement, target):
        """
        Compute the PID control action given a noisy measurement and target.

        measurement: observed concentration (noisy)
        target: desired concentration
        """
        # Error between target and measurement
        error = target - measurement

        # Integral term accumulates error over time
        self.integral += error

        # Derivative term approximates the slope of the error
        derivative = error - self.prev_error

        # PID formula: proportional + integral + derivative terms
        action = (
            self.Kp * error +
            self.Ki * self.integral +
            self.Kd * derivative
        )

        # Clip to actuator bounds to respect the environment limits
        action = np.clip(action, self.action_low, self.action_high)

        # Store current error for next derivative computation
        self.prev_error = error

        return float(action)


# =============================
# 3. Run PID on the Environment
# =============================
def run_pid(env, pid, episodes=3, time_steps=200):
    """
    Run PID control on the environment for a number of episodes.

    Returns:
        all_conc: list of lists, true concentration trajectories
        all_inj:  list of lists, injection action trajectories
    """
    all_conc = []
    all_inj = []

    for ep in range(episodes):
        obs = env.reset()     # noisy observation
        pid.reset()

        conc_history = []
        inj_history = []

        for t in range(time_steps):
            # Measurement is the noisy observation from the environment
            measured_conc = float(obs[0])

            # PID computes control action based on noisy measurement and target
            action = pid.compute_action(measured_conc, env.target)

            # Environment expects an array-like action with shape (1,)
            obs, reward, done, info = env.step(np.array([action], dtype=np.float32))

            # Store true concentration (from info) and the applied injection
            true_conc = float(info["true_concentration"][0])
            conc_history.append(true_conc)
            inj_history.append(action)

            if done:
                break

        all_conc.append(conc_history)
        all_inj.append(inj_history)

        print(f"[PID] Episode {ep+1}/{episodes} completed.")

    return all_conc, all_inj


# =============================
# 4. Visualization
# =============================
def plot_pid(conc, inj, env, title_suffix=""):
    """
    Plot concentration and injection trajectory for a single episode.
    """
    # Plot true concentration over time
    plt.figure(figsize=(8, 5))
    plt.plot(conc, label="True Concentration")
    plt.axhline(env.target, color='r', linestyle='--', label='Target')
    plt.title(f"PID - Drug Concentration Over Time{title_suffix}")
    plt.xlabel("Time Step")
    plt.ylabel("Concentration")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot injection over time
    plt.figure(figsize=(8, 5))
    plt.plot(inj, label="Injection Rate")
    plt.title(f"PID - Injection Actions{title_suffix}")
    plt.xlabel("Time Step")
    plt.ylabel("Action")
    plt.grid(True)
    plt.show()


# =============================
# 5. Main: run PID demo
# =============================
if __name__ == "__main__":
    # Create environment
    env = DrugVesselEnv()

    # Create PID controller (you can tune these gains)
    pid = PIDController(
        Kp=21.0,   # proportional gain
        Ki=0.05,  # integral gain
        Kd=0.3,   # derivative gain
        action_low=env.action_space.low[0],
        action_high=env.action_space.high[0],
    )

    # Run PID for a few episodes
    episodes = 1800
    time_steps = 200
    all_conc, all_inj = run_pid(env, pid, episodes=episodes, time_steps=time_steps)

    # Plot the last episode as an example
    plot_pid(all_conc[-1], all_inj[-1], env, title_suffix=f" (Episode {episodes})")