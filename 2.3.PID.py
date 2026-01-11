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


env = PKEnv(
    model_type="two_compartment",
    use_nonlinear=True,
    use_noise=False,
    dt=0.1,
)



class PIDController:
    def __init__(self, kp, ki, kd, dt, output_limits=(0, 2)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_limits = output_limits
        
        self.integral = 0.0
        self.previous_error = 0.0
        self.target = 1.0  # Target concentration
    
    def reset(self):
        self.integral = 0.0
        self.previous_error = 0.0
    
    def compute(self, measurement):
        error = self.target - measurement
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term
        self.integral += error * self.dt
        integral_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / self.dt
        derivative_term = self.kd * derivative
        
        # PID output
        output = proportional + integral_term + derivative_term
        
        # Limit output range
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        self.previous_error = error
        
        return np.array([output], dtype=np.float32)


# Create PID controller
pid = PIDController(kp=1.5, ki=0.8, kd=0.1, dt=env.dt)

# Run PID control with different patients
concentrations = []
actions = []
targets = []
times = []
patient_types = []  # Track patient types

num_steps = 1000
patient_change_interval = 200  # Change patient every 200 steps

for step in range(num_steps):
    # Change patient parameters periodically
    if step % patient_change_interval == 0:
        obs = env.reset()
        pid.reset()  # Reset PID integral and previous error when patient changes
        print(f"Step {step}: Changed to patient type {env.patient_type} "
              f"(k10={env.k10:.2f}, k12={env.k12:.2f}, k21={env.k21:.2f})")
    
    # PID control
    action = pid.compute(obs[0])
    
    # Environment step
    obs, reward, done, info = env.step(action)
    
    # Record data
    concentrations.append(info["central_concentration"])
    actions.append(action[0])
    targets.append(pid.target)
    times.append(step * env.dt)
    patient_types.append(env.patient_type)

# Plot results
plt.figure(figsize=(14, 12))

# Concentration tracking plot
plt.subplot(4, 1, 1)
plt.plot(times, concentrations, 'b-', label='Actual Concentration', linewidth=2)
plt.plot(times, targets, 'r--', label='Target Concentration', linewidth=2)

# Add vertical lines and background colors for patient changes
for i in range(0, num_steps, patient_change_interval):
    plt.axvline(x=i * env.dt, color='gray', linestyle=':', alpha=0.7)
    # Add background color for patient types
    if i + patient_change_interval <= num_steps:
        plt.axvspan(i * env.dt, (i + patient_change_interval) * env.dt, 
                   alpha=0.1, color='orange' if patient_types[i] == 0 else 'green')

plt.ylabel('Drug Concentration')
plt.title('PID Control - Drug Concentration Tracking with Different Patients')
plt.legend()
plt.grid(True)

# Control action plot
plt.subplot(4, 1, 2)
plt.plot(times, actions, 'g-', label='Control Action', linewidth=2)

# Add vertical lines for patient changes
for i in range(0, num_steps, patient_change_interval):
    plt.axvline(x=i * env.dt, color='gray', linestyle=':', alpha=0.7)

plt.ylabel('Infusion Rate')
plt.legend()
plt.grid(True)

# Error plot
plt.subplot(4, 1, 3)
errors = [target - conc for target, conc in zip(targets, concentrations)]
plt.plot(times, errors, 'm-', label='Tracking Error', linewidth=2)

# Add vertical lines for patient changes
for i in range(0, num_steps, patient_change_interval):
    plt.axvline(x=i * env.dt, color='gray', linestyle=':', alpha=0.7)

plt.ylabel('Error')
plt.legend()
plt.grid(True)

# Patient type plot
plt.subplot(4, 1, 4)
plt.plot(times, patient_types, 'k-', label='Patient Type', linewidth=2)
plt.yticks([0, 1], ['Slow Metabolism', 'Fast Metabolism'])

# Add vertical lines for patient changes
for i in range(0, num_steps, patient_change_interval):
    plt.axvline(x=i * env.dt, color='gray', linestyle=':', alpha=0.7)

plt.ylabel('Patient Type')
plt.xlabel('Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Performance statistics for each patient type
slow_indices = [i for i, pt in enumerate(patient_types) if pt == 0]
fast_indices = [i for i, pt in enumerate(patient_types) if pt == 1]

if slow_indices:
    slow_errors = [errors[i] for i in slow_indices[-100:]]
    slow_concentrations = [concentrations[i] for i in slow_indices[-100:]]
    print(f"\nSlow Metabolism Patient Performance:")
    print(f"  Final MAE: {np.mean(np.abs(slow_errors)):.4f}")
    print(f"  Concentration Mean: {np.mean(slow_concentrations):.4f}")
    print(f"  Concentration Std: {np.std(slow_concentrations):.4f}")

if fast_indices:
    fast_errors = [errors[i] for i in fast_indices[-100:]]
    fast_concentrations = [concentrations[i] for i in fast_indices[-100:]]
    print(f"\nFast Metabolism Patient Performance:")
    print(f"  Final MAE: {np.mean(np.abs(fast_errors)):.4f}")
    print(f"  Concentration Mean: {np.mean(fast_concentrations):.4f}")
    print(f"  Concentration Std: {np.std(fast_concentrations):.4f}")

# Overall performance
final_error = np.mean(np.abs(errors[-100:]))
print(f"\nOverall Performance:")
print(f"Final MAE: {final_error:.4f}")
print(f"Concentration Mean: {np.mean(concentrations):.4f}")
print(f"Concentration Std: {np.std(concentrations):.4f}")