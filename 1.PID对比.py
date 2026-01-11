import numpy as np
import matplotlib.pyplot as plt

# =============================
# 1. Same environment as PPO
# =============================
class DrugVesselEnv:
    def __init__(self):
        self.target = 1.0
        self.decay = 0.9
        self.state = None
        self.action_low = 0.0
        self.action_high = 2.0

    def reset(self):
        self.state = np.array([0.5], dtype=np.float32)
        return self.state

    def step(self, action):
        # clip action to [0, 2]
        action = np.clip(action, self.action_low, self.action_high)
        injection = float(action)

        conc = self.decay * self.state + 0.1 * injection
        self.state = conc.astype(np.float32)

        error = conc[0] - self.target
        # 用和 PPO 一样的 reward 形式（这里只是为了对比，不一定要用）
        reward = -(6.0 * (error ** 2)) - 0.01 * (injection ** 2)

        done = False
        info = {}
        return self.state, reward, done, info


# =============================
# 2. PID controller
# =============================
class PIDController:
    def __init__(self, Kp, Ki, Kd, dt=1.0, u_min=0.0, u_max=2.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max

        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, error):
        # PID formula: u = Kp*e + Ki*∫e dt + Kd*de/dt
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt

        u = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error

        # 限幅到动作范围
        u = np.clip(u, self.u_min, self.u_max)
        return u


# =============================
# 3. Simulation
# =============================
env = DrugVesselEnv()

# 你可以自己调这三个参数看响应变化
'''
	Kp 大 → 上升更快，但容易 overshoot + 振荡
	Ki 大 → 能消除稳态误差，但也会容易 overshoot
	Kd 大 → 抑制 overshoot、让曲线更平滑，但太大会变慢、噪声敏感
'''
Kp = 10.0
Ki = 1.0
Kd = 4.0
pid = PIDController(Kp, Ki, Kd, dt=1.0, u_min=0.0, u_max=2.0)

time_steps = 200
state = env.reset()
pid.reset()

conc_history = []
action_history = []

for t in range(time_steps):
    conc = state[0]
    error = env.target - conc          # 和 RL 相反，用 target - current
    u = pid.step(error)                # PID 输出剂量

    next_state, reward, done, _ = env.step(u)

    conc_history.append(conc)
    action_history.append(u)
    state = next_state

# =============================
# 4. Visualisation
# =============================

# (1) Concentration over time
plt.figure(figsize=(8, 5))
plt.plot(conc_history, label="Concentration")
plt.axhline(env.target, color="red", linestyle="--", label="Target")
plt.xlabel("Time Step")
plt.ylabel("Concentration")
plt.title("PID Control – Drug Concentration Over Time")
plt.legend()
plt.grid(True)
plt.show()

# (2) Injection (control signal) over time
plt.figure(figsize=(8, 5))
plt.plot(action_history, label="Injection (PID output)")
plt.xlabel("Time Step")
plt.ylabel("Injection dose")
plt.title("PID Control Signal Over Time")
plt.legend()
plt.grid(True)
plt.show()
