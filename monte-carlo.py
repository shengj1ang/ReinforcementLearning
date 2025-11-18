import random

STATES = [1, 2, 3, 4, 5]


def step(state):
    """
    随机向左右移动一步
    返回：next_state, reward, done
    """
    # 随机走左或右
    if random.random() < 0.5:
        next_state = state - 1
    else:
        next_state = state + 1

    # 终止判断
    if next_state == 6:      # 右终点
        return next_state, 1.0, True
    elif next_state == 0:    # 左终点
        return next_state, 0.0, True
    else:
        return next_state, 0.0, False  # 中间状态无奖励


def generate_episode(start_state=3):
    """
    生成一条完整 episode：
    返回：[(s0, r1), (s1, r2), ...]
    """
    episode = []
    state = start_state
    done = False

    while not done:
        next_state, reward, done = step(state)
        episode.append((state, reward))
        state = next_state

    return episode


def mc_prediction(num_episodes=10000):
    # 累加回报(sum) 和 访问次数(count)
    returns_sum = {s: 0.0 for s in STATES}
    returns_count = {s: 0   for s in STATES}

    # 初始估计 V(s)
    V = {s: 0.0 for s in STATES}

    for _ in range(num_episodes):

        episode = generate_episode()

        # 记录首次访问的位置
        first_visit_time = {}
        for t, (s, r) in enumerate(episode):
            if s not in first_visit_time:
                first_visit_time[s] = t

        # 利用首次出现之后的奖励更新 V
        for s, t in first_visit_time.items():
            G = 0.0
            # γ = 1，不需要折扣因子变量
            for k in range(t, len(episode)):
                _, r = episode[k]
                G += r  # 因为 gamma = 1

            returns_sum[s] += G
            returns_count[s] += 1
            V[s] = returns_sum[s] / returns_count[s]

    return V


if __name__ == "__main__":
    V = mc_prediction(5000)
    for s in STATES:
        print(f"V({s}) ≈ {V[s]:.3f}")