from collections import defaultdict

OPTIMISTIC_INIT = 30.0

def make_q():
    return defaultdict(lambda: OPTIMISTIC_INIT)

def epsilon_greedy(Q, state, epsilon, n_actions, rng):
    if rng.random() < epsilon:
        return rng.randrange(n_actions)
    return greedy_action(Q, state, n_actions, rng)


def greedy_action(Q, state, n_actions, rng):
    qvals = [Q[(state, a)] for a in range(n_actions)]
    best_val = max(qvals)
    best_actions = [a for a, q in enumerate(qvals) if q == best_val]
    return rng.choice(best_actions)  # break ties randomly


def run_nstep_sarsa_episode(env, Q, n, alpha, gamma, epsilon, rng):
    n_actions = env.action_space.n
    states, actions, rewards = {}, {}, {}

    state, _ = env.reset()
    states[0] = state
    actions[0] = epsilon_greedy(Q, state, epsilon, n_actions, rng)

    T = float('inf')
    was_truncated = False
    t = 0
    ep_return = 0.0

    while True:
        if t < T:
            next_state, reward, terminated, truncated, _ = env.step(actions[t])
            rewards[t + 1] = reward
            states[t + 1] = next_state
            ep_return += reward

            if terminated:
                T = t + 1
            else:
                actions[t + 1] = epsilon_greedy(Q, next_state, epsilon, n_actions, rng)
                if truncated:
                    T = t + 1
                    was_truncated = True

        tau = t - n + 1
        if tau >= 0:
            end = min(tau + n, T)
            G = 0.0
            for i in range(tau + 1, end + 1):
                G += (gamma ** (i - tau - 1)) * rewards[i]
            if end < T or was_truncated:
                G += (gamma ** (end - tau)) * Q[(states[end], actions[end])]

            s_tau, a_tau = states[tau], actions[tau]
            Q[(s_tau, a_tau)] += alpha * (G - Q[(s_tau, a_tau)])

        if tau == T - 1:
            break
        t += 1

    return ep_return


def run_qlearning_episode(env, Q, alpha, gamma, epsilon, rng):

    n_actions = env.action_space.n
    state, _ = env.reset()
    ep_return = 0.0

    while True:
        action = epsilon_greedy(Q, state, epsilon, n_actions, rng)
        next_state, reward, terminated, truncated, _ = env.step(action)
        ep_return += reward

        best_next = max(Q[(next_state, a)] for a in range(n_actions))
        target = reward if terminated else reward + gamma * best_next
        Q[(state, action)] += alpha * (target - Q[(state, action)])

        state = next_state
        if terminated or truncated:
            break

    return ep_return


def greedy_eval(env, Q, n_episodes, rng):
    n_actions = env.action_space.n
    total = 0.0
    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_return = 0.0
        while True:
            action = greedy_action(Q, state, n_actions, rng)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            if terminated or truncated:
                break
        total += ep_return
    return total / n_episodes
