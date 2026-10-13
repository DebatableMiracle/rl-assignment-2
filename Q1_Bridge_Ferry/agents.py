import numpy as np

Q_INIT = 49


def epsilon_greedy(Q, s, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(Q.shape[1])

    q_values = Q[s]
    max_q = np.max(q_values)

    # Random tie-breaking among all greedy actions
    greedy_actions = np.flatnonzero(np.isclose(q_values, max_q))

    return rng.choice(greedy_actions)
    

def greedy_action(Q, s):
    return np.argmax(Q[s])


def sarsa(env, n_episodes, seed, epsilon=0.1, alpha=0.1, gamma=1.0, max_steps=500,
          track_visits=True, q_init=Q_INIT):
    rng = np.random.default_rng(seed)
    Q = np.full((env.n_states, env.action_space.n), q_init, dtype=float)
    online_returns = np.zeros(n_episodes)
    visits = np.zeros(env.n_states) if track_visits else None

    for ep in range(n_episodes):
        s, _ = env.reset(seed=int(rng.integers(1_000_000)))
        if track_visits:
            visits[s] += 1
        a = epsilon_greedy(Q, s, epsilon, rng)
        ep_return = 0.0

        for t in range(max_steps):
            s_next, r, terminated, truncated, _ = env.step(a)
            if track_visits:
                visits[s_next] += 1
            a_next = epsilon_greedy(Q, s_next, epsilon, rng)
            td_target = r + gamma * Q[s_next, a_next] * (0 if terminated else 1)
            Q[s, a] += alpha * (td_target - Q[s, a])
            s, a = s_next, a_next
            ep_return += r
            if terminated or truncated:
                break

        online_returns[ep] = ep_return

    return Q, online_returns, visits


def expected_sarsa(env, n_episodes, seed, epsilon=0.1, alpha=0.1, gamma=1.0, max_steps=500,
                    track_visits=True, q_init=Q_INIT):
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n
    Q = np.full((env.n_states, n_actions), q_init, dtype=float)
    online_returns = np.zeros(n_episodes)
    visits = np.zeros(env.n_states) if track_visits else None

    for ep in range(n_episodes):
        s, _ = env.reset(seed=int(rng.integers(1_000_000)))
        if track_visits:
            visits[s] += 1
        a = epsilon_greedy(Q, s, epsilon, rng)
        ep_return = 0.0

        for t in range(max_steps):
            s_next, r, terminated, truncated, _ = env.step(a)
            if track_visits:
                visits[s_next] += 1

            probs = np.full(n_actions, epsilon / n_actions)
            probs[np.argmax(Q[s_next])] += 1 - epsilon
            expected_q = np.dot(probs, Q[s_next])
            td_target = r + gamma * expected_q * (0 if terminated else 1)
            Q[s, a] += alpha * (td_target - Q[s, a])

            a_next = epsilon_greedy(Q, s_next, epsilon, rng)
            s, a = s_next, a_next
            ep_return += r
            if terminated or truncated:
                break

        online_returns[ep] = ep_return

    return Q, online_returns, visits


def q_learning(env, n_episodes, seed, epsilon=0.1, alpha=0.1, gamma=1.0, max_steps=500,
               track_visits=True, q_init=Q_INIT):
    rng = np.random.default_rng(seed)
    Q = np.full((env.n_states, env.action_space.n), q_init, dtype=float)
    online_returns = np.zeros(n_episodes)
    visits = np.zeros(env.n_states) if track_visits else None

    for ep in range(n_episodes):
        s, _ = env.reset(seed=int(rng.integers(1_000_000)))
        if track_visits:
            visits[s] += 1
        ep_return = 0.0

        for t in range(max_steps):
            a = epsilon_greedy(Q, s, epsilon, rng)
            s_next, r, terminated, truncated, _ = env.step(a)
            if track_visits:
                visits[s_next] += 1
            td_target = r + gamma * np.max(Q[s_next]) * (0 if terminated else 1)
            Q[s, a] += alpha * (td_target - Q[s, a])
            s = s_next
            ep_return += r
            if terminated or truncated:
                break

        online_returns[ep] = ep_return

    return Q, online_returns, visits


ALGOS = {"sarsa": sarsa, "expected_sarsa": expected_sarsa, "qlearning": q_learning}


def evaluate_greedy(env, Q, n_episodes=500, seed=0, max_steps=500):

    rng = np.random.default_rng(seed)
    returns, non_terminating = [], 0

    for ep in range(n_episodes):
        s, _ = env.reset(seed=int(rng.integers(1_000_000)))
        ep_return = 0.0
        terminated = False

        for t in range(max_steps):
            a = greedy_action(Q, s)
            s, r, terminated, truncated, _ = env.step(a)
            ep_return += r
            if terminated or truncated:
                break

        if terminated:
            returns.append(ep_return)
        else:
            non_terminating += 1

    mean_return = float(np.mean(returns)) if returns else float("nan")
    return mean_return, non_terminating


def classify_greedy_route(env, Q, seed=0, max_steps=500):
    from env import GOAL, FERRY
    rng = np.random.default_rng(seed)
    s, _ = env.reset(seed=int(rng.integers(1_000_000)))
    path = [env.state_to_rc(s)]
    terminated = False

    for t in range(max_steps):
        a = greedy_action(Q, s)
        s, r, terminated, truncated, _ = env.step(a)
        path.append(env.state_to_rc(s))
        if terminated or truncated:
            break

    if not terminated:
        return "non_terminating"
    end = path[-1]
    if end == FERRY:
        return "ferry"
    if end == GOAL:
        rows_visited = {rc[0] for rc in path}
        if rows_visited <= {3}:
            return "bridge"
        return "detour"
    return "chasm"