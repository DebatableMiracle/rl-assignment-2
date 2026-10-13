import numpy as np
from env import BridgeAndFerryEnv
from agents import ALGOS, evaluate_greedy, classify_greedy_route

FERRY_RETURN = 37.0  
N_BRIDGE_CELLS = 5


def bridge_expected_return(p):

    full_success = (1 - p) ** N_BRIDGE_CELLS * 40
    fail_term = 0.0
    for k in range(1, N_BRIDGE_CELLS + 1):
        steps_before_fail = 3 + k - 1
        R_k = -steps_before_fail - 101
        prob_k = (1 - p) ** (k - 1) * p
        fail_term += prob_k * R_k
    return full_success + fail_term


def solve_p_star(tol=1e-8):

    def f(p):
        return bridge_expected_return(p) - FERRY_RETURN

    lo, hi = 0.0, 1.0
    assert f(lo) > 0 and f(hi) < 0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run_pwind_sweep(pwind_values, n_train_episodes, n_seeds, epsilon=0.1,
                     alpha=0.1, gamma=1.0, max_steps=500):

    results = {algo: {} for algo in ALGOS}
    for algo, fn in ALGOS.items():
        for p in pwind_values:
            bridge_count = 0
            for seed in range(n_seeds):
                env = BridgeAndFerryEnv(p_wind=p, sigma=0.0)
                Q, _, _ = fn(env, n_train_episodes, seed=seed, epsilon=epsilon,
                             alpha=alpha, gamma=gamma, max_steps=max_steps, track_visits=False)
                route = classify_greedy_route(env, Q, seed=1000 + seed, max_steps=max_steps)
                if route == "bridge":
                    bridge_count += 1
            results[algo][p] = bridge_count / n_seeds
    return results


def run_sigma_sweep(sigma_values, n_train_episodes, n_seeds, p_wind=0.02,
                     epsilon=0.1, alpha=0.1, gamma=1.0, max_steps=500, eval_episodes=500):

    results = {algo: {} for algo in ALGOS}
    for algo, fn in ALGOS.items():
        for sigma in sigma_values:
            ferry_count = 0
            seed_returns = []
            for seed in range(n_seeds):
                env = BridgeAndFerryEnv(p_wind=p_wind, sigma=sigma)
                Q, _, _ = fn(env, n_train_episodes, seed=seed, epsilon=epsilon,
                             alpha=alpha, gamma=gamma, max_steps=max_steps, track_visits=False)
                route = classify_greedy_route(env, Q, seed=2000 + seed, max_steps=max_steps)
                if route == "ferry":
                    ferry_count += 1
                mean_ret, _ = evaluate_greedy(env, Q, n_episodes=eval_episodes,
                                               seed=3000 + seed, max_steps=max_steps)
                seed_returns.append(mean_ret)
            seed_returns = np.array(seed_returns)
            n_valid = int(np.sum(~np.isnan(seed_returns)))
            mean_ret = float(np.nanmean(seed_returns)) if n_valid > 0 else float("nan")
            se = float(np.nanstd(seed_returns, ddof=1) / np.sqrt(n_valid)) if n_valid > 1 else float("nan")
            results[algo][sigma] = {
                "ferry_fraction": ferry_count / n_seeds,
                "mean_return": mean_ret,
                "ci95": 1.96 * se if n_valid > 1 else float("nan"),
                "n_non_terminating_seeds": n_seeds - n_valid,
            }
    return results