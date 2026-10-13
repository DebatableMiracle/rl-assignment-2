

import json
import os
import random

import numpy as np

from env import SupplyRunEnv
from nstep_agents import make_q, run_nstep_sarsa_episode, run_qlearning_episode, greedy_eval
import visualize

# ---- settings ----
N_EPISODES = 8000
N_SEEDS = 10
GAMMA = 0.99          
EPSILON = 0.1
MAX_STEPS = 200
N_VALUES = [1, 2, 4, 8, 16, 24, 32]

EVAL_EVERY = 200          # how often (in episodes) we check greedy performance during training
EVAL_EPISODES_DURING = 100  # cheap eval used for the learning-curve checkpoints
EVAL_EPISODES_FINAL = 500  

ALPHA = 0.1

def train_nstep_sarsa(n, seed):
    env = SupplyRunEnv(max_steps=MAX_STEPS)
    eval_env = SupplyRunEnv(max_steps=MAX_STEPS)
    rng = random.Random(seed)
    eval_rng = random.Random(seed + 10_000)
    Q = make_q()

    checkpoints, curve = [], []
    for ep in range(1, N_EPISODES + 1):
        run_nstep_sarsa_episode(env, Q, n, ALPHA, GAMMA, EPSILON, rng)
        if ep % EVAL_EVERY == 0 or ep == N_EPISODES:
            g = greedy_eval(eval_env, Q, EVAL_EPISODES_DURING, eval_rng)
            checkpoints.append(ep)
            curve.append(g)

    final_greedy = greedy_eval(eval_env, Q, EVAL_EPISODES_FINAL, eval_rng)
    return checkpoints, curve, final_greedy


def train_qlearning(seed):
    env = SupplyRunEnv(max_steps=MAX_STEPS)
    eval_env = SupplyRunEnv(max_steps=MAX_STEPS)
    rng = random.Random(seed)
    eval_rng = random.Random(seed + 10_000)
    Q = make_q()

    checkpoints, curve = [], []
    for ep in range(1, N_EPISODES + 1):
        run_qlearning_episode(env, Q, ALPHA, GAMMA, EPSILON, rng)
        if ep % EVAL_EVERY == 0 or ep == N_EPISODES:
            g = greedy_eval(eval_env, Q, EVAL_EPISODES_DURING, eval_rng)
            checkpoints.append(ep)
            curve.append(g)

    final_greedy = greedy_eval(eval_env, Q, EVAL_EPISODES_FINAL, eval_rng)
    return checkpoints, curve, final_greedy


def main():
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print(f"n-step SARSA sweep, n = {N_VALUES}, {N_SEEDS} seeds, {N_EPISODES} episodes each")
    print("-" * 60)

    nstep_curve_data = {}
    final_greedy_returns = {n: [] for n in N_VALUES}

    for n in N_VALUES:
        all_curves = []
        cps = None
        for seed in range(N_SEEDS):
            cps, curve, g = train_nstep_sarsa(n, seed)
            all_curves.append(curve)
            final_greedy_returns[n].append(g)
        nstep_curve_data[n] = {"checkpoints": cps, "values": np.array(all_curves)}
        mean_g = np.mean(final_greedy_returns[n])
        std_g = np.std(final_greedy_returns[n])
        print(f"n={n:>3d}  greedy return  mean={mean_g:7.2f}  std={std_g:6.2f}")

    print("-" * 60)
    print(f"Q-learning baseline, {N_SEEDS} seeds, {N_EPISODES} episodes")

    q_all_curves = []
    q_final = []
    q_cps = None
    for seed in range(N_SEEDS):
        q_cps, curve, g = train_qlearning(seed)
        q_all_curves.append(curve)
        q_final.append(g)
    q_curve_data = {"checkpoints": q_cps, "values": np.array(q_all_curves)}
    q_mean, q_std = float(np.mean(q_final)), float(np.std(q_final))
    print(f"Q-learning  greedy return  mean={q_mean:7.2f}  std={q_std:6.2f}")

    best_n = max(N_VALUES, key=lambda n: np.mean(final_greedy_returns[n]))
    print("-" * 60)
    print(f"best n by final greedy return: n={best_n}")

    results = {
        "settings": {
            "episodes": N_EPISODES,
            "seeds": N_SEEDS,
            "gamma": GAMMA,
            "alpha": ALPHA,
            "epsilon": EPSILON,
            "max_steps": MAX_STEPS,
        },
        "n_values": N_VALUES,
        "nstep_sarsa": {
            str(n): {
                "mean_greedy_return": float(np.mean(final_greedy_returns[n])),
                "std_greedy_return": float(np.std(final_greedy_returns[n])),
                "per_seed_greedy_returns": final_greedy_returns[n],
            }
            for n in N_VALUES
        },
        "q_learning": {
            "mean_greedy_return": q_mean,
            "std_greedy_return": q_std,
            "per_seed_greedy_returns": q_final,
        },
        "best_n": best_n,
    }

    with open('results/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    visualize.plot_learning_curves(nstep_curve_data, q_curve_data, N_VALUES,
                                    'plots/learning_curves.png')
    visualize.plot_n_sweep(N_VALUES, final_greedy_returns, q_mean, q_std,
                            'plots/n_sweep.png')

    print("-" * 60)
    print("saved plots/learning_curves.png, plots/n_sweep.png, results/results.json")


if __name__ == '__main__':
    main()
