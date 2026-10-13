import os
import numpy as np

from env import BridgeAndFerryEnv
from agents import ALGOS, evaluate_greedy
from analysis import solve_p_star, run_pwind_sweep, run_sigma_sweep
from visualize import (plot_learning_curves, plot_greedy_bar, plot_heatmap,
                        plot_pwind_sweep, plot_sigma_return, plot_sigma_policy)

N_TRAIN_EPISODES = 100000    
SWEEP_TRAIN_EPISODES = 10000   
N_SEEDS = 10
EPSILON, ALPHA, GAMMA, MAX_STEPS = 0.1, 0.1, 1.0, 500
PLOTS_DIR = "plots"


def stage_iii():
    """Train all three algorithms at p_wind=0; report online + greedy return
    and save a state-visitation heatmap per algorithm."""
    online_curves, greedy_means, greedy_sems = {}, {}, {}
    n_states = BridgeAndFerryEnv().n_states

    for algo, fn in ALGOS.items():
        online_all = np.zeros((N_SEEDS, N_TRAIN_EPISODES))
        greedy_all = []
        visits_sum = np.zeros(n_states)
        non_term_total = 0

        for seed in range(N_SEEDS):
            env = BridgeAndFerryEnv(p_wind=0.0, sigma=0.0)
            Q, online_returns, visits = fn(env, N_TRAIN_EPISODES, seed=seed, epsilon=EPSILON,
                                            alpha=ALPHA, gamma=GAMMA, max_steps=MAX_STEPS)
            online_all[seed] = online_returns
            visits_sum += visits / N_TRAIN_EPISODES
            mean_ret, non_term = evaluate_greedy(env, Q, n_episodes=500, seed=1000 + seed, max_steps=MAX_STEPS)
            greedy_all.append(mean_ret)
            non_term_total += non_term

        mean_curve = online_all.mean(axis=0)
        se_curve = online_all.std(axis=0, ddof=1) / np.sqrt(N_SEEDS)
        online_curves[algo] = (mean_curve, se_curve)

        greedy_all = np.array(greedy_all)
        n_valid = int(np.sum(~np.isnan(greedy_all)))
        greedy_means[algo] = float(np.nanmean(greedy_all)) if n_valid > 0 else float("nan")
        greedy_sems[algo] = float(np.nanstd(greedy_all, ddof=1) / np.sqrt(n_valid)) if n_valid > 1 else float("nan")
        print(f"{algo}: greedy return = {greedy_means[algo]:.2f} +- {greedy_sems[algo]:.2f}, "
              f"non-terminating episodes (across all seeds' evals) = {non_term_total}, "
              f"fully non-terminating seeds = {N_SEEDS - n_valid}")

        plot_heatmap(visits_sum / N_SEEDS, f"{algo} state visitation", f"{PLOTS_DIR}/heatmap_{algo}.png")

    plot_learning_curves(online_curves, f"{PLOTS_DIR}/learning_curves.png")
    plot_greedy_bar(greedy_means, greedy_sems, f"{PLOTS_DIR}/greedy_comparison.png")


def stage_iv():
    """Analytic p* + empirical p_wind switch-over sweep."""
    p_star = solve_p_star()
    print(f"Analytic p* = {p_star:.5f}")

    pwind_values = [0, 0.002, 0.004, 0.008, 0.016, 0.032]
    pwind_sweep = run_pwind_sweep(pwind_values, n_train_episodes=SWEEP_TRAIN_EPISODES, n_seeds=N_SEEDS,
                                   epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA, max_steps=MAX_STEPS)
    for algo in ALGOS:
        print(algo, {p: round(f, 2) for p, f in pwind_sweep[algo].items()})
    plot_pwind_sweep(pwind_sweep, p_star, f"{PLOTS_DIR}/pwind_switchover.png")
    return p_star, pwind_sweep


def stage_v():
    """Ferry reward variance (sigma) sweep."""
    sigma_values = [0, 5, 15, 30]
    sigma_res = run_sigma_sweep(sigma_values, n_train_episodes=SWEEP_TRAIN_EPISODES, n_seeds=N_SEEDS,
                                 p_wind=0.02, epsilon=EPSILON, alpha=ALPHA, gamma=GAMMA,
                                 max_steps=MAX_STEPS, eval_episodes=500)
    for algo in ALGOS:
        for s in sigma_values:
            r = sigma_res[algo][s]
            print(f"{algo} sigma={s}: ferry_frac={r['ferry_fraction']:.2f} "
                  f"mean_return={r['mean_return']:.2f} +- {r['ci95']:.2f} "
                  f"(non-terminating seeds: {r['n_non_terminating_seeds']})")
    plot_sigma_return(sigma_res, f"{PLOTS_DIR}/sigma_return.png")
    plot_sigma_policy(sigma_res, f"{PLOTS_DIR}/sigma_policy.png")
    return sigma_res


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("=== (iii) Main comparison (p_wind=0) ===")
    stage_iii()
    print("\n=== (iv) Bridge threshold ===")
    stage_iv()
    print("\n=== (v) Ferry reward variance ===")
    stage_v()
    print("\nDone. 8 plots saved in", PLOTS_DIR)


if __name__ == "__main__":
    main()