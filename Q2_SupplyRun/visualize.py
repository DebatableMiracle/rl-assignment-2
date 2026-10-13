import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(nstep_curves, q_curve, n_values, path):
    """nstep_curves[n] = {"checkpoints": [...], "values": array (seeds x checkpoints)}"""
    plt.figure(figsize=(9, 6))

    for n in n_values:
        cps = nstep_curves[n]["checkpoints"]
        vals = nstep_curves[n]["values"]
        mean = vals.mean(axis=0)
        plt.plot(cps, mean, label=f"n={n}")

    q_cps = q_curve["checkpoints"]
    q_mean = q_curve["values"].mean(axis=0)
    plt.plot(q_cps, q_mean, label="Q-learning", color="black", linestyle="--", linewidth=2)

    plt.xlabel("training episodes")
    plt.ylabel("greedy return (avg over eval episodes)")
    plt.title("SupplyRun: greedy return over the course of training")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_n_sweep(n_values, greedy_returns, q_mean, q_std, path):
    """greedy_returns[n] = list of final greedy returns, one per seed"""
    means = [np.mean(greedy_returns[n]) for n in n_values]
    stds = [np.std(greedy_returns[n]) for n in n_values]

    plt.figure(figsize=(8, 5))
    plt.errorbar(n_values, means, yerr=stds, marker='o', capsize=4,
                 label='n-step SARSA', color='tab:blue')

    plt.axhline(q_mean, color='black', linestyle='--', label=f'Q-learning (mean={q_mean:.1f})')
    plt.fill_between([min(n_values), max(n_values)], q_mean - q_std, q_mean + q_std,
                      color='gray', alpha=0.15)

    plt.xlabel("n")
    plt.ylabel("final greedy return (mean ± std over seeds)")
    plt.title("SupplyRun: choosing n for n-step SARSA")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
