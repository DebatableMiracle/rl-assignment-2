import numpy as np
import matplotlib.pyplot as plt
from env import N_ROWS, N_COLS


def moving_average(x, w):
    """x: (n_seeds, n_episodes) -> (n_seeds, n_episodes - w + 1), mean over each window of w episodes."""
    c = np.cumsum(np.insert(x, 0, 0.0, axis=1), axis=1)
    return (c[:, w:] - c[:, :-w]) / w


def plot_learning_curves(online_by_algo, save_path, window=1000):
    """online_by_algo: {algo_name: array of shape (n_seeds, n_episodes)}"""
    plt.figure(figsize=(8, 5))
    for name, online_all in online_by_algo.items():
        sm = moving_average(online_all, window)              # smooth each seed first
        mean = sm.mean(axis=0)
        se = sm.std(axis=0, ddof=1) / np.sqrt(sm.shape[0])   # SE across seeds
        x = np.arange(window, online_all.shape[1] + 1)       # episode at the end of each window
        plt.plot(x, mean, label=name)
        plt.fill_between(x, mean - se, mean + se, alpha=0.25)
    plt.xlabel("Training episode")
    plt.ylabel(f"Online return ({window}-episode moving average)")
    plt.title("Online (during-training) return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_greedy_bar(greedy_means, greedy_sems, save_path):
    names = list(greedy_means.keys())
    means = [greedy_means[n] for n in names]
    sems = [greedy_sems[n] for n in names]
    plt.figure(figsize=(6, 5))
    plt.bar(names, means, yerr=sems, capsize=6)
    plt.ylabel("Mean greedy return")
    plt.title("Final greedy performance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_heatmap(visits, title, save_path):
    grid = visits.reshape(N_ROWS, N_COLS)
    plt.figure(figsize=(9, 5))
    plt.imshow(grid, cmap="viridis")
    plt.colorbar(label="Avg visits / episode")
    plt.title(title)
    plt.xlabel("col")
    plt.ylabel("row")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pwind_sweep(sweep_results, p_star, save_path):
    plt.figure(figsize=(7, 5))
    for algo, per_p in sweep_results.items():
        xs = sorted(per_p.keys())
        ys = [per_p[p] for p in xs]
        plt.plot(xs, ys, marker="o", label=algo)
    plt.axvline(p_star, color="k", linestyle="--", label=f"p* = {p_star:.4f}")
    plt.xlabel("p_wind")
    plt.ylabel("Fraction of seeds choosing bridge")
    plt.title("Bridge switch-over vs p_wind")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_sigma_return(sigma_results, save_path):
    plt.figure(figsize=(7, 5))
    for algo, per_sigma in sigma_results.items():
        xs = sorted(per_sigma.keys())
        ys = [per_sigma[s]["mean_return"] for s in xs]
        errs = [per_sigma[s]["ci95"] for s in xs]
        plt.errorbar(xs, ys, yerr=errs, marker="o", capsize=4, label=algo)
    plt.xlabel("Ferry reward std dev (sigma)")
    plt.ylabel("Mean greedy return (95% CI)")
    plt.title("Effect of ferry reward variance on greedy return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_sigma_policy(sigma_results, save_path):
    plt.figure(figsize=(7, 5))
    for algo, per_sigma in sigma_results.items():
        xs = sorted(per_sigma.keys())
        ys = [per_sigma[s]["ferry_fraction"] for s in xs]
        plt.plot(xs, ys, marker="o", label=algo)
    plt.xlabel("Ferry reward std dev (sigma)")
    plt.ylabel("Fraction of seeds choosing ferry")
    plt.title("Effect of ferry reward variance on policy choice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()