import sys
import numpy as np
import matplotlib.pyplot as plt
from env import BridgeAndFerryEnv, is_chasm, FERRY, GOAL
from agents import sarsa, expected_sarsa, q_learning

q_init   = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
episodes = int(sys.argv[2])   if len(sys.argv) > 2 else 20000
seed     = int(sys.argv[3])   if len(sys.argv) > 3 else 0

ALGOS = [("sarsa", sarsa), ("expected_sarsa", expected_sarsa), ("qlearning", q_learning)]
ACTS  = ["Up", "Down", "Left", "Right"]
ARROW = ["↑", "↓", "←", "→"]
VMIN, VMAX = -50, 50          # colour clip; chasm-bound actions (~-101) just saturate


def valid(r, c):
    return not is_chasm(r, c) and (r, c) not in (FERRY, GOAL)


def build(env, Q):
    """Per-action Q grids, V = max_a Q, and greedy-arrow labels. NaN on chasm/terminal cells."""
    R, C = env.n_rows, env.n_cols
    g = np.full((4, R, C), np.nan)
    V = np.full((R, C), np.nan)
    arrows = [[""] * C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            if valid(r, c):
                q = Q[env.rc_to_state(r, c)]
                g[:, r, c] = q
                V[r, c] = q.max()
                tie = np.sum(np.isclose(q, q.max())) > 1
                arrows[r][c] = ARROW[int(np.argmax(q))] + ("?" if tie else "")
    return g, V, arrows


def mark(ax, color="black"):
    for (r, c), s in ((FERRY, "F"), (GOAL, "G")):
        ax.text(c, r, s, ha="center", va="center", fontsize=11, fontweight="bold", color=color)


def heat(ax, grid, title, annotate=True):
    cm = plt.get_cmap("RdYlGn").copy()
    cm.set_bad("lightgrey")
    im = ax.imshow(grid, cmap=cm, vmin=VMIN, vmax=VMAX)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    ax.tick_params(labelsize=6)
    if annotate:
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if not np.isnan(grid[r, c]):
                    ax.text(c, r, f"{grid[r, c]:.0f}", ha="center", va="center", fontsize=6)
    mark(ax)
    return im


all_visits = {}
for name, fn in ALGOS:
    env = BridgeAndFerryEnv(p_wind=0.0, sigma=0.0)
    Q, _, visits = fn(env, episodes, seed=seed, q_init=q_init)
    all_visits[name] = visits.reshape(env.n_rows, env.n_cols)
    g, V, arrows = build(env, Q)

    fig, axes = plt.subplots(2, 3, figsize=(19, 8), constrained_layout=True)
    for a in range(4):
        im = heat(axes.flat[a], g[a], f"Q(s, {ACTS[a]})")

    ax = axes.flat[4]
    heat(ax, V, "max_a Q(s,a) with greedy action (? = tie)", annotate=False)
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            if arrows[r][c]:
                ax.text(c, r, f"{arrows[r][c]}\n{V[r, c]:.0f}", ha="center", va="center", fontsize=7)

    ax = axes.flat[5]                      # Q along the bridge row
    cols = np.arange(env.n_cols)
    for a in range(4):
        ax.plot(cols, g[a, 3, :], marker="o", label=ACTS[a])
    ax.axvspan(2.5, 7.5, color="orange", alpha=0.12, label="bridge cells")
    ax.axhline(37, color="gray", ls="--", lw=1, label="ferry value from S (37)")
    ax.set_ylim(-110, 60)
    ax.set_xticks(cols)
    ax.set_xlabel("column on row 3")
    ax.set_ylabel("Q")
    ax.set_title("Q along the bridge row (row 3)")
    ax.legend(fontsize=7)

    fig.colorbar(im, ax=list(axes.flat[:5]), shrink=0.6,
                 label=f"Q (colour clipped to [{VMIN}, {VMAX}])")
    fig.suptitle(f"{name} | q_init={q_init:g} | {episodes} episodes | seed {seed}")
    fig.savefig(f"qvalues_{name}_init{q_init:g}_{episodes}.png", dpi=150)

# where did the agent actually go? (explains why values never propagate)
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), constrained_layout=True)
for ax, (name, v) in zip(axes, all_visits.items()):
    im = ax.imshow(np.log10(1 + v), cmap="viridis")
    ax.set_title(f"{name}: log10(1 + visits)")
    mark(ax, color="red")
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle(f"State visits during training | q_init={q_init:g} | {episodes} episodes | seed {seed}")
fig.savefig(f"visits_init{q_init:g}_{episodes}.png", dpi=150)
plt.show()