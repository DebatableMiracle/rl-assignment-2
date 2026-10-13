import sys
import numpy as np
from env import BridgeAndFerryEnv, is_chasm, FERRY, GOAL
from agents import sarsa, expected_sarsa, q_learning, classify_greedy_route, greedy_action

q_init = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
ARROWS = "^v<>"

def grid(env, Q):
    for r in range(env.n_rows):
        line = ""
        for c in range(env.n_cols):
            if is_chasm(r, c): line += " X"
            elif (r, c) == FERRY: line += " F"
            elif (r, c) == GOAL: line += " G"
            else:
                s = env.rc_to_state(r, c)
                tie = np.sum(np.isclose(Q[s], Q[s].max())) > 1
                line += " " + ARROWS[greedy_action(Q, s)] + ("?" if tie else " ")
        print(line)

for name, fn in [("sarsa", sarsa), ("expected_sarsa", expected_sarsa), ("qlearning", q_learning)]:
    routes = []
    for seed in range(5):
        env = BridgeAndFerryEnv(p_wind=0.0, sigma=0.0)
        Q, online, _ = fn(env, 20000, seed=seed, q_init=q_init)
        routes.append(classify_greedy_route(env, Q, seed=1000 + seed))
        if seed == 0:
            print(f"\n== {name}, q_init={q_init}, seed 0 (? = tied Q) ==")
            grid(env, Q)
            print("Q at S (U,D,L,R):", np.round(Q[env.rc_to_state(3, 0)], 1))
            for c in (2, 3, 4):
                print(f"Q at (3,{c}):", np.round(Q[env.rc_to_state(3, c)], 1))
    print(name, "routes over 5 seeds:", routes)