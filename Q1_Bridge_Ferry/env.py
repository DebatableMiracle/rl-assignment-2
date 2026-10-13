import numpy as np
import gymnasium as gym
from gymnasium import spaces

N_ROWS = 7
N_COLS = 11

START = (3, 0)
GOAL = (3, 10)
FERRY = (0, 5)

CHASM_ROWS = {1, 2, 4, 5}
CHASM_COLS = set(range(3, 8))  # columns 3..7
BRIDGE_ROW = 3
BRIDGE_COLS = set(range(3, 8))

STEP_COST = -1
CHASM_PENALTY = -100
GOAL_REWARD = 50
FERRY_MEAN = 45

ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # Up, Down, Left, Right
ACTION_NAMES = ["Up", "Down", "Left", "Right"]


def is_chasm(row, col):
    return row in CHASM_ROWS and col in CHASM_COLS


def is_bridge(row, col):
    return row == BRIDGE_ROW and col in BRIDGE_COLS


class BridgeAndFerryEnv(gym.Env):
    """Episodic gridworld MDP built by hand (no built-in Gymnasium env)."""

    metadata = {"render_modes": []}

    def __init__(self, p_wind=0.0, sigma=0.0):
        super().__init__()
        self.p_wind = p_wind
        self.sigma = sigma
        self.n_rows = N_ROWS
        self.n_cols = N_COLS
        self.n_states = N_ROWS * N_COLS
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(4)
        self._agent_rc = None

    # ---- helpers -----------------------------------------------------
    def rc_to_state(self, row, col):
        return row * self.n_cols + col

    def state_to_rc(self, s):
        return divmod(s, self.n_cols)

    def _in_bounds(self, row, col):
        return 0 <= row < self.n_rows and 0 <= col < self.n_cols

    # ---- Gymnasium API -------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_rc = START
        return self.rc_to_state(*START), {}

    def step(self, action):
        row, col = self._agent_rc
        dr, dc = ACTIONS[action]
        nrow, ncol = row + dr, col + dc

        # Wind override applies only when *leaving* a bridge cell.
        if is_bridge(row, col) and self.np_random.random() < self.p_wind:
            # displaced vertically, up or down with equal prob
            if self.np_random.random() < 0.5:
                nrow, ncol = row - 1, col
            else:
                nrow, ncol = row + 1, col
        # otherwise the chosen action executes normally (possibly hitting a wall)

        if not self._in_bounds(nrow, ncol):
            nrow, ncol = row, col  # wall bump, stay in place

        self._agent_rc = (nrow, ncol)
        reward = STEP_COST
        terminated = False

        if is_chasm(nrow, ncol):
            reward += CHASM_PENALTY
            terminated = True
        elif (nrow, ncol) == FERRY:
            payout = self.np_random.normal(FERRY_MEAN, self.sigma) if self.sigma > 0 else FERRY_MEAN
            reward += payout
            terminated = True
        elif (nrow, ncol) == GOAL:
            reward += GOAL_REWARD
            terminated = True

        next_state = self.rc_to_state(nrow, ncol)
        return next_state, reward, terminated, False, {}

    # ---- utilities for analysis/visualization --------------------------
    def all_states(self):
        return list(range(self.n_states))