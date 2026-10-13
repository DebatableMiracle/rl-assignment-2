

import gymnasium as gym
from gymnasium import spaces


class SupplyRunEnv(gym.Env):

    ROWS = 8
    COLS = 8
    FUEL_CAP = 15

    START = (7, 0)
    DEPOT = (0, 7)
    DROPOFF = (7, 7)
    REFUEL_CELLS = [(1, 2), (4, 4)]

    # action -> (d_row, d_col). 0 up, 1 down, 2 left, 3 right
    ACTIONS = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }

    def __init__(self, max_steps=200):
        super().__init__()
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.ROWS),
            spaces.Discrete(self.COLS),
            spaces.Discrete(2),                 # has_cargo flag
            spaces.Discrete(self.FUEL_CAP + 1),
        ))

        self._row = None
        self._col = None
        self._has_cargo = None
        self._fuel = None
        self._steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._row, self._col = self.START
        self._has_cargo = 0
        self._fuel = self.FUEL_CAP
        self._steps = 0
        return self._state(), {}

    def _state(self):
        return (self._row, self._col, self._has_cargo, self._fuel)

    def step(self, action):
        self._steps += 1

        dr, dc = self.ACTIONS[action]
        nr, nc = self._row + dr, self._col + dc

        # walking into a wall just keeps you where you are, still costs fuel
        if 0 <= nr < self.ROWS and 0 <= nc < self.COLS:
            self._row, self._col = nr, nc

        self._fuel -= 1
        pos = (self._row, self._col)

        if pos in self.REFUEL_CELLS:
            self._fuel = self.FUEL_CAP

        if pos == self.DEPOT:
            self._has_cargo = 1

        reward = 0.0
        terminated = False

        if pos == self.DROPOFF and self._has_cargo:
            reward = 100.0
            terminated = True
        elif self._fuel <= 0:
            reward = -20.0
            terminated = True

        truncated = (not terminated) and (self._steps >= self.max_steps)

        return self._state(), reward, terminated, truncated, {}

    def render(self):
        grid = [['.'] * self.COLS for _ in range(self.ROWS)]
        r, c = self.DEPOT
        grid[r][c] = 'D'
        r, c = self.DROPOFF
        grid[r][c] = 'X'
        for r, c in self.REFUEL_CELLS:
            grid[r][c] = 'R'
        grid[self._row][self._col] = 'A'
        print('\n'.join(' '.join(row) for row in grid))
        print(f"fuel={self._fuel} cargo={self._has_cargo}")
