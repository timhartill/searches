"""
spatial pathfinding problem and utilities

some code adapted from https://github.com/brean/python-pathfinding

"""

import math
import numpy as np

# Constants
SQRT2 = math.sqrt(2)
EMPTY = 0
OBSTACLE = 1
START = 2
GOAL = 3

def manhattan(dx, dy) -> float:
    """manhattan heuristics"""
    return dx + dy


def euclidean(dx, dy) -> float:
    """euclidean distance heuristics"""
    return math.sqrt(dx * dx + dy * dy)


def chebyshev(dx, dy) -> float:
    """ Chebyshev distance. """
    return max(dx, dy)


def octile(dx, dy) -> float:
    f = SQRT2 - 1
    if dx < dy:
        return f * dx + dy
    else:
        return f * dy + dx
    
h_map = {
    'manhattan': manhattan,
    'euclidean': euclidean,
    'chebyshev': chebyshev,
    'octile': octile
}




class GridProblem:
    """
    Implements the standard problem interface for a grid.
    grid_file = the full path to a numpy file containing the grid that must be a 2d int matrix with elements: 
        0=empty/walkable, 1=obstacle, 2=start, 3=goal
    if initial_state = None, the start state is calculated from the grid, otherwise the grid value of 2 is ignored by setting to 0
    if goal_state = None, the goal state is calculated from the grid, otherwise the grid value of 3 is ignored by setting to 0
    Can use uniform horiz/vertical cost (1) or fixed horiz/cost (cost_multiplier > 1). 
    If diagonal movement is allowed this will introduce variable edge cost
    Can allow diagonal movement or not - if so, will prevent movement if an obstacle is present on either "manhattan" path 
    """
    def __init__(self, grid_file, initial_state=None, goal_state=None, 
                 cost_multiplier=1, 
                 make_heuristic_inadmissable=False,
                 degradation=0,
                 allow_diagonal=False,
                 heuristic='manhattan'):
        self.grid = np.load(grid_file)
        self.max_rows, self.max_cols = self.grid.shape
        if initial_state is not None:
            locations_to_clear = self.grid == START
            if np.any(locations_to_clear):
                self.grid[locations_to_clear] = EMPTY
            self.grid[initial_state[0], initial_state[1]] = START
        else:
            r,c = np.where(self.grid==START)
            if r.shape[0] == 0 or c.shape[0] == 0:
                raise ValueError("Grid does not contain a start position (2) and no initial start position was provided.")
            initial_state = [int(r[0]), int(c[0])]
        if goal_state is not None:
            locations_to_clear = self.grid == GOAL
            if np.any(locations_to_clear):
                self.grid[locations_to_clear] = EMPTY
            self.grid[goal_state[0], goal_state[1]] = GOAL
        else:
            r,c = np.where(self.grid==GOAL)
            if r.shape[0] == 0 or c.shape[0] == 0:
                raise ValueError("Grid does not contain a goal position (3) and no initial goal position was provided.")
            goal_state = [int(r[0]), int(c[0])]
        self.initial_state_tuple = tuple(initial_state)  # start coordinates (rows, columns)    
        self.goal_state_tuple = tuple(goal_state)
            
        self.allow_diagonal = allow_diagonal
        self.use_variable_costs = allow_diagonal
        self.optimality_guaranteed = True  # update this based on heuristic and allow_diagonal
        self.h_str = heuristic
        self.h_func = h_map.get(heuristic)
        if self.h_func is None:
            raise ValueError(f"Invalid heuristic '{heuristic}'. Available heuristics: {list(h_map.keys())}")
        self.make_heuristic_inadmissable = make_heuristic_inadmissable
        if make_heuristic_inadmissable:
            self.h_multiplier = degradation+10  # update
            self.optimality_guaranteed = False
        else:
            self.h_multiplier = 1    
        self.degradation = degradation    
        cost_type = "VarCost" if self.use_variable_costs else "UnitCost"
        self._str_repr = f"SlidingTile-{self.max_rows}x{self.max_cols}-{cost_type}-h{heuristic}-d{degradation}-a{not make_heuristic_inadmissable}"

    def initial_state(self): 
        return self.initial_state_tuple
        
    def goal_state(self): 
        return self.goal_state_tuple
        
    def is_goal(self, state): 
        return state == self.goal_state_tuple

    def get_neighbors(self, state):
        """Returns list of tuples: (neighbor_state, moved_tile_value) from state
        where moved_tile_value is the value (label) of the tile that was moved to the blank space.
        """
        #TODO UPDATE
        neighbors = []
        blank_index = state.index(0)    # index of blank in state list 
        row, col = divmod(blank_index, self.n)
        moves = {'up': (row - 1, col), 'down': (row + 1, col), 'left': (row, col - 1), 'right': (row, col + 1)}
        for move_dir, (new_row, new_col) in moves.items():
            if 0 <= new_row < self.n and 0 <= new_col < self.n:  # if valid move
                new_blank_index = new_row * self.n + new_col  # index of blank if move this direction
                new_state_list = list(state)
                moved_tile_value = new_state_list[new_blank_index] 
                # swap blank with the tile in the new position:
                new_state_list[blank_index], new_state_list[new_blank_index] = new_state_list[new_blank_index], new_state_list[blank_index]
                neighbors.append((tuple(new_state_list), moved_tile_value)) 
        return neighbors 

    def get_cost(self, state1, state2, move_info=None):
        """
        Returns cost of move. If use_variable_costs is True, cost is the
        value of the tile moved (min 1). Otherwise, cost is 1.
        """
        #TODO UPDATE
        if not self.use_variable_costs:
            return 1 
            
        if move_info is not None:
            moved_tile_value = move_info
            return max(1, moved_tile_value) 
        else: 
            blank1_idx, blank2_idx, moved_tile = -1, -1, -1
            # Ensure state1 and state2 are tuples/sequences before len()
            if not hasattr(state1, '__len__') or not hasattr(state2, '__len__') or len(state1) != len(state2):
                print(f"Warning: Invalid states for cost calculation fallback: {state1}, {state2}")
                return 1 # Fallback cost
            for i in range(len(state1)):
                if state1[i] == 0: blank1_idx = i
                if state2[i] == 0: blank2_idx = i
            if blank1_idx != -1 and blank2_idx != -1 :
                 moved_tile = state2[blank1_idx] 
                 return max(1, moved_tile)
            else:
                 print(f"Warning: Could not determine moved tile between {state1} and {state2}")
                 return 1 

    def heuristic(self, state):
        """
        Calculates the Manhattan distance heuristic (number of steps).
        NOTE: This heuristic counts steps (cost=1). If variable (positive) costs are used,
        its effectiveness will decrease but still admissable since var costs >= unit costs.
        """
        #TODO UPDATE
        distance = 0
        multiplier = 1
        ignored_tiles = set(range(self.degradation + 1))
        for i, tile in enumerate(state):
            if tile not in ignored_tiles:
                current_pos = divmod(i, self.n)
                goal_idx = self._goal_positions.get(tile)
                if goal_idx is None: continue 
                goal_pos = divmod(goal_idx, self.n)
                if self.make_heuristic_inadmissable:
                    multiplier = i
                distance += multiplier * (abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1]))
        return distance * self.h_multiplier
        
    def __str__(self): 
        return self._str_repr




