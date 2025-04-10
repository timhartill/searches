"""
spatial pathfinding problem and utilities

some code adapted from https://github.com/brean/python-pathfinding

"""

import math
import random
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
    """ Octile distance modifies manhattan distance to account for diagonal movement 
        ie remains admissable. """
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
    Can use uniform horiz/vertical cost (1 and SQRT2) or fixed horiz/cost (cost_multiplier > 1). 
    If diagonal movement is allowed this will introduce variable edge cost and manhattan becomes inadmissable (can switch to octile for admissability)
    Can allow diagonal movement or not - if so, will prevent movement if an obstacle is present on either "manhattan" path 
    """
    def __init__(self, grid_file, initial_state=None, goal_state=None, 
                 cost_multiplier=1, 
                 make_heuristic_inadmissable=False,
                 degradation=0,
                 allow_diagonal=False,
                 heuristic='manhattan'):
        self.grid = np.load(grid_file)
        if len(self.grid.shape) != 2:
            raise ValueError("Grid must be a 2D numpy array of type int.")
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
        self.optimality_guaranteed = True  # TODO update this based on heuristic and allow_diagonal
        self.h_str = heuristic
        self.h_func = h_map.get(heuristic)
        if self.h_func is None:
            raise ValueError(f"Invalid heuristic '{heuristic}'. Available heuristics: {list(h_map.keys())}")
        if allow_diagonal and heuristic == 'manhattan':
            self.optimality_guaranteed = False
        self.make_heuristic_inadmissable = make_heuristic_inadmissable
        if make_heuristic_inadmissable:
            self.h_multiplier = cost_multiplier * (degradation+10)  
            self.optimality_guaranteed = False
        else:
            self.h_multiplier = 1    
        self.degradation = degradation   
        self.cost_multiplier = cost_multiplier 
        cost_type = "VarCost" if self.use_variable_costs else "UnitCost"
        self._str_repr = f"Grid-{self.max_rows}x{self.max_cols}-{cost_type}-h{heuristic}-d{degradation}-a{not make_heuristic_inadmissable}"

    def initial_state(self): 
        return self.initial_state_tuple
        
    def goal_state(self): 
        return self.goal_state_tuple
        
    def is_goal(self, state): 
        return state == self.goal_state_tuple

    def get_neighbors(self, state):
        """Returns list of tuples: (neighbor_state, cost) from state
        state = (row, col) and neighbor_state = (new_row, new_col)
        """
        neighbors = []
        row, col = state
        # north = "up"
        valid_set = set()
        moves = {'north': (row-1, col), 'south': (row+1, col), 'east': (row, col-1), 'west': (row, col+1)}
        for move_dir, (new_row, new_col) in moves.items():
            if 0 <= new_row < self.max_rows and 0 <= new_col < self.max_cols:  # if on grid
                if self.grid[new_row, new_col] == OBSTACLE: 
                    continue  # if obstacle
                new_state_tuple = (new_row, new_col)
                neighbors.append( (new_state_tuple, self.cost_multiplier) )
                valid_set.add(move_dir)

        if self.allow_diagonal and valid_set:  
            cost = SQRT2 * self.cost_multiplier  # equiv to sqrt(cost_multiplier^2 + cost_multiplier^2)
            moves_diag = {'nw': (row-1, col-1), 'ne': (row-1, col+1), 'sw': (row+1, col-1), 'se': (row+1, col+1)}
            for move_dir, (new_row, new_col) in moves_diag.items():
                if 0 <= new_row < self.max_rows and 0 <= new_col < self.max_cols:  # if on grid
                    if self.grid[new_row, new_col] == OBSTACLE: 
                        continue  # if obstacle

                    # valid if either manhatten walk has no obstacle
                    if move_dir == 'nw' and ('north' in valid_set or 'west' in valid_set):  
                        new_state_tuple = (new_row, new_col)
                        neighbors.append( (new_state_tuple, cost) )
                    elif move_dir == 'ne' and ('north' in valid_set or 'east' in valid_set):
                        new_state_tuple = (new_row, new_col)
                        neighbors.append( (new_state_tuple, cost) )
                    elif move_dir == 'sw' and ('south' in valid_set or 'west' in valid_set):
                        new_state_tuple = (new_row, new_col)
                        neighbors.append( (new_state_tuple, cost) )
                    elif move_dir == 'se' and ('south' in valid_set or 'east' in valid_set):
                        new_state_tuple = (new_row, new_col)
                        neighbors.append( (new_state_tuple, cost) )
        return neighbors 

    def get_cost(self, state1, state2, move_info=None):
        """
        Returns cost of move. 
        If diagonal, cost is sqrt(2) * cost_multiplier, otherwise, cost is 1 * cost_multiplier.
        """

        if move_info is not None:
            return max(1, move_info)

        return euclidean(abs(state1[0] - state2[0]), 
                            abs(state1[1] - state2[1])) * self.cost_multiplier


    def heuristic(self, state):
        """
        Calculates the heuristic.
        NOTE: This heuristic assumes unit cost (cost=1 or SQRT2). If cost multipier > 1 is used,
        its effectiveness will decrease but still admissable since multiplied costs >= unit costs.
        """
        dx = abs(state[0] - self.goal_state_tuple[0])
        dy = abs(state[1] - self.goal_state_tuple[1])


        distance = self.h_func(dx, dy)

        if self.degradation > 0:
            degrade = self.degradation+1 #random.choice(range(1,self.degradation+1))
            distance = distance / degrade  # (self.degradation+1)  # random.choice(range(1,self.degradation+1))


        return distance * self.h_multiplier
        
    def __str__(self): 
        return self._str_repr

"""
# tests
#'/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_10yX10x.npy'
test = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_10yX10x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=True, heuristic='manhattan')
test._str_repr
test.grid
test.initial_state() # (1, 0)
test.goal_state() # (6,9)
test.is_goal((1,0))
test.get_neighbors((1,0))
test.get_cost((1,0), (2,1), 5)
test.heuristic((1,0))
test.heuristic((6,9))

test = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_10yX10x.npy',
                   initial_state=[0,0], goal_state=[9,9], cost_multiplier=5,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=True, heuristic='manhattan')
test._str_repr
test.grid
test.initial_state() # (0, 0)
test.goal_state() # (9,9)
test.get_neighbors((0,0))
test.get_cost((0,0), (1,1), 5)
test.heuristic((0,0))

test = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_10yX10x.npy',
                   initial_state=[0,0], goal_state=[9,9], cost_multiplier=5,
                   make_heuristic_inadmissable=True, degradation=0,
                   allow_diagonal=True, heuristic='manhattan')
test.heuristic((0,0))  # 900

test = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_10yX10x.npy',
                   initial_state=[0,0], goal_state=[9,9], cost_multiplier=5,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=True, heuristic='euclidean')
test.heuristic((0,0))  # 12.72

test = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_10yX10x.npy',
                   initial_state=[0,0], goal_state=[9,9], cost_multiplier=5,
                   make_heuristic_inadmissable=True, degradation=0,
                   allow_diagonal=True, heuristic='euclidean')
test.heuristic((0,0))  # 636.39

test = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_10yX10x.npy',
                   initial_state=[0,0], goal_state=[9,9], cost_multiplier=5,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=True, heuristic='octile')
test.heuristic((0,0))  # 636.39

test = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_10yX10x.npy',
                   initial_state=[0,0], goal_state=[9,9], cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=True, heuristic='manhattan')
test.heuristic((0,0))  # 18

test = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_10yX10x.npy',
                   initial_state=[0,0], goal_state=[9,9], cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=4,
                   allow_diagonal=True, heuristic='manhattan')
test.heuristic((0,0))  # 3.6

"""

