"""
spatial pathfinding problem and utilities

heuristic fns and diagonal movement logic adapted from https://github.com/brean/python-pathfinding

"""

import os
import math
import random
import numpy as np
from PIL import Image
from typing import Dict, Tuple # Optional: for type hinting

import util


# Constants
SQRT2 = math.sqrt(2)
EMPTY = 0
OBSTACLE = 1
START = 2
GOAL = 3
PATH = 4
MEET = 8
EXPANDED_FWD = 6    # Yellow
EXPANDED_BWD = 10   # GreenYellow
EXPANDED_BOTH = 12  # MediumSpringGreen - bluish green

# --- Color Mapping Definition ---
# Using RGB tuples (Red, Green, Blue)  Obtain by:
# from PIL import ImagePalette
# ImagePalette.ImageColor.getrgb('mediumspringgreen')  # (0, 250, 154)
COLOR_MAP: Dict[int, Tuple[int, int, int]] = {
    0: (255, 255, 255),  # White - empty
    1: (0, 255, 255),      # Cyan - obstacle
    2: (0, 255, 0),      # Green - start
    3: (255, 0, 0),      # Red - goal
    4: (128, 0, 128),     # Purple - path
    5: (0, 0, 0),      # Black
    6: (255, 255, 0),  # Yellow
    7: (0, 0, 255),      # Blue
    8: (255, 165, 0),  # Orange
    9: (128, 128, 128), # Grey
    10: (173, 255, 47), # GreenYellow
    11: (127, 255, 0), # Chartreuse
    12: (0, 250, 154), # MediumSpringGreen
}
DEFAULT_COLOR: Tuple[int, int, int] = (0, 0, 0) # Black for values outside the map



def manhattan(dx, dy) -> float:
    """manhattan heuristics - inadmissable of diagonal movement allowed"""
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
        self.grid_file = grid_file
        self.basename = os.path.basename(grid_file)
        self.dirname = os.path.dirname(grid_file)
        self.basename_no_ext = os.path.splitext(self.basename)[0]
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
        self.optimality_guaranteed = True  
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
        self._str_repr = f"Grid-{self.max_rows}x{self.max_cols}-{util.make_prob_str(file_name=self.basename, initial_state=self.initial_state_tuple, goal_state=self.goal_state_tuple)}-{cost_type}-h{heuristic}-d{degradation}-a{self.optimality_guaranteed and not make_heuristic_inadmissable}-cm{cost_multiplier}"

    def initial_state(self): 
        return self.initial_state_tuple
        
    def goal_state(self): 
        return self.goal_state_tuple
        
    def is_goal(self, state, backward=False): 
        if backward:
            return state == self.initial_state_tuple
        return state == self.goal_state_tuple

    def get_neighbors(self, state):
        """Returns list of tuples: (neighbor_state, cost) from state
        state = (row, col) and neighbor_state = (new_row, new_col)
        """
        neighbors = []
        row, col = state
        # north = "up"
        valid_set = set()
        moves = {'north': (row-1, col), 'south': (row+1, col), 'east': (row, col+1), 'west': (row, col-1)}
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
            return move_info

        return euclidean(abs(state1[0] - state2[0]), 
                            abs(state1[1] - state2[1])) * self.cost_multiplier


    def heuristic(self, state, backward=False):
        """
        Calculates the heuristic.
        NOTE: This heuristic assumes unit cost (cost=1 or SQRT2). If cost multipier > 1 is used,
        its effectiveness will decrease but still admissable since multiplied costs >= unit costs.
        """
        if backward:
            dx = abs(state[0] - self.initial_state_tuple[0])
            dy = abs(state[1] - self.initial_state_tuple[1])
        else:    
            dx = abs(state[0] - self.goal_state_tuple[0])
            dy = abs(state[1] - self.goal_state_tuple[1])


        distance = self.h_func(dx, dy)

        if self.degradation > 0:
            degrade = self.degradation+1 #random.choice(range(1,self.degradation+1))
            distance = distance / degrade  # (self.degradation+1)  # random.choice(range(1,self.degradation+1))


        return distance * self.h_multiplier
    

    def visualise(self, cell_size: int = 10, path: list = None, meeting_node: tuple = None, 
                  visited_fwd: set = None, visited_bwd: set = None,
                  path_type: str = '', output_file_ext: str = 'png',
                  display: bool = False, return_image: bool = False):
        """
        Converts a 2D NumPy array of integers into an image file, mapping
        values to specific colors and creating blocks of pixels.

        Args:
            numpy_array: The 2D NumPy array (dtype=int) with values typically 0-4.
            output_filename: The path and name for the output image file
            cell_size: The width and height (in pixels) of the square cell
                    representing each array element (default: 10).

        Raises:
            TypeError: If input is not a NumPy array.
            ValueError: If input array is not 2-dimensional or cell_size is not positive.
            Exception: Catches potential errors during image saving.
        """
        # --- Input Validation ---
        if not isinstance(cell_size, int) or cell_size <= 0:
            raise ValueError("cell_size must be a positive integer.")

        # --- Image Creation Logic ---
        rows, cols = self.max_rows, self.max_cols

        # Create an intermediate RGB array based on the input array and color map
        # This array will have dimensions (rows, cols, 3) for RGB values
        rgb_array = np.zeros((rows, cols, 3), dtype=np.uint8)
        grid_draw = self.grid.copy()

        if visited_fwd:
            # If visited_fwd is provided, set expanded values in grid_draw to EXPANDED_FWD
            for r, c in visited_fwd:
                grid_draw[r, c] = EXPANDED_FWD
        if visited_bwd:
            # If visited_bwd is provided, set expanded values in grid_draw to EXPANDED_BWD or EXPANDED_BOTH
            for r, c in visited_bwd:
                if grid_draw[r, c] == EXPANDED_FWD:
                    grid_draw[r, c] = EXPANDED_BOTH
                else:
                    grid_draw[r, c] = EXPANDED_BWD

        if path:
            # If path is provided, set path values in grid_draw to PATH
            if meeting_node:
                meet_r, meet_c = meeting_node
            else:
                meet_r, meet_c = -1, -1
 
            for r, c in path:
                if r == meet_r and c == meet_c:
                    grid_draw[r, c] = MEET
                else:
                    grid_draw[r, c] = PATH

        for r in range(rows):
            for c in range(cols):
                value = grid_draw[r, c]
                # Get color from map, use default if value not found
                rgb_array[r, c] = COLOR_MAP.get(value, DEFAULT_COLOR)

        # Create a small PIL image from the rgb_array
        # Each pixel in this small image corresponds to one element in the numpy_array
        small_image = Image.fromarray(rgb_array, 'RGB')

        # Calculate the final image dimensions by scaling up
        final_width = cols * cell_size
        final_height = rows * cell_size

        if cell_size > 1:
            # Resize the small image to the final dimensions
            # Use NEAREST resampling to keep the blocky pixel look without blurring
            # Note: Pillow versions >= 9.1.0 use Image.Resampling.NEAREST
            # Older versions (< 9.1.0) use Image.NEAREST
            try:
                resample_filter = Image.Resampling.NEAREST
            except AttributeError: # Handle older Pillow versions
                resample_filter = Image.NEAREST

            final_image = small_image.resize((final_width, final_height), resample_filter)
        else:
            # If cell_size is 1, no resizing is needed
            final_image = small_image

        if path_type != '':
            out_dir = os.path.join(self.dirname, self.basename_no_ext)
            os.makedirs(out_dir, exist_ok=True)
            output_filename = os.path.join(out_dir, f"{self._str_repr}_{path_type}.{output_file_ext}")
            final_image.save(output_filename)  #saved in format inferred from file extension
        if display:
            final_image.show()
        if return_image:    
            return final_image
        elif path_type != '':
            return os.path.basename(output_filename)
        return None

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

test = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=True, heuristic='manhattan')
test.visualise(cell_size=10, output_filename='/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_20yX100x.png', display=True)

grid_path = '/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_20yX100x.npy'
"""

