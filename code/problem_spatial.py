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

# --- Color Mapping Definition ---
# Using RGB tuples (Red, Green, Blue)  Obtain by:
# from PIL import ImagePalette
# ImagePalette.ImageColor.getrgb('mediumspringgreen')  # (0, 250, 154)
COLOR_MAP: Dict[int, Tuple[int, int, int]] = {
    util.EMPTY: (255, 255, 255),  # White - empty
    util.OBSTACLE: (0, 255, 255),      # Cyan - obstacle
    util.START: (0, 255, 0),      # Green - start
    util.GOAL: (255, 0, 0),      # Red - goal
    util.PATH: (128, 0, 128),     # Purple - path
    5: (0, 0, 0),      # Black
    util.EXPANDED_FWD: (255, 255, 0),  # Yellow
    7: (0, 0, 255),      # Blue
    util.MEET: (255, 165, 0),  # Orange
    9: (128, 128, 128), # Grey
    util.EXPANDED_BWD: (173, 255, 47), # GreenYellow
    11: (127, 255, 0), # Chartreuse
    util.EXPANDED_BOTH: (0, 250, 154), # MediumSpringGreen - bluish green
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
        ie remains admissable with variable cost. """
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
    file = the full path to a numpy file containing the grid that must be a 2d int matrix with elements: 
        0=empty/walkable, 1=obstacle, 2=start, 3=goal or a .map file that we convert into this format
    if initial_state = None, the start state is calculated from the grid, otherwise the grid value of 2 is ignored by setting to 0
    if goal_state = None, the goal state is calculated from the grid, otherwise the grid value of 3 is ignored by setting to 0
    Can use uniform horiz/vertical cost (1 and SQRT2) or fixed horiz/cost (cost_multiplier > 1). 
    If diagonal movement is allowed this will introduce variable edge cost and manhattan becomes inadmissable (can switch to octile for admissability)
    Can allow diagonal movement or not - if so, will prevent movement if an obstacle is present on either "manhattan" path 
    Some hog2 grids c* annotations assume diagonal cost=2 but others are different hence one would generally pass cstar=None for hog2
    and rely on the search calculations for cstar and nodes expanded below cstar
    """
    def __init__(self, file, initial_state=None, goal_state=None, 
                 cost_multiplier=1.0, 
                 make_heuristic_inadmissable=False,
                 degradation=0,
                 allow_diagonal=False,
                 diag_cost = 1.5,
                 heuristic='octile',
                 cstar=None):
        if file is None or not os.path.exists(file):
            raise ValueError(f"Grid file {file} does not exist.")
        if file.endswith('.npy'):
            self.grid = np.load(file)
        elif file.endswith('.map'):
            self.grid, heuristic_default = util.load_map_file(file)
            if heuristic is None and heuristic_default in h_map:
                heuristic = heuristic_default
        else:
            raise ValueError(f"Grid file {file} must be a .npy or .map file.")
        if len(self.grid.shape) != 2:
            raise ValueError("Grid must be a 2D numpy array of type int.")
        self.file = file
        self.basename = os.path.basename(file)
        self.dirname = os.path.dirname(file)
        self.griddomain = os.path.basename(self.dirname)
        self.basename_no_ext = os.path.splitext(self.basename)[0]
        self.max_rows, self.max_cols = self.grid.shape
        if initial_state is not None:
            locations_to_clear = self.grid == util.START
            if np.any(locations_to_clear):
                self.grid[locations_to_clear] = util.EMPTY
            self.grid[initial_state[0], initial_state[1]] = util.START
        else:
            r,c = np.where(self.grid==util.START)
            if r.shape[0] == 0 or c.shape[0] == 0:
                raise ValueError("Grid does not contain a start position (2) and no initial start position was provided.")
            initial_state = [int(r[0]), int(c[0])]
        if goal_state is not None:
            locations_to_clear = self.grid == util.GOAL
            if np.any(locations_to_clear):
                self.grid[locations_to_clear] = util.EMPTY
            self.grid[goal_state[0], goal_state[1]] = util.GOAL
        else:
            r,c = np.where(self.grid==util.GOAL)
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
        self.diag_cost = diag_cost
        self.cstar = cstar
        self.cost_type = "VarCost" if self.use_variable_costs else "UnitCost"
        self.admissible = self.optimality_guaranteed and not self.make_heuristic_inadmissable
        self._str_repr = f"{self.griddomain}-R{self.max_rows}xC{self.max_cols}-{self.cost_type}-dc{self.diag_cost}-cm{self.cost_multiplier}-{util.make_prob_str(file_name=self.basename, initial_state=self.initial_state_tuple, goal_state=self.goal_state_tuple)}-h{self.h_str}-d{self.degradation}-a{self.admissible}-cs{self.cstar}"
        self.prob_str = f"{self.griddomain}-{self.cost_type}-dc{self.diag_cost}-cm{self.cost_multiplier}"


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
                if self.grid[new_row, new_col] == util.OBSTACLE: 
                    continue  # if obstacle
                new_state_tuple = (new_row, new_col)
                neighbors.append( (new_state_tuple, self.cost_multiplier) )
                valid_set.add(move_dir)

        if self.allow_diagonal and valid_set:  
            cost = self.diag_cost * self.cost_multiplier #SQRT2 * self.cost_multiplier  # equiv to sqrt(cost_multiplier^2 + cost_multiplier^2)
            moves_diag = {'nw': (row-1, col-1), 'ne': (row-1, col+1), 'sw': (row+1, col-1), 'se': (row+1, col+1)}
            for move_dir, (new_row, new_col) in moves_diag.items():
                if 0 <= new_row < self.max_rows and 0 <= new_col < self.max_cols:  # if on grid
                    if self.grid[new_row, new_col] == util.OBSTACLE: 
                        continue  # if obstacle

                    # valid if either manhattan walk has no obstacle
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
        If diagonal, cost is diag_cost * cost_multiplier, otherwise, cost is 1 * cost_multiplier.
        """
        if move_info is not None:
            return move_info

        cost = euclidean(abs(state1[0] - state2[0]), 
                         abs(state1[1] - state2[1]))
        if cost == SQRT2:
            cost = self.diag_cost
        return math.ceil(cost * self.cost_multiplier * 100) / 100


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

        return math.floor(distance * self.h_multiplier * 100) / 100
    

    def visualise(self, cell_size: int = 10, path: list = None, meeting_node: tuple = None, 
                  visited_fwd: set = None, visited_bwd: set = None,
                  path_type: str = '', output_file_ext: str = 'png',
                  display: bool = False, return_image: bool = False,
                  visualise_dirname: str = ''):
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
                grid_draw[r, c] = util.EXPANDED_FWD
        if visited_bwd:
            # If visited_bwd is provided, set expanded values in grid_draw to EXPANDED_BWD or EXPANDED_BOTH
            for r, c in visited_bwd:
                if grid_draw[r, c] == util.EXPANDED_FWD:
                    grid_draw[r, c] = util.EXPANDED_BOTH
                else:
                    grid_draw[r, c] = util.EXPANDED_BWD

        if path:
            # If path is provided, set path values in grid_draw to PATH
            if meeting_node:
                meet_r, meet_c = meeting_node
            else:
                meet_r, meet_c = -1, -1
 
            for r, c in path:
                if r == meet_r and c == meet_c:
                    grid_draw[r, c] = util.MEET
                else:
                    grid_draw[r, c] = util.PATH

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
            out_dir = os.path.join(visualise_dirname, self.basename_no_ext)
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


##############################################################
# Problem loading and instantiation routines
##############################################################

def create_grid_probs(args):
    """ Load grid problems from a scen file and return list of problem instances
    """
    problems = []
    for domain in args.grid_dir_full:
        random.seed(args.seed)  # run the same random order each time on each domain
        scen_files = os.listdir(domain)
        for scen_file in scen_files:
            if not scen_file.endswith('.scen'):
                continue
            scenarios = util.load_scen_file( os.path.join(domain, scen_file) )  # NOTE: adds path to map_dir
            if args.grid_reverse_scen_order:
                scenarios.reverse()
            if args.grid_random_scen_order:
                random.shuffle(scenarios)
            count=0
            for i, scenario in enumerate(scenarios):
                if not scenario['map_dir'] or not os.path.exists(scenario['map_dir']):
                    print(f"Skipping scenario in {scen_file} as map file '{scenario['map_dir']}' does not exist.")
                    continue
                if not scenario['initial_state']:
                    print(f"Skipping scenario in {scen_file} as initial state '{scenario['initial_state']}' is invalid.")
                    continue
                if not scenario['goal_state']:
                    print(f"Skipping scenario in {scen_file} as goal state '{scenario['goal_state']}' is invalid.")
                    continue
                if count >= args.grid_max_per_scen:
                    break
                for heuristic in args.grid_heur:
                    for degradation in args.grid_degs:
                        if args.grid_ignore_cstar: 
                            cstar = None
                        else: 
                            cstar = scenario['cstar']
                        problem = GridProblem(  file=scenario['map_dir'],
                                                initial_state=scenario['initial_state'], 
                                                goal_state=scenario['goal_state'],
                                                cost_multiplier=args.grid_cost_multipier,
                                                make_heuristic_inadmissable=args.grid_inadmiss,
                                                degradation=degradation,
                                                allow_diagonal=args.grid_allow_diag,
                                                diag_cost=args.grid_diag_cost,
                                                heuristic=heuristic,
                                                cstar=cstar )
                        problems.append(problem)
                count += 1
    return problems



