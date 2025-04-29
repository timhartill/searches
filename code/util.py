"""
Misc Utility fns
"""

import os
import csv
import numpy as np
import heapq

#NP Grid  value constants
EMPTY = 0
OBSTACLE = 1
START = 2
GOAL = 3
PATH = 4
MEET = 8
EXPANDED_FWD = 6    # Yellow
EXPANDED_BWD = 10   # GreenYellow
EXPANDED_BOTH = 12  # MediumSpringGreen - bluish green

# The following dicts are for loading scenario files from the MovingAI benchmarks
SCEN_COL_MAP = {"Bucket":0, "map": 1, "width": 2, "height": 3, "start_x": 4, "start_y": 5, "goal_x": 6,	"goal_y": 7, "cstar": 8}
SCEN_COL_TYPES = {"Bucket": str, "map": str, "width": int, "height": int, "start_x": int, "start_y": int, "goal_x": int, "goal_y": int, "cstar": float}

# Values in movingAI map files mapped to our equivalent values
GRID_MAP = {
    '.': EMPTY, # passable terrain
    'G': EMPTY, # passable terrain
    '@': OBSTACLE, # out of bounds
    'O': OBSTACLE, # out of bounds
    'T': OBSTACLE, # trees (unpassable)
    'S': EMPTY,    # swamp (passable from regular terrain)
    'W': OBSTACLE # water (traversable, but not passable from terrain) NOT SUPPORTED HERE
}


def make_prob_serial(prob, prefix="__", suffix=""):
    """ Make csv-friendly key for a problem description eg initial state """
    prob_str = str(prob)
    prob_str = prob_str.replace(" ", "_").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(",", "").replace("'", "").replace("'", "").replace('"', "")
    return prefix + prob_str + suffix

def make_prob_str(file_name='', initial_state=None, goal_state=None, prefix="__", suffix="__"):
    """ Make a string for the problem description eg initial state """
    if file_name:
        file_name = prefix + os.path.basename(file_name)
    prob_str = file_name
    if initial_state is not None:
        prob_str += make_prob_serial(initial_state, prefix=prefix, suffix="")
    if goal_state is not None:
        prob_str += make_prob_serial(goal_state, prefix=prefix, suffix="")
    return prob_str + suffix


def write_jsonl_to_csv(all_results, csv_file_path, del_keys=['path'], 
                       delimiter=',', lineterminator='\n', verbose=True):
    """ Write a list of dictionaries to a CSV file optionally deleting some keys and making the columns
        consistent across all rows by adding header as superset of all keys and adding blanks to rows where necessary.
    """
    all_keys = set()
    for result in all_results:
        if del_keys:
            for del_key in del_keys:
                if del_key in result:
                    del result[del_key] 
        all_keys.update(result.keys())
    for result in all_results:
        for key in all_keys:
            if key not in result:
                result[key] = None
    with open(csv_file_path, 'w') as csv_file:
        fieldnames = all_results[0].keys() if all_results else []
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=delimiter, lineterminator=lineterminator)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)   
    if verbose:
        print(f"Results written to {csv_file_path}")
    return csv_file_path


def load_scen_file(file_path):
    """ Load a scenario file downloaded from https://www.movingai.com/benchmarks/grids.html  
    The first line is the version and is skipped
    Remaining lines are tab delimited columns: 
    Bucket	map	map width	map height	start x-coordinate	start y-coordinate	goal x-coordinate	goal y-coordinate	optimal length 

    The .scen files are in a directory named after the domain + "-scen" eg dao-scen
    The corresponding .map files are in a directory named after the domain + "-map" eg dao-map
    """
    if not file_path.endswith('.scen'):
        raise ValueError(f"File {file_path} is not a .scen file")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    dir_basepath = os.path.dirname(file_path)       # '/home/user/dao-scen'
    dir_name = os.path.basename(dir_basepath)       # 'dao-scen'
    dir_basepath = os.path.dirname(dir_basepath)    # '/home/user'
    if not dir_name.endswith('-scen'):
        dir_name_map = ''
    else:
        dir_name_map = os.path.join(dir_basepath, dir_name[:-len('-scen')] + '-map')  # '/home/user/dao-map'
       
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = lines[1:]  # Skip the first line
    scenarios = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            cols = line.split('\t')
            scenario = {}
            for col, index in SCEN_COL_MAP.items():
                scenario[col] = SCEN_COL_TYPES[col](cols[index])
            scenario['map_dir'] = os.path.join(dir_name_map, scenario['map'])
            scenarios.append(scenario)
    return scenarios


def load_map_file(file_path):
    """ Load a map file downloaded from https://www.movingai.com/benchmarks/grids.html  
    The first 4 lines are like:
        type octile
        height 792
        width 538
        map
    Remaining lines are tab delimited columns: 
    Bucket	map	map width	map height	start x-coordinate	start y-coordinate	goal x-coordinate	goal y-coordinate	optimal length 
    """
    if not file_path.endswith('.map'):
        raise ValueError(f"File {file_path} is not a .map file")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    heuristic = lines[0].strip().split()[1].lower()     # type octile
    height = int(lines[1].strip().split()[1])           # height 792
    width = int(lines[2].strip().split()[1])            # width 538
    lines = lines[4:]                                   # Now skip the first 4 lines
    matrix_data = np.zeros((height, width), dtype=int)  # empty default
    for row, line in enumerate(lines):
        for col, char in enumerate(line.strip()):
            if char in GRID_MAP:
                matrix_data[row, col] = GRID_MAP[char]
            else:
                print(f"Warning: Invalid character {char} in map file {file_path} at row {row}, col {col}")
    return matrix_data, heuristic


class PriorityQueue:
    """ Priority Queue implementation supporting 3 levels of priority: heuristic value, tiebreaker1, tiebreaker2
    ie list of tuples (priority, tiebreaker1, tiebreaker2, item)
    """
    def __init__(self, tiebreaker2='FIFO'):
        self.heap = []
        self.tiebreaker2 = tiebreaker2
        if tiebreaker2 == 'FIFO':
            self.increment = 1
        elif tiebreaker2 == 'LIFO':
            self.increment = -1
        else:
            raise ValueError("tiebreaker2 must be either 'FIFO' or 'LIFO'")
        self.count = 0
        self.max_heap_size = 0

    def push(self, item, priority, tiebreaker1=0):
        entry = (priority, tiebreaker1, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += self.increment
        if self.max_heap_size < len(self.heap):
            self.max_heap_size = len(self.heap)

    def pop(self, item_only=True):
        if not self.isEmpty():
            priority, tiebreaker1, tiebreaker2, item = heapq.heappop(self.heap)
            if item_only:
                return item
            return item, priority, tiebreaker1, tiebreaker2
        return None

    def isEmpty(self):
        return len(self.heap) == 0

    def peek(self, priority_only=True):
        """View the lowest priority element without popping it
        """
        if not self.isEmpty():
            if priority_only:
                return self.heap[0][0]  
            else:
                # Return the whole entry (priority, tiebreaker1, tiebreaker2, item)
                return self.heap[0]
        return None



