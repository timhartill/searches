"""
Misc Utility fns
"""

import os
import csv
import json
import time
import random
import traceback # For error reporting
import numpy as np

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

# Dicts for loading our problem files
#PROBLEM_COL_MAP = {"problem_type": 0, "initial_state": 2, "goal_state": 3, "cstar": 4}
PROBLEM_COL_TYPES = {"problem_type": str, "initial_state": json.loads, "goal_state": json.loads, "cstar": float}


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
    """ Make filename-friendly key for a problem description eg initial state """
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
    """ Load a scenario file possibly downloaded from HOG2 (https://www.movingai.com/benchmarks/grids.html)
    The first line is the version and is skipped

    Remaining lines are tab delimited columns: 
    Bucket	map	map width	map height	start x-coordinate	start y-coordinate	goal x-coordinate	goal y-coordinate	optimal length 

    if using one of our numpy files as the grid, the corresponding scen file must contain at least Bucket and map
        
    The .scen files are in a directory named after the domain + "-scen" eg dao-scen
    The corresponding .map files are in a directory named after the domain + "-map" eg dao-map.

    /dao-scen and /dao-map are assumed to be in the same directory eg .../problems/matrices

    Outputs a list of scenario dictionaries of format:
    "Bucket": str, "map": str, "width": int, "height": int, "start_x": int, "start_y": int, 
    "goal_x": int, "goal_y": int, "cstar": float, "map_dir": str, "initial_state": list, "goal_state": list

    Note: Different HOG2 problems seem to use different diagonal costs hence it is hard to consistently replicate their cstar calculations. 
         Some eg maze512-1-6.map.scen use diag cost is 2 (and our cost multiplier = 1.0) but others use differing costs...
         Therefore we have created scen files for all the HOG2 problems that we have downloaded to output with diag cost 1.5 following Alacazar 2020
    """
    if not file_path.endswith('.scen'):
        raise ValueError(f"File {file_path} is not a .scen file")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    
    dir_basepath = os.path.dirname(file_path)       # '/problems/matrices/dao-scen'
    dir_name = os.path.basename(dir_basepath)       # 'dao-scen'
    dir_basepath = os.path.dirname(dir_basepath)    # '/problems/matrices'
    if not dir_name.endswith('-scen'):
        dir_name_map = ''
    else:
        dir_name_map = os.path.join(dir_basepath, dir_name[:-len('-scen')] + '-map')  # '/problems/matrices/dao-map'
       
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = lines[1:]  # Skip the first line
    scenarios = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            cols = line.split('\t') 
            num_cols = len(cols)
            scenario = {}
            for col, index in SCEN_COL_MAP.items():
                if index >= num_cols:
                    scenario[col] = None
                    continue
                if cols[index].lower().strip() in ["", "[]", "()", "none", "null"]:  #movingai probs have all cols filled out. For ours we just need the .npy filename from here since start, goal coords are already on the map
                    scenario[col] = None
                    continue
                scenario[col] = SCEN_COL_TYPES[col](cols[index])
            scenario['map_dir'] = os.path.join(dir_name_map, scenario['map']) # map col is mandatory and hence also bucket 
            scenario['initial_state'] = None
            if scenario['start_x'] and scenario['start_y']:
                scenario['initial_state'] = [scenario['start_y'], scenario['start_x']]
            scenario['goal_state'] = None
            if scenario['goal_x'] and scenario['goal_y']:
                scenario['goal_state'] = [scenario['goal_y'], scenario['goal_x']]
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

def load_csv_file(file_path, delimiter=';', apply_col_types=True):
    """ load a generic semicolon-delimited problem file with header 
        (well actually it can load any delimited text file with a header, just set apply_col_types=False)
    columns:
    problem_type; problem_id; initial_state; goal_state; cstar

    If cstar is supplied it will be the based on unit costs = 1.
    If goal_state is not supplied it will be created in problem classes as the standard goal state for that problem type. 
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)  # csv.DictReader automatically uses the first row as headers
        for row in reader:
            data.append(row)  # DictReader yields dictionaries of strings directly and trailing blank lines are ignored
    if apply_col_types:
        for problem in data:
            for col_name, col_value in problem.items():
                if col_name in PROBLEM_COL_TYPES:   # if not, retain as str
                    convert_func = PROBLEM_COL_TYPES[col_name]
                    if convert_func != str:         # already str
                        if col_value.lower().strip() in ["", "[]", "()", "none", "null"]:
                            problem[col_name] = None
                        else:
                            try:
                                problem[col_name] = convert_func(col_value)
                            except Exception as e:
                                print(f"Error converting column {col_name} with value {col_value} to {convert_func}: {e}")
    return data
    

def run_experiments(problems, algorithms, out_dir, out_prefix='search_eval', seed=42):
    """ Run a set of algorithms on a set of problems and save the results to a CSV file (without path)
    and a json file (with path) in the specified output directory.
    Args:
        problems (list): List of problems to solve
        algorithms (list): List of algorithms to use
        output_dir (str): Directory to save the results
    """
    out_file_base = f"{out_dir}/{out_prefix}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    all_results = []
    for problem in problems:  # For each problem
        print(f"\n{'=' * 20}")
        print(f"Solving: {problem}\nInitial State: {problem.initial_state()}")
        print(f"Goal State:    {problem.goal_state()}")
        print(f"Initial Heuristic: {problem.heuristic(problem.initial_state())}")
        print(f"{'-' * 20}")
        problem_results = []
        
        for algo in algorithms:  # For each algorithm
            print(f"Running {str(algo)}...")
            result = None
            random.seed(seed)  # Reset random seed for reproducibility before each algorithm run on each problem
            try:
                result = algo.search(problem) # Call the runner                
                # Set algorithm name in result consistently
                result['algorithm'] = str(algo)                
                print(f"{str(algo)} Done. Time: {result.get('time', -1):.4f}secs Nodes Expanded: {result.get('nodes_expanded', -1)} Path Cost: {result.get('cost', 'N/A')} Length: {len(result['path']) if result['path'] else 'No Path Found'}")

            except Exception as e:
                print(f"!!! ERROR during {str(algo)} on {problem}: {e}")
                traceback.print_exc() 
                result = { "path": None, "cost": -1, "nodes_expanded": -1, "time": -1, 
                           "algorithm": str(algo), "error": str(e)}

            if result: 
                 result['problem'] = str(problem)
                 if 'path' in result and result['path']:
                     result['unit_cost'] = len(result['path']) - 1
                 else:
                     result['unit_cost'] = -1
                 problem_results.append(result)
                 all_results.append(result)

        # --- Print Results for this Problem ---
        print(f"\n{'=' * 10} Results for {problem} {'=' * 10}")
        for res in problem_results:
            print(f"\nAlgorithm: {res.get('algorithm','N/A')}")
            if 'optimal' in res: print(f"Optimality Guaranteed: {res['optimal']}")
            if res.get('algorithm','').startswith("MCTS") and 'iterations' in res : print(f"MCTS Iterations: {res.get('iterations', 'N/A')}")
            if res.get('algorithm','').startswith("MCTS") and 'tree_root_visits' in res : print(f"MCTS Root Visits: {res.get('tree_root_visits', 'N/A')}")
            print(f"Time Taken: {res.get('time', -1):.4f} seconds")
            print(f"Nodes Expanded/Explored: {res.get('nodes_expanded', -1)}")
            print(f"Path Found: {'Yes' if res.get('path') else 'No'}")
            if res.get('path'): print(f"Path Cost: {res.get('cost', 'N/A')} Length: {len(res['path'])}")
            else:
                 print("Path Cost: N/A")
                 if res.get('algorithm','').startswith("MCTS") and 'best_next_state_estimate' in res and res['best_next_state_estimate']: print(f"MCTS Best Next State Estimate: {res['best_next_state_estimate']}")
                 if 'error' in res: print(f"ERROR during run: {res['error']}")
        print("=" * (34 + len(str(problem)))) # Adjusted length

    # Overall Summary
    print(f"\n{'*'*15} Overall Summary {'*'*15}")
    for res in all_results:
         status = f"Cost: {res.get('cost', 'N/A')} Length: {len(res['path'])}" if res.get('path') else ("No Path Found" if 'error' not in res else f"ERROR: {res['error']}")
         #print("Path:", res['path']) # Uncomment to see the full path states

         optimal_note = f"(Optimal: {res['optimal']})" if 'optimal' in res else ""
         algo_name = res.get('algorithm','N/A') 
         print(f"- Problem: {res.get('problem','N/A')}, Algorithm: {algo_name}, Time: {res.get('time',-1):.4f}s, Nodes: {res.get('nodes_expanded',-1)}, Status: {status} {optimal_note}")

    # --- Save Results to JSON ---
    json_file_path = f"{out_file_base}.json"
    with open(json_file_path, 'w') as json_file:
        json.dump(all_results, json_file, indent=4)
    print(f"Results saved to {json_file_path}") 

    # --- Save Results to CSV ---
    # Ensure all results have the same keys for CSV
    # If some results are missing keys, fill them with None
    csv_file_path = f"{out_file_base}.csv"
    write_jsonl_to_csv(all_results, csv_file_path, del_keys=['path'])
    return csv_file_path



