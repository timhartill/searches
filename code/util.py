"""
Misc Utility fns + the Experiment runner
"""

import os
import sys
import csv
import json
import time
import random
import traceback # For error reporting
import psutil
import copy
import itertools
import math

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

def iter_product(A, B):
    """ Return iterable that will generate cartesian product of iterable A and Iterable B
    eg: For tup_C in iter_product(A_list, B_tuple): print(tup_C)
    """
    return itertools.product(A, B)

def iter_permute(A):
    """ Return iterable that will generate all permutations of iterable A as tuples
    eg: For rand_state_tuple in iter_permute(A_state): print(rand_state_tuple) 
    """
    return itertools.permutations(A)

def rand_shuffle(A):
    """ Return a randomly shuffled copy of list A """
    acopy = list(A).copy()
    random.shuffle(acopy)
    return acopy

def get_puzzle_size(state):
    """ Get the dimensions of a tile puzzle represented as a flat list """
    n = int(math.sqrt(len(state)))
    if n * n != len(state):
        max_cols = n
        # Check if the state is in the form of n x n or n+1 x n
        max_rows, col_check = divmod(len(state), max_cols)
        if col_check != 0:
            raise ValueError("Invalid state length for a sliding tile puzzle. Must be n x n or n+1 x n.")
    else: # square puzzle
        max_rows = n
        max_cols = n
    return max_rows, max_cols

def get_inversion_count(state):
    """ Count inversions in a list of numbers ignoring the blank (0) """
    arr = [x for x in state if x != 0]
    inv_count = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv_count += 1
    return inv_count

def find_blank_row_from_bottom(state, row_size):
    """ Find the row of the blank (0) counting from bottom (1-based) """
    index = state.index(0)
    row_from_top = index // row_size
    return row_size - row_from_top

def tile_prob_solvable(state):
    """ Check if a tile problem is solvable by checking the number of inversions in the state
        and the parity of the blank tile position.
        For odd width puzzles, the puzzle is solvable if the number of inversions is even
        For even width puzzles, the puzzle is solvable if the sum of inversions and blank row is even
        Note This assumes the goal state is an ordered list with the blank at the beginning (Korf standard) 
             This will fail on different goal states with even width and blank on different row eg blank at end. 
    """
    rows, cols = get_puzzle_size(state)
    inversions = get_inversion_count(state)
    if cols % 2 == 1:  # Odd width, ignore rows
        return inversions % 2 == 0
    else:  # Even width
        blank_row = find_blank_row_from_bottom(state, rows)
        return (inversions + blank_row) % 2 == 0


def make_random_permutations(A, num_samples, giveup = 10000000, is_tile=False):
    """ Return num_samples permuted versions of list A that don't include A and arent duplicated
        If is_tile is True, only return solvable permutations of a tile problem
    """
    A = tuple(A)
    permutations = set()
    curr_count = 0
    while curr_count < num_samples and giveup > 0:
        giveup -= 1
        perm = tuple(rand_shuffle(A))
        if A == perm:
            continue
        if is_tile and not tile_prob_solvable(perm):
            continue
        permutations.add(perm)
        curr_count = len(permutations)
    return [list(p) for p in permutations]

def make_random_substitutions(A, substitutions, num_samples, giveup = 10000000):
    """ Return num_samples randomly substituted versions of list A that don't include A and arent duplicated"""
    A = tuple(A)
    length = len(A)
    perturbations = set()
    curr_count = 0
    while curr_count < num_samples and giveup > 0:
        giveup -= 1
        pert = tuple([random.choice(substitutions) for i in range(length)])
        if A == pert:
            continue
        perturbations.add(pert)
        curr_count = len(perturbations)
    return [list(p) for p in perturbations]



def bytes_to_gb(bytes_value):
    """    Converts bytes to gigabytes.    """
    return bytes_value / (1024**3)


def get_available_ram():
  """  Gets the amount of available RAM in GB.  """
  mem = psutil.virtual_memory()
  return round(bytes_to_gb(mem.available), 2)    # mem.available is the actual available memory that can be given instantly to processes



def get_size(obj):
    """ Get the amount of RAM occupied by a python object in GB 
        Note: for complex objects like a dictionary this is only the size of the base object not the memory taken by the items within it
    """
    return round(bytes_to_gb(sys.getsizeof(obj)), 4)

def encode_list(state):
    """ Encode a list or tuple of 1 char strings as a byte string """
    return "".join(tuple(state)).encode('utf-8')

def decode_list(bstr, tup=True):
    """ Decode a byte string into a list of 1 char strings """
    if tup: return tuple(bstr.decode('utf-8'))
    return list(bstr.decode('utf-8'))


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


def write_jsonl_to_csv(results, csv_file_path, del_keys=['path'], 
                       delimiter=',', lineterminator='\n', verbose=True):
    """ Write a list of dictionaries to a CSV file optionally deleting some keys and making the columns
        consistent across all rows by adding header as superset of all keys and adding blanks to rows where necessary.
        The order of the output columns will be the order of the keys in the first results dict minus any deleted keys plus any keys appearing in subsequent results dicts not in the first dict 
    """
    all_results = copy.deepcopy(results)
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
            if scenario['start_x'] is not None and scenario['start_y'] is not None:
                scenario['initial_state'] = [scenario['start_y'], scenario['start_x']]
            scenario['goal_state'] = None
            if scenario['goal_x'] is not None and scenario['goal_y'] is not None:
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
                                if convert_func == json.loads:  # special case for json.loads
                                    problem[col_name] = convert_func(col_value.replace("'", '"'))  # replace single quotes with double quotes for valid JSON
                                else:    
                                    problem[col_name] = convert_func(col_value)
                            except Exception as e:
                                print(f"load_csv_file: Error converting column {col_name} with value {col_value} to {convert_func}: {e}")
    return data


def save_to_json(data, filename, verbose=False):
    """ Saves a Python object to a JSON file.  """
    try:
        # Ensure the directory exists if the filename includes a path
        dirname = os.path.dirname(filename)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(filename, 'w', encoding='utf-8') as f:
            # Use json.dump to write the data to the file
            # indent=4 makes the output pretty-printed
            json.dump(data, f, ensure_ascii=False, indent=4)
        if verbose:    
            print(f"Data successfully saved to {filename}")
        return True
    except TypeError as e:
        print(f"ERROR saving data to {filename}: Object of type {e} is not JSON serializable.")
    except IOError as e:
        print(f"ERROR saving data to {filename}: {e}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while saving to {filename}: {e}")
    return False

def load_from_json(filename, verbose=False):
    """  Loads a Python object from a JSON file.  
    Returns:
        The loaded Python object, or None if an error occurred.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Error loading data: File not found at {filename}")

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            # Use json.load to read the data from the file
            data = json.load(f)
        if verbose:
            print(f"Data successfully loaded from {filename}")
        return data
    except json.JSONDecodeError as e:
        print(f"Error loading data from {filename}: Invalid JSON format - {e}")
    except IOError as e:
        print(f"Error loading data from {filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while loading from {filename}: {e}")
    return None


def run_search(algorithm, problem, seed=None, logger=None, save_path=True):
    """ Run an algorithm on a problem and return results """
    if not logger: log = print
    else: log = logger.info
    if seed: random.seed(seed)
    result = None
    try:
        result = algorithm.search(problem) # Call the runner
        result['algorithm'] = str(algorithm)  # Set algorithm name in result consistently
        log(f"{str(algorithm)} Done. Time: {result.get('time', -1):.4f}secs Nodes Expanded: {result.get('nodes_expanded', -1)} Path Cost: {result.get('cost', 'N/A')} Length: {len(result['path']) if result['path'] else 'No Path Found'}")

    except Exception as e:
        log(f"!!! ERROR during {str(algorithm)} on {str(problem)}: {e}")
        log(traceback.format_exc())
        #traceback.print_exc() 
        result = { "path_length": -1, "cost": -1, "nodes_expanded": -1, "time": -1, 
                    "algorithm": str(algorithm), "status": str(e)}
    if result: 
        if 'status' not in result:
            result['status'] = 'No status supplied from algorithm.'
        result['problem'] = str(problem)
        if 'path' in result and result['path']:
            result['unit_cost'] = len(result['path']) - 1
            result['path_length'] = len(result['path'])
        else:
            result['unit_cost'] = -1
            result['path_length'] = -1
        if not save_path:  # if not saving path, remove it from the result
            del result['path']
        result['seed'] = seed  # store seed in result for reproducibility and also so we can aggregate results over sampled runs
        log(f"{ {key: value for key, value in result.items() if key !='path'} }") # log result without path to avoid cluttering the log too much
    return result        
        

def run_experiments(problems, algorithms, out_dir, out_prefix='search_eval', 
                    seed=42, timestamp=None, logger=None, save_path=True):
    """ Run a set of algorithms on a set of problems and save the results to a CSV file (without path)
        and a json file (with path) in the specified output directory.
    Args:
        problems (list): List of problems to solve
        algorithms (list): List of algorithms to use
        output_dir (str): Directory to save the results
    """
    if not logger: log = print
    else: log = logger.info
    if not timestamp:
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    out_file_base = os.path.join(out_dir,f"{out_prefix}_{timestamp}")
    json_file_path = f"{out_file_base}.json"
    csv_file_path = f"{out_file_base}.csv"
    total_experiments = len(problems) * len(algorithms)
    curr_experiment = 0
    log(f"Running {total_experiments} experiments with {len(problems)} problems and {len(algorithms)} algorithms")

    all_results = []
    for problem in problems:  # For each problem
        log(f"\n{'=' * 20}")
        log(f"Solving: {problem}")
        if hasattr(problem, "initial_state_tuple"):
            log(f"Initial State: {problem.initial_state_tuple}")
        else:
            log(f"No tuple representation in problem: initial_state_tuple. This should be remedied")
        if hasattr(problem, "goal_state_tuple"):
            log(f"Goal State:    {problem.goal_state_tuple}")
        else:
            log(f"No tuple representation in problem: goal_state_tuple. This should be remedied")
        log(f"Initial Heuristic: {problem.heuristic(problem.initial_state())}")
        log(f"{'-' * 20}")
        
        for algorithm in algorithms:  # For each algorithm
            curr_experiment += 1
            log(f"Experiment: {curr_experiment}/{total_experiments} Running {str(algorithm)} on {str(problem)}...")
            log(f"Available RAM (GB) before experiment: {get_available_ram()}")
            result = run_search(algorithm, problem, seed=seed, logger=logger, save_path=save_path)  # Run the search
            if result:
                all_results.append(result)
                with open(json_file_path, 'w') as json_file:                                        # output results as we go
                    json.dump(all_results, json_file, indent=4)                                     # solution path in json 
                log(f"In progress results saved to {json_file_path}") 
                write_jsonl_to_csv(all_results, csv_file_path, del_keys=['path'], verbose=False)    # solution path not in csv
                log(f"In progress results saved to {csv_file_path}") 
            log(f"Finished experiment:{curr_experiment}/{total_experiments} Available RAM (GB) after experiment: {get_available_ram()}")
        problem = None  # Clear problem to free up memory

    # Overall Summary
    log(f"\n{'*'*15} Overall Summary {'*'*15}")
    for res in all_results:
        if res.get('path_length', -1) >= 0:
            summary = f"Cost: {res.get('cost', 'N/A')} Length: {len(res['path_length'])}"
        else: 
            summary = "No Path Found"
        #log("Path:", res.get('path') ) # Uncomment to see the full path states if saved
        optimal_note = f"(Optimal: {res['optimal']})" if 'optimal' in res else ""
        algo_name = res.get('algorithm','N/A') 
        log(f"- Problem: {res.get('problem','N/A')}, Algorithm: {algo_name}, Time: {res.get('time',-1):.4f}s, Nodes: {res.get('nodes_expanded',-1)}, {summary} {optimal_note} {res['status']}")

    # --- Save Results ---
    with open(json_file_path, 'w') as json_file:
        json.dump(all_results, json_file, indent=4)
    log(f"Final Results saved to {json_file_path}") 
    write_jsonl_to_csv(all_results, csv_file_path, del_keys=['path'])
    log("Finished!")
    return csv_file_path


def find_cstar(algorithm, problems, csv_list, file, seed=None, logger=None):
    """ Find cstar for a set of problems, updating the corresponding csv file after each search """
    if not logger: log = print
    else: log = logger.info
    warnings = 0
    for i, problem in enumerate(problems):
        log(f"Running A* to get C* for problem {i} {csv_list[i]["initial_state"]}...")
        result = run_search(algorithm, problem, seed=seed, logger=logger)
        if result and result.get('cost') is not None:
            if result['cost'] < 0:
                log(f"WARNING: C* search failed. No solution found so unable to record C* for problem {i} {csv_list[i]["initial_state"]}")
                warnings += 1
                continue
            csv_list[i]["cstar"] = result.get('cost')
            write_jsonl_to_csv(csv_list, file, del_keys=None, delimiter=';', verbose=False)
            log(f'Updated C* in file {file} for problem {i} {csv_list[i]["initial_state"]} to {csv_list[i]["cstar"]}')
        else:
            warnings += 1
            log(f"ERROR: Search failed. Unable to record C* for problem {i} {csv_list[i]["initial_state"]}")
    if warnings == 0:        
        log(f"C* search completed. {file} was updated for all problems.")
    else:
        log(f"C* search completed with WARNINGS. {file} was not updated for {warnings} problems where search failed!!'")
    return

