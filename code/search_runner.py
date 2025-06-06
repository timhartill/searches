"""
This code implements various search algorithms for solving 
  - Sliding Tile, Pancake, Pathfinder and Towers of Hanoi problems.
Using:  
Dijkstra/Uniform cost (g), Best first (h) ,A* f=g+h, 
Bidirectional A*/UC/BFS, Bidirectional LBPairs
MCTS, Heuristic MCTS

Author: Tim Hartill
Some code generated from Gemini 2.5.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import os
import sys
import random
import time
import json
import argparse
import logging

import util

# problems
import problem_puzzle
import problem_spatial

#from problem_puzzle import SlidingTileProblem, PancakeProblem, TowersOfHanoiProblem
#from problem_spatial import GridProblem

# searches
from search_mcts import heuristic_mcts_search
from search_unidirectional import generic_search
from search_bidirectional import bd_generic_search, bd_lb_search

# map search algorithm input string to a search class and it's parameters
# Edit this dict to add new algorithms and if necessary add "from newsearch import supersearch" above
SEARCH_MAP = {
    "astar": {"class":generic_search, "priority_key": 'f', "tiebreaker1": 'NONE', "tiebreaker2": 'NONE'},
    "astar_negg": {"class":generic_search, "priority_key": 'f', "tiebreaker1": '-g', "tiebreaker2": 'NONE'},
    "astar_fifo": {"class":generic_search, "priority_key": 'f', "tiebreaker1": 'FIFO', "tiebreaker2": 'NONE'},
    "uc": {"class":generic_search, "priority_key": 'g', "tiebreaker1": 'NONE', "tiebreaker2": 'NONE'},
    "huc": {"class":generic_search, "priority_key": 'g', "tiebreaker1": 'f', "tiebreaker2": 'NONE'},
    "bfs": {"class":generic_search, "priority_key": 'h', "tiebreaker1": 'g', "tiebreaker2": 'NONE'},
    "bd_astar": {"class":bd_generic_search, "priority_key": 'f', "tiebreaker1": '-g', "tiebreaker2": 'NONE'},
    "bd_uc": {"class":bd_generic_search, "priority_key": 'g', "tiebreaker1": 'NONE', "tiebreaker2": 'NONE'},
    "bd_huc": {"class":bd_generic_search, "priority_key": 'g', "tiebreaker1": 'f', "tiebreaker2": 'NONE'},
    "bd_bfs": {"class":bd_generic_search, "priority_key": 'h', "tiebreaker1": 'g', "tiebreaker2": 'NONE'},
    "lb_nbs_a": {"class": bd_lb_search, "tiebreaker1": 'NBS', "tiebreaker2": 'NONE', "version": 'A', "min_edge_cost": 0.0},  
    "lb_nbs_f": {"class": bd_lb_search, "tiebreaker1": 'NBS', "tiebreaker2": 'NONE', "version": 'F', "min_edge_cost": 0.0},  
    "lb_nbs_a_eps": {"class": bd_lb_search, "tiebreaker1": 'NBS', "tiebreaker2": 'NONE', "version": 'A', "min_edge_cost": 1.0},  
    "lb_nbs_f_eps": {"class": bd_lb_search, "tiebreaker1": 'NBS', "tiebreaker2": 'NONE', "version": 'F', "min_edge_cost": 1.0},  
    # for MCTS "heuristic_weight" > 0 indicates heuristic weight in selection. The actual value will then come from args.algo_mcts_heur_weight 
    "mcts_noheur": {"class":heuristic_mcts_search, "heuristic_weight": 0.0, "heuristic_rollout": False},
    "mcts_selectheur": {"class":heuristic_mcts_search, "heuristic_weight": 100.0, "heuristic_rollout": False},
    "mcts_rolloutheur": {"class":heuristic_mcts_search, "heuristic_weight": 0.0, "heuristic_rollout": True},
    "mcts_bothheur": {"class":heuristic_mcts_search, "heuristic_weight": 100.0, "heuristic_rollout": True},
}


# --- Main Execution Logic ---
if __name__ == "__main__":
    #sys.argv = ['']  # For VS Code interactive window run this before the below argparse code to work around invalid error that occurs
    parser = argparse.ArgumentParser(description="Search Algorithm Comparison Runner")
    parser.add_argument("--out_dir", default='/media/tim/dl3storage/gitprojects/searches/outputs', type=str,
                        help="Full path to output directory. CSV and JSON output files will be written here.")  
    parser.add_argument("--out_prefix", default='search-eval', type=str,
                        help="Log, CSV and JSON output file prefix. Date and time will be added to make unique.")  
    parser.add_argument("--in_dir", default='/media/tim/dl3storage/gitprojects/searches/problems', type=str,
                        help="Full path to input directory BASE. Expected subdirs off here are matrices, pancake, tile and toh. matrices should have np-scen and np-map and eg dao-map and dao-scen etc off it.")
    parser.add_argument('--seed', default=42, type=int,
                        help="random seed. Used to set problem order and also to create random probs. If sample_count=0 then reset to this seed before each algo/problem run.")
    parser.add_argument('--sample_count', default=0, type=int,
                        help="Number of samples to run if > 0. random seed. If > 0, runs each algo on each problem sample_count times with seed=incrementing number from 0 to sample_count-1.")
    
    # Matrices / Grids params
    parser.add_argument("--grid_dir", default='matrices', type=str,
                        help="Grid subdir off in_dir.")
    parser.add_argument("--grid", nargs="*", default='NONE', type=str,
                        help="Space-separated list of the domain portion of the grid problems to run eg --grid dao mazes. 'NONE' means don't run. for each domain, this will be expanded to the ...matrices/dao-scen subdir and all .scen files in there will be attempted. Will look for corresponding grids in dao-maps subdir")
    parser.add_argument('--grid_max_per_scen', default=21, type=int,
                        help="Max number of problems to run from any ONE .scen file. Eg if 21 and 156 .scen files in chosen subdir we will run 21 * 156 problems in total")
    parser.add_argument('--grid_reverse_scen_order', action='store_true',
                        help="If set, reverse the order of entries in each scen file before selection of --grid_max_per_scen entries (noting that higher cstar problems tend to occur later in file i.e. files are ordered by c* buckets of ~10 problems)")
    parser.add_argument('--grid_random_scen_order', action='store_true',
                        help="If set, randomise the order of entries in each scen file before selection of --grid_max_per_scen entries (noting that higher cstar problems tend to occur later in file i.e. files are ordered by c* buckets of ~10 problems this will eliminate thate bias)")
    parser.add_argument('--grid_heur', nargs="*", default="octile", type=str, 
                        help="grid heuristics. Eg --grid_heur octile euclidean chebyshev manhattan")
    parser.add_argument('--grid_degs', nargs="*", default=0, type=int, 
                        help="grid heuristic degradation(s) to run. Eg 0 1 2 3. Degrades the heuristic by dividing it by degradation+1")
    parser.add_argument('--grid_inadmiss', action='store_true', 
                        help="grid heuristic admissable or inadmissable Eg --grid_inadmiss means make inadmissable heuristic.")
    parser.add_argument('--grid_cost_multipier', default=1.0, type=float,
                        help="Any number > 1.0 will multiply the unit cost, hence weakening the heuristic which isn't multiplied by this number.")
    parser.add_argument("--grid_allow_diag", action='store_true',
                        help="Allow diagonal movement in the grid problems. Default is False. Note when enabled this sets the variable cost flag.")
    parser.add_argument('--grid_diag_cost', default=1.5, type=float,
                        help="Cost of diagonal move before cost multiplication. Some HOG2 grid Cstar calculations in .scen files use 2.0, other .scen diag costs vary. Some papers use 1.5. Heuristic (and correct) estimate remains sqrt(2)=1.4142135623730951.")
    parser.add_argument("--grid_ignore_cstar", action='store_true',
                        help="If set the cstar in .scen files is not used. This is typically set when using a different diagonal cost than what the cstar in the .scen file was based on, or you are using a cost multiplier other than 1.")
    
    # Sliding Tiles params
    parser.add_argument("--tiles_dir", default='tile', type=str,
                        help="Tiles subdir off in_dir.")
    parser.add_argument("--tiles", default='NONE', type=str,
                        help="File name of the sliding tile problems to run eg 15_puzzle_korf_std100.csv or 'NONE' to skip. Should be in the tiles subdir.")
    parser.add_argument('--tiles_max', default=100, type=int,
                        help="Max number of tile problems to run from the chosen tile file. Eg if 100 and 1000 problems are in the file, we will run 100 tile problems in total")
    parser.add_argument('--tiles_heur', nargs="*", default="manhattan", type=str, 
                        help="tiles heuristics. Only manhattan implemented. Eg --tiles_heur manhattan")
    parser.add_argument('--tiles_degs', nargs="*", default=0, type=int, 
                        help="tiles heuristic degradation(s) to run. Eg 0 1 2 3")
    parser.add_argument('--tiles_inadmiss', action='store_true', 
                        help="tiles heuristic admissable or inadmissable Eg --tiles_inadmiss means make inadmissable heuristic.")
    parser.add_argument("--tiles_var_cost", action='store_true',
                        help="When enabled this uses the tile value as the cost rather than 1. Default is false.")
    parser.add_argument("--tiles_ignore_cstar", action='store_true',
                        help="If set the cstar in .csv files is not used. This is typically set when using variable costs or an inadmissable heuristic.")
    parser.add_argument('--tiles_make_random', default=0, type=int,
                        help="Create this number of tile problems to run and save back as standard csv into the tiles_dir subdir")
    parser.add_argument('--tiles_make_size', default=16, type=int,
                        help="Size of state for random tile puzzles to be created.")
    parser.add_argument("--tiles_add_cstar", action='store_true',
                        help="If set, cstar for randomly created puzzles is added into output .csv file. This may of course take far longer.")
    
    # Pancake params
    parser.add_argument("--pancakes_dir", default='pancake', type=str,
                        help="Pancakes subdir off in_dir.")
    parser.add_argument("--pancakes", default='NONE', type=str,
                        help="File name of the pancake problems to run or 'NONE' to skip. Should be in the pancake subdir.")
    parser.add_argument('--pancakes_max', default=50, type=int,
                        help="Max number of pancake problems to run from the chosen pancake file. Eg if 100 and 1000 problems in the file we will run 100 pancake problems in total")
    parser.add_argument('--pancakes_heur', nargs="*", default="symgap", type=str, 
                        help="pancakes heuristics. Only symmetric gap implemented. Eg --pancakes_heur symgap")
    parser.add_argument('--pancakes_degs', nargs="*", default=0, type=int, 
                        help="pancakes heuristic degradation(s) to run. Eg 0 1 2 3")
    parser.add_argument('--pancakes_inadmiss', action='store_true', 
                        help="pancakes heuristic admissable or inadmissable Eg --pancakes_inadmiss means make inadmissable heuristic.")
    parser.add_argument("--pancakes_var_cost", action='store_true',
                        help="When enabled this uses the num pancakes flipped as the cost rather than 1. Default is false.")
    parser.add_argument("--pancakes_ignore_cstar", action='store_true',
                        help="If set the cstar in .csv files is not used. This is typically set when using variable costs or an inadmissable heuristic.")
    parser.add_argument('--pancakes_make_random', default=0, type=int,
                        help="Create this number of pancake problems to run and save back as standard csv into the pancakes_dir subdir")
    parser.add_argument('--pancakes_make_size', default=14, type=int,
                        help="Size of state for random pancake puzzles to be created. Note this is WITHOUT the table. So 4-pancake could have input state (1,4,3,2) to which the table will be added dynamically when the problem is run.")
    parser.add_argument("--pancakes_add_cstar", action='store_true',
                        help="If set, cstar for randomly created puzzles is added into output .csv file. This may of course take far longer.")

    # Tower of Hanoi params
    parser.add_argument("--toh_dir", default='toh', type=str,
                        help="Toh subdir off in_dir.")
    parser.add_argument("--toh", default='NONE', type=str,
                        help="File name of the towers of hanoi problems to run or 'NONE' to skip. Should be in the toh subdir.")
    parser.add_argument('--toh_max', default=50, type=int,
                        help="Max number of toh problems to run from the chosen toh file. Eg if 100 and 1000 problems in the file we will run 100 toh problems in total")
    parser.add_argument('--toh_heur', nargs="*", default="infinitepegrelaxation", type=str, 
                        help="toh heuristics. pdb_P_X_Y, infinitepegrelaxation and 3pegstd implemented eg --toh_heur 3pegstd infinitepegrelaxation pdb_4_10+2 pdb_4_6+6. For pdb_P_X_Y: P = peg count and X+Y = number of disks. PDBs are cached in the toh subdir, to recreate just delete the pdb subdirectory.")
    parser.add_argument('--toh_degs', nargs="*", default=0, type=int, 
                        help="toh heuristic degradation(s) to run. Eg 0 1 2 3")
    parser.add_argument('--toh_inadmiss', action='store_true', 
                        help="toh heuristic admissable or inadmissable Eg --toh_inadmiss means make inadmissable heuristic.")
    parser.add_argument("--toh_ignore_cstar", action='store_true',
                        help="If set the cstar in .csv files is not used. This is typically set when using variable costs or an inadmissable heuristic.")
    parser.add_argument('--toh_make_random', default=0, type=int,
                        help="Create this number of toh problems to run and save back as standard csv into the toh_dir subdir")
    parser.add_argument('--toh_num_disks', default=12, type=int,
                        help="Size of state (number of disks) for random toh puzzles to be created.")
    parser.add_argument('--toh_num_pegs', default=4, type=int,
                        help="Number of pegs for random toh puzzles to be created.")
    parser.add_argument("--toh_add_cstar", action='store_true',
                        help="If set, cstar for randomly created puzzles is added into output .csv file. This may of course take far longer. To enable this you must set --toh_heur to the heuristic to use, typically a pdb eg --toh_heur pdb_4_10+2")
    
    # Algorithm params over all searches
    parser.add_argument('--algo_timeout', default=30.0, type=float,
                        help="Maximum time in minutes to allow any algorithm to run for. If exceeded, statistics to that point are returned along with status of 'timeout'. Normal return status = 'completed'. It is important to set this value such that no algorithm OOMs on the particular machine running the experiments.") 
    parser.add_argument('--algo_min_remaining_gb', default=2.0, type=float,
                        help="Minimum GB RAM remaining before algorithm is killed.") 
    parser.add_argument('--algo_visualise', action='store_true', 
                        help="Output .png files showing nodes expanded, path, meeting point etc for each algorithm and problem type that supports this.")
    parser.add_argument('--algo_save_path_in_json', action='store_true', 
                        help="If set, the path from start to goal is saved in the JSON output file otherwise it is not, typically to save memory on runs of many algorithms and many problems. Note that the CSV output file never includes the path.")

    # Heuristic search args
    parser.add_argument('--algo_heur', nargs="*", default="astar bd_astar", type=str, 
                        help="which unidirectional and bidirectional heuristic searches to run. Pass NONE to not run any: eg --algo_heur astar us bfs bd_astar. Will set priority key to g+h, g and/or h appropriately. --algo_heur names must be specified in SEARCH_MAP at the top of search_runner.py")

    # Monte Carlo Tree Search (MCTS) args
    parser.add_argument('--algo_mcts', nargs="*", default="mcts_noheur mcts_bothheur", type=str, 
                        help="which MCTS searches to run. Pass NONE to not run any: eg --algo_mcts NONE")
    parser.add_argument('--algo_mcts_iterations', default=100, type=int,
                        help="Number of MCTS iterations to be run for each MCTS algorithm.") 
    parser.add_argument('--algo_mcts_max_depth', default=150, type=int, 
                        help="Maximum depth of MCTS Tree.")
    parser.add_argument('--algo_mcts_exploration_weight', default=1.41, type=float, 
                        help="MCTS Exploration Weight")
    parser.add_argument('--algo_mcts_heur_weight', default=100.0, type=float, 
                        help="MCTS Heuristic Weight (if applicable)")

    args = parser.parse_args()
    # parser.print_help()

    # Set blank args to NONE for consistency
    if args.tiles.strip() == "": args.tiles = 'NONE'
    if args.pancakes.strip() == "": args.pancakes = 'NONE'
    if args.toh.strip() == "": args.toh = 'NONE'
    
    # Set up output directories if they don't exist
    os.makedirs(args.out_dir, exist_ok=True)
    args.visualise_dir = os.path.join(args.out_dir, 'visualise')
    os.makedirs(args.visualise_dir, exist_ok=True)
    if args.tiles.upper() != "NONE":
        args.tiles_file_full = os.path.join(args.in_dir, args.tiles_dir, args.tiles)
        assert os.path.exists(args.tiles_file_full)
    if args.pancakes.upper() != "NONE":
        args.pancakes_file_full = os.path.join(args.in_dir, args.pancakes_dir, args.pancakes)
        assert os.path.exists(args.pancakes_file_full)
    if args.toh.upper() != "NONE":
        args.toh_file_full = os.path.join(args.in_dir, args.toh_dir, args.toh)
        assert os.path.exists(args.toh_file_full)
    if args.grid[0].upper() != 'NONE':
        args.grid_dir_full = []
        for domain in args.grid:
            grid_scen_dir = domain + "-scen"
            args.grid_dir_full.append( os.path.join(args.in_dir, args.grid_dir, grid_scen_dir) )
            assert os.path.exists(args.grid_dir_full[-1])

    args.timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f"{args.out_prefix}_{args.timestamp}.log"
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.FileHandler(os.path.join(args.out_dir, log_filename)),
                              logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(f"Running search comparison at {args.timestamp}")
    logger.info(args)


    # Set up the problems to be run ###################################
    tile_list, pancake_list, toh_list, grid_list = [], [], [], []
    if args.tiles.upper() != "NONE":
        random.seed(args.seed)
        logger.info("Loading Tile problems...")
        tile_list = problem_puzzle.create_tile_probs(args)
        logger.info(f"Created {len(tile_list)} tile problems from {args.tiles_file_full}")
    if args.pancakes.upper() != "NONE":
        random.seed(args.seed)
        logger.info("Loading Pancake problems...")
        pancake_list = problem_puzzle.create_pancake_probs(args)
        logger.info(f"Created {len(pancake_list)} pancake problems from {args.pancakes_file_full}")
    if args.toh.upper() != "NONE":
        random.seed(args.seed)
        logger.info("Loading Towers of Hanoi problems...")
        toh_list = problem_puzzle.create_toh_probs(args)
        logger.info(f"Created {len(toh_list)} Towers of Hanoi problems from {args.toh_file_full}")
    if args.grid[0].upper() != 'NONE':  # NOTE: grids are currently the only prob type that uses random and seed is set in create_grid_probs for each domain
        logger.info("Loading Grid problems...")
        grid_list = problem_spatial.create_grid_probs(args)
        logger.info(f"Created {len(grid_list)} grid problems from {args.grid_dir_full}")

    problems = toh_list + grid_list + pancake_list + tile_list

    logger.info("######")
    logger.info(f"The following {len(problems)} problems will be run:")
    for prob in problems:
        logger.info(str(prob))

    random.seed(args.seed)

    algorithms = []
    # Set up the heuristic algorithms to be run ########################
    # Heuristic algorithms must accept the parameters as shown here.. 
    if args.algo_heur[0].upper() != "NONE":
        for algo in args.algo_heur:
            assert algo in SEARCH_MAP
            algo_class = SEARCH_MAP[algo]['class']
            if algo.startswith('lb_'):
                # LB Pairs-based algorithms
                algo_instance = algo_class(tiebreaker1 = SEARCH_MAP[algo]['tiebreaker1'],
                                           tiebreaker2 = SEARCH_MAP[algo]['tiebreaker2'],
                                           visualise = args.algo_visualise,
                                           visualise_dirname = args.visualise_dir,
                                           min_ram = args.algo_min_remaining_gb,
                                           timeout = args.algo_timeout,
                                           version=SEARCH_MAP[algo]['version'],
                                           min_edge_cost=SEARCH_MAP[algo]['min_edge_cost'])
            else:
                algo_instance = algo_class(priority_key = SEARCH_MAP[algo]['priority_key'],
                                        tiebreaker1 = SEARCH_MAP[algo]['tiebreaker1'],
                                        tiebreaker2 = SEARCH_MAP[algo]['tiebreaker2'],
                                        visualise = args.algo_visualise,
                                        visualise_dirname = args.visualise_dir,
                                        min_ram = args.algo_min_remaining_gb,
                                        timeout = args.algo_timeout)
            algorithms.append(algo_instance)
    else:
        logger.info("Not running any heuristic algorithms.")

    random.seed(args.seed)

    # Set up the MCTS algorithms to be run ########################
    # MCTS algorithms must accept the parameters as shown here.. 
    if args.algo_mcts[0].upper() != "NONE":
        for algo in args.algo_mcts:
            assert algo in SEARCH_MAP
            algo_class = SEARCH_MAP[algo]['class']
            hw = 0.0
            if SEARCH_MAP[algo]['heuristic_weight'] != 0.0:
                hw = args.algo_mcts_heur_weight
            algo_instance = algo_class(iterations = args.algo_mcts_iterations,
                                       max_depth = args.algo_mcts_max_depth,
                                       exploration_weight = args.algo_mcts_exploration_weight,
                                       heuristic_weight = hw,
                                       heuristic_rollout = SEARCH_MAP[algo]['heuristic_rollout'],
                                       epsilon=1e-6)
            algorithms.append(algo_instance)
    else:
        logger.info("Not running any MCTS algorithms.")

    logger.info("######")
    if len(algorithms) > 0:
        logger.info(f"Running the following {len(algorithms)} algorithms:")
        for a in algorithms:
            logger.info(str(a))

        # --- Run Experiments ---
        if sample_count == 0:
            util.run_experiments(problems, algorithms, args.out_dir, args.out_prefix, 
                                seed=args.seed, timestamp=args.timestamp, logger=logger, save_path=args.algo_save_path_in_json)
        else:
            logger.info(f"Running each algorithm {args.sample_count} times on each problem with seed set to 0, 1, ..., {args.sample_count-1}.")
            for i in range(args.sample_count):
                util.run_experiments(problems, algorithms, args.out_dir, args.out_prefix, 
                                    seed=i, timestamp=args.timestamp, logger=logger, save_path=args.algo_save_path_in_json)

        logger.info(f"Finished search comparison at {time.strftime('%Y-%m-%d %H:%M:%CS')}")


    # --- Create random problems ----
    algo = 'astar'  # Use astar to determine cstar, if specified 
    algo_class = SEARCH_MAP[algo]['class']
    algo_instance = algo_class(priority_key = SEARCH_MAP[algo]['priority_key'],
                                tiebreaker1 = SEARCH_MAP[algo]['tiebreaker1'],
                                tiebreaker2 = SEARCH_MAP[algo]['tiebreaker2'],
                                visualise = False,
                                min_ram = args.algo_min_remaining_gb,
                                timeout = args.algo_timeout)

    if args.tiles_make_random > 0:
        random.seed(args.seed)
        logger.info(f"Creating {args.tiles_make_random} Sliding Tile problems of state size {args.tiles_make_size} with cstar {args.tiles_add_cstar}...")
        goal_state = list(range(0, args.tiles_make_size)) # 0 at beginning per Korf standard
        states_list = util.make_random_permutations(goal_state, num_samples=args.tiles_make_random, is_tile=True)
        logger.info(f"Goal state: {goal_state}")
        file = os.path.join(args.in_dir, args.tiles_dir, f"{args.tiles_make_size-1}_puzzle_probs{args.tiles_make_random}_seed{args.seed}_{args.timestamp}.csv")
        problems = []
        csv_list = []
        for state in states_list: 
            csv_list.append({"problem_type":'tile', "initial_state": state, "goal_state": goal_state, "cstar": None })
            problems.append(problem_puzzle.SlidingTileProblem(initial_state=state, goal_state=goal_state,
                                                        use_variable_costs=False, make_heuristic_inadmissable=False, degradation=0,
                                                        heuristic=args.tiles_heur[0], cstar=None, file=file) )
        f = util.write_jsonl_to_csv(csv_list, file, del_keys=None, delimiter=';', verbose=False)
        logger.info(f"Tile probs without C* written to: {f}")
        if args.tiles_add_cstar:
            util.find_cstar(algo_instance, problems, csv_list, file, seed=None, logger=logger)

    if args.pancakes_make_random > 0:
        random.seed(args.seed)
        logger.info(f"Creating {args.pancakes_make_random} Pancake problems of size {args.pancakes_make_size} with cstar {args.pancakes_add_cstar}...")
        goal_state = list(range(1, args.pancakes_make_size+1)) # exclude table
        states_list = util.make_random_permutations(goal_state, num_samples=args.pancakes_make_random)
        logger.info(f"Goal state excl table: {goal_state}")
        file = os.path.join(args.in_dir, args.pancakes_dir, f"{args.pancakes_make_size}_pancake_probs{args.pancakes_make_random}_seed{args.seed}_{args.timestamp}.csv")
        problems = []
        csv_list = []
        for state in states_list: 
            csv_list.append({"problem_type":'pancake', "initial_state": state, "goal_state": goal_state, "cstar": None })
            problems.append(problem_puzzle.PancakeProblem(initial_state=state, goal_state=goal_state,
                                                        use_variable_costs=False, make_heuristic_inadmissable=False, degradation=0,
                                                        heuristic=args.pancakes_heur[0], cstar=None, file=file) )
        f = util.write_jsonl_to_csv(csv_list, file, del_keys=None, delimiter=';', verbose=False)        
        logger.info(f"Pancake probs without C* written to: {f}")
        if args.pancakes_add_cstar:
            util.find_cstar(algo_instance, problems, csv_list, file, seed=None, logger=logger)

    if args.toh_make_random > 0:
        random.seed(args.seed)
        logger.info(f"Creating {args.toh_make_random} Towers of Hanoi problems with {args.toh_num_disks} disks and {args.toh_num_pegs} pegs with cstar {args.toh_add_cstar}...")
        goal_state = [chr(ord('A') + args.toh_num_pegs-1)] * args.toh_num_disks
        pegs = ['A']
        for i in range(1, args.toh_num_pegs): 
            pegs.append(chr(ord('A') + i))
        states_list = util.make_random_substitutions(goal_state, substitutions=pegs, num_samples=args.toh_make_random)
        logger.info(f"Goal state: {goal_state} Pegs:{pegs}")
        file = os.path.join(args.in_dir, args.toh_dir, f"{args.toh_num_disks}_toh_{args.toh_num_pegs}_peg_probs{args.toh_make_random}_seed{args.seed}_{args.timestamp}.csv")
        problems = []
        csv_list = []
        for state in states_list: 
            csv_list.append({"problem_type":'toh', "initial_state": state, "goal_state": goal_state, "cstar": None })
            problems.append(problem_puzzle.TowersOfHanoiProblem(initial_state=state, goal_state=goal_state,
                                                        make_heuristic_inadmissable=False, degradation=0,
                                                        heuristic=args.toh_heur[0], cstar=None, file=file) )
        f = util.write_jsonl_to_csv(csv_list, file, del_keys=None, delimiter=';', verbose=False)
        logger.info(f"Tower of Hanoi probs without C* written to: {f}")
        if args.toh_add_cstar:
            util.find_cstar(algo_instance, problems, csv_list, file, seed=None, logger=logger)


    logger.info(f"Finished at {time.strftime('%Y-%m-%d %H:%M:%CS')}")

    """
    # --- Example Problems ---
    tile_initial = [8, 6, 7, 2, 5, 4, 3, 0, 1] # harder 3x3 unit C*=27
    #tile_initial = [15, 11, 12, 14, 9, 13, 10, 8, 6, 7, 2, 5, 4, 3, 0, 1] # harderer 4x4 unit C*= >>31 A* ran out of memory @ 48GB
    sliding_tile_unit_cost = SlidingTileProblem(initial_state=tile_initial, 
                                                use_variable_costs=False, 
                                                make_heuristic_inadmissable=False,
                                                degradation=0, cstar=27)
    tile_initial = [8, 6, 7, 2, 5, 4, 3, 0, 1, 10, 11, 9] # harder 4x3 unit C*=37
    sliding_tile_unit_cost43 = SlidingTileProblem(initial_state=tile_initial, 
                                                use_variable_costs=False, 
                                                make_heuristic_inadmissable=False,
                                                degradation=0, cstar=37)
    tile_scenarios = util.load_csv_file('/media/tim/dl3storage/gitprojects/searches/problems/tile/15_puzzle_probs100_korf_std.csv')
    sliding_tile_korf8 = SlidingTileProblem(initial_state=tile_scenarios[8]['initial_state'], 
                                                use_variable_costs=False, 
                                                make_heuristic_inadmissable=False,
                                                degradation=0, cstar=tile_scenarios[8]['cstar'])
    pancake_initial = (8, 3, 5, 1, 6, 4, 2, 7) # C*=8
    pancake_unit_cost = PancakeProblem(initial_state=pancake_initial, 
                                       use_variable_costs=False,
                                       make_heuristic_inadmissable=False,
                                       degradation=0, cstar=8)
    pancake_scenarios = util.load_csv_file('/media/tim/dl3storage/gitprojects/searches/problems/pancake/14_pancake_probs1_test.csv')
    pancake_unit_cost_14 = PancakeProblem(initial_state=pancake_scenarios[0]['initial_state'], 
                                       use_variable_costs=False,
                                       make_heuristic_inadmissable=False,
                                       degradation=0, cstar=pancake_scenarios[0]['cstar'])  # c* = 13
    hanoi_problem_4Tower_InfPeg = TowersOfHanoiProblem(initial_state= ['A','A','A','A','A','A','A'],
                                                       goal_state = ['D','D','D','D','D','D','D'],
                                                       make_heuristic_inadmissable=False,  heuristic='InfinitePegRelaxation',
                                                       degradation=0, cstar=25)
    hanoi_problem_4Tower_InfPeg_state2 = TowersOfHanoiProblem(initial_state= ['B', 'B', 'C', 'C', 'D', 'A', 'A'],
                                                       goal_state = ['D','D','D','D','D','D','D'],
                                                       make_heuristic_inadmissable=False,  heuristic='InfinitePegRelaxation',
                                                       degradation=hanoi_degradation, cstar=18)
    grid_scenarios = util.load_scen_file('/media/tim/dl3storage/gitprojects/searches/problems/matrices/np-scen/matrix_1000yX1000x.scen')
    grid_harder1000x1000_unit_diag_mh_d0_cm1 = GridProblem(grid_scenarios[0]['map_dir'],
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=False, heuristic='manhattan')
    # Load grid problems from .map files
    grid_scenarios = util.load_scen_file('/media/tim/dl3storage/gitprojects/searches/problems/matrices/dao-scen/brc505d.map.scen')    
    grid_dao1 = GridProblem(grid_scenarios[2]['map_dir'], 
                            initial_state=[grid_scenarios[2]['start_y'], grid_scenarios[2]['start_x']], 
                            goal_state=[grid_scenarios[2]['goal_y'], grid_scenarios[2]['goal_x']], 
                            cost_multiplier=1,
                            make_heuristic_inadmissable=False, degradation=0,
                            allow_diagonal=True, 
                            heuristic='octile')
    grid_scenarios = util.load_scen_file('/media/tim/dl3storage/gitprojects/searches/problems/matrices/maze-scen/maze512-1-6.map.scen')    
    grid_dao5 = GridProblem(grid_scenarios[12346]['map_dir'], 
                            initial_state=[grid_scenarios[12346]['start_y'], grid_scenarios[12346]['start_x']], 
                            goal_state=[grid_scenarios[12346]['goal_y'], grid_scenarios[12346]['goal_x']], 
                            cost_multiplier=1,
                            make_heuristic_inadmissable=False, degradation=0,
                            allow_diagonal=True, diag_cost=2.0,
                            heuristic='octile', cstar=grid_scenarios[12346]['cstar'])


    # --- Define Algorithms ie give algorithm setups with differing params, unique fn names ---
    run_ucs = generic_search(priority_key='g', tiebreaker1='NONE', tiebreaker2='NONE', visualise=True, visualise_dirname=args.visualise_dir)
    run_hucs = generic_search(priority_key='g', tiebreaker1='f', tiebreaker2='NONE', visualise=True, visualise_dirname=args.visualise_dir)
    run_greedy_bfs = generic_search(priority_key='h', tiebreaker1='g', tiebreaker2='NONE', visualise=True, visualise_dirname=args.visualise_dir)
    run_astar = generic_search(priority_key='f', tiebreaker1='-g', tiebreaker2='NONE', visualise=True, visualise_dirname=args.visualise_dir)
    run_bidir_astar = bd_generic_search(priority_key='f', tiebreaker1='-g', tiebreaker2='NONE', visualise=True, visualise_dirname=args.visualise_dir)
    run_bidir_ucs = bd_generic_search(priority_key='g', tiebreaker1='NONE', tiebreaker2='NONE', visualise=True, visualise_dirname=args.visualise_dir)
    run_bidir_greedy = bd_generic_search(priority_key='h', tiebreaker1='g', tiebreaker2='NONE', visualise=True, visualise_dirname=args.visualise_dir)
    # Wrapper for standard MCTS (no heuristic guidance)
    run_mcts_standard = heuristic_mcts_search(iterations=iterations, max_depth=max_depth, 
                                              heuristic_weight=0.0, heuristic_rollout=False)
    # MCTS with heuristic in selection only
    run_mcts_h_select = heuristic_mcts_search(iterations=iterations, max_depth=max_depth, 
                                              heuristic_weight=heuristic_weight, heuristic_rollout=False) # Tune weight
    # MCTS with heuristic in rollout only
    run_mcts_h_rollout = heuristic_mcts_search(iterations=iterations, max_depth=max_depth, 
                                               heuristic_weight=0.0, heuristic_rollout=True)
    # MCTS with heuristic in both selection and rollout
    run_mcts_h_both = heuristic_mcts_search(iterations=iterations, max_depth=max_depth, 
                                            heuristic_weight=heuristic_weight, heuristic_rollout=True) # Tune weight


    algorithms = [
        run_astar,
        run_ucs,
        run_greedy_bfs,
        run_bidir_astar,
        run_bidir_ucs,
        run_bidir_greedy,
        run_mcts_standard,
    ]
"""

