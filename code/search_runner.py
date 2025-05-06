"""
This code implements various search algorithms for solving 
  - Sliding Tile, Pancake, Pathfinder and Towers of Hanoi problems.
Using:  
Dijkstra/Uniform cost (g), Best first (h) ,A* f=g+h, 
Bidirectional A*/UC/BFS, 
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
from search_bidirectional import bd_generic_search

# map search algorithm input string to a search class and it's parameters
# Edit this dict to add new algorithms and if necessary add "from newsearch import supersearch" above
SEARCH_MAP = {
    "astar": {"class":generic_search, "priority_key": 'f', "tiebreaker1": '-g', "tiebreaker2": 'NONE'},
    "uc": {"class":generic_search, "priority_key": 'g', "tiebreaker1": 'NONE', "tiebreaker2": 'NONE'},
    "huc": {"class":generic_search, "priority_key": 'g', "tiebreaker1": 'f', "tiebreaker2": 'NONE'},
    "bfs": {"class":generic_search, "priority_key": 'h', "tiebreaker1": 'g', "tiebreaker2": 'NONE'},
    "bd_astar": {"class":bd_generic_search, "priority_key": 'f', "tiebreaker1": '-g', "tiebreaker2": 'NONE'},
    "bd_uc": {"class":bd_generic_search, "priority_key": 'g', "tiebreaker1": 'NONE', "tiebreaker2": 'NONE'},
    "bd_huc": {"class":bd_generic_search, "priority_key": 'g', "tiebreaker1": 'f', "tiebreaker2": 'NONE'},
    "bd_bfs": {"class":bd_generic_search, "priority_key": 'h', "tiebreaker1": 'g', "tiebreaker2": 'NONE'},
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
                        help="random seed. Reset before running each algorithm on each problem.") 

    # Matrices / Grids params
    parser.add_argument("--grid_dir", default='matrices', type=str,
                        help="Grid subdir off in_dir.")
    parser.add_argument("--grid", default='', type=str,
                        help="Domain portion of the grid problems to run eg dao. '' means don't run. This will be expanded to the ...matrices/dao-scen subdir and all .scen files in there will be attempted. Will look for corresponding grids in dao-maps subdir")
    parser.add_argument('--grid_max_per_scen', default=21, type=int,
                        help="Max number of problems to run from any ONE .scen file. Eg if 21 and 156 .scen files in chosen subdir we will run 21 * 156 problems in total")
    parser.add_argument('--grid_reverse_scen_order', action='store_true',
                        help="If set, reverse the order of entries in each scen file before selection of --grid_max_per_scen entries (noting that higher cstar problems tend to occur later in file i.e. files are ordered by c* buckets of ~10 problems)")
    
    parser.add_argument('--grid_heur', nargs="*", default="octile", type=str, 
                        help="grid heuristics. Eg --grid_heur octile euclidean chebyshev manhattan")
    parser.add_argument('--grid_degs', nargs="*", default=0, type=int, 
                        help="grid heuristic degradation(s) to run. Eg 0 1 2 3")
    parser.add_argument('--grid_inadmiss', action='store_true', 
                        help="grid heuristic admissable or inadmissable Eg --grid_inadmiss means make inadmissable heuristic.")
    parser.add_argument('--grid_cost_multipier', default=1.0, type=float,
                        help="Any number > 1.0 will multiply the unit cost, hence weakening the heuristic which isn't multiplied by this number.")
    parser.add_argument("--grid_allow_diag", action='store_true',
                        help="Allow diagonal movement in the grid problems. Default is False. Note when enabled this sets the variable cost flag.")
    parser.add_argument('--grid_diag_cost', default=2.0, type=float,
                        help="Cost of diagonal move before cost multiplication. HOG2 grid Cstar calculations in .scen files use 2.0. Some papers use 1.5. Heuristic (and correct) estimate remains sqrt(2)=1.4142135623730951.")
    
    # Sliding Tiles params
    parser.add_argument("--tiles_dir", default='tile', type=str,
                        help="Tiles subdir off in_dir.")
    parser.add_argument("--tiles", default='', type=str,
                        help="File name of the sliding tile problems to run eg 15_puzzle_korf_std100.csv or '' to skip. Should be in the tiles subdir.")
    parser.add_argument('--tiles_max', default=100, type=int,
                        help="Max number of tile problems to run from the chosen tile file. Eg if 100 and 1000 problems in the file we will run 100 tile problems in total")
    parser.add_argument('--tiles_heur', nargs="*", default="manhattan", type=str, 
                        help="tiles heuristics. Only manhattan implemented. Eg --tiles_heur manhattan")
    parser.add_argument('--tiles_degs', nargs="*", default=0, type=int, 
                        help="tiles heuristic degradation(s) to run. Eg 0 1 2 3")
    parser.add_argument('--tiles_inadmiss', action='store_true', 
                        help="tiles heuristic admissable or inadmissable Eg --tiles_inadmiss means make inadmissable heuristic.")
    parser.add_argument("--tiles_var_cost", action='store_true',
                        help="When enabled this uses the tile value as the cost rather than 1. Default is false.")
    
    # Pancake params
    parser.add_argument("--pancakes_dir", default='pancake', type=str,
                        help="Pancakes subdir off in_dir.")
    parser.add_argument("--pancakes", default='', type=str,
                        help="File name of the pancake problems to run or '' to skip. Should be in the pancake subdir.")
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

    # Tower of Hanoi params
    parser.add_argument("--toh_dir", default='toh', type=str,
                        help="Toh subdir off in_dir.")
    parser.add_argument("--toh", default='', type=str,
                        help="File name of the towers of hanoi problems to run or '' to skip. Should be in the toh subdir.")
    parser.add_argument('--toh_max', default=50, type=int,
                        help="Max number of toh problems to run from the chosen toh file. Eg if 100 and 1000 problems in the file we will run 100 toh problems in total")
    parser.add_argument('--toh_heur', nargs="*", default="infinitepegrelaxation", type=str, 
                        help="toh heuristics. infinitepegrelaxation and 3pegstd implemented. Eg --toh_heur 3pegstd infinitepegrelaxation")
    parser.add_argument('--toh_degs', nargs="*", default=0, type=int, 
                        help="toh heuristic degradation(s) to run. Eg 0 1 2 3")
    parser.add_argument('--toh_inadmiss', action='store_true', 
                        help="toh heuristic admissable or inadmissable Eg --toh_inadmiss means make inadmissable heuristic.")
    
    # Algorithm params over all searches
    parser.add_argument('--algo_timeout', default=30.0, type=float,
                        help="Maximum time in minutes to allow any algorithm to run for. If exceeded, statistics to that point are returned along with status of 'timeout'. Normal return status = 'completed'. It is important to set this value such that no algorithm OOMs on the particular machine running the experiments.") 
    parser.add_argument('--algo_min_remaining_gb', default=2.0, type=float,
                        help="Minimum GB RAM remaining before algorithm is killed.") 
    parser.add_argument('--algo_visualise', action='store_true', 
                        help="Output .png files showing nodes expanded, path, meeting point etc for each algorithm and problem type that supports this.")

    # Heuristic search args
    parser.add_argument('--algo_heur', nargs="*", default="astar bd_astar", type=str, 
                        help="which unidirectional and bidirectional heuristic searches to run. Pass NONE to not run any: eg --algo_heur astar uniformcost bestfirst bdastar. Will set priority key to g+h, g and/or h appropriately")

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

    # Set up output directories if they don't exist
    os.makedirs(args.out_dir, exist_ok=True)
    args.visualise_dir = os.path.join(args.out_dir, 'visualise')
    os.makedirs(args.visualise_dir, exist_ok=True)
    if args.tiles:
        args.tiles_file_full = os.path.join(args.in_dir, args.tiles_dir, args.tiles)
        assert os.path.exists(args.tiles_file_full)
    if args.pancakes:
        args.pancakes_file_full = os.path.join(args.in_dir, args.pancakes_dir, args.pancakes)
        assert os.path.exists(args.pancakes_file_full)
    if args.toh:
        args.toh_file_full = os.path.join(args.in_dir, args.toh_dir, args.toh)
        assert os.path.exists(args.toh_file_full)
    if args.grid:
        grid_scen_dir = args.grid + "-scen"
        args.grid_dir_full = os.path.join(args.in_dir, args.grid_dir, grid_scen_dir)
        assert os.path.exists(args.grid_dir_full)


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

    random.seed(args.seed)

    # Set up the problems to be run ###################################
    tile_list, pancake_list, toh_list, grid_list = [], [], [], []
    if args.tiles:
        tile_list = problem_puzzle.create_tile_probs(args)
    if args.pancakes:
        pancake_list = problem_puzzle.create_pancake_probs(args)
    if args.toh:
        toh_list = problem_puzzle.create_toh_probs(args)
    if args.grid:
        grid_list = problem_spatial.create_grid_probs(args)

    problems = tile_list + pancake_list + toh_list + grid_list

    logger.info("######")
    logger.info("The following problems will be run:")
    for prob in problems:
        logger.info(str(prob))

    algorithms = []
    # Set up the heuristic algorithms to be run ########################
    # Heuristic algorithms must accept the parameters as shown here.. 
    if args.algo_heur[0] != "NONE":
        for algo in args.algo_heur:
            assert algo in SEARCH_MAP
            algo_class = SEARCH_MAP[algo]['class']
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

    # Set up the MCTS algorithms to be run ########################
    # MCTS algorithms must accept the parameters as shown here.. 
    if args.algo_mcts[0] != "NONE":
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
    logger.info("Running the following algorithms:")
    for a in algorithms:
        logger.info(str(a))

    # --- Run Experiments ---
    util.run_experiments(problems, algorithms, args.out_dir, args.out_prefix, 
                         seed=args.seed, timestamp=args.timestamp, logger=logger)

    logger.info(f"Finished search comparison at {time.strftime('%Y-%m-%d %H:%M:%CS')}")


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

