"""
Dijkstra/Uniform cost (g), Best first (h) ,A* f=g+h, Bidirectional A*, MCTS for Sliding Tile, Pancake, Towers of Hanoi
- This code implements various search algorithms for solving the Sliding Tile, Pancake Sorting, Pathfinder and Towers of Hanoi problems.

Some code generated from Gemini 2.5.
"""
import os
import random
import time
import json
import argparse

import util

# problems
from problem_puzzle import SlidingTileProblem, PancakeProblem, TowersOfHanoiProblem
from problem_spatial import GridProblem

# searches
from search_mcts import heuristic_mcts_search
from search_unidirectional import generic_search
from search_bidirectional import bidirectional_a_star_search

# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default='/media/tim/dl3storage/gitprojects/searches/outputs', type=str,
                        help="Full path to output directory. CSV and JSON output files will be written here.")  #TJH , required=True)
    parser.add_argument("--out_prefix", default='search_eval', type=str,
                        help="CSV and JSON output file prefix. Date and time will be added to make unique.")  #TJH , required=True)
    parser.add_argument('--seed', default=42, type=int,
                        help="random seed") 
    #parser.add_argument("--do_train", action='store_true')  # example boolean
    #parser.add_argument("--learning_rate", default=1e-5, type=float)  # example float
    args = parser.parse_args()

    print(f"Running search comparison at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(args)

    random.seed(args.seed)

    # Problem parameters - set here globally or individually in problem definition section
    make_heuristic_inadmissable = False # Set to True to make all heuristics for all problems inadmissible
    tile_degradation = 0
    pancake_degradation = 4
    hanoi_degradation = 0

    #matrix_10yX10x.npy   # w/o diagonal C*=22 allowing diag C* = 15.899
    #matrix_20yX100x.npy w/o diag C*=176 with diag C*= 152.58
    #matrix_1000yX1000x.npy w/o diag C*= 4330 with diag C* = 3881.87

    # Search Parameters - set here globally or individually in algorithm definition section
    # MCTS
    iterations = 100            # MCTS  1000000 finds near-optimal paths in 8-puzzle and occasionally pancake
    max_depth = 150             # MCTS
    heuristic_weight = 100.0    # MCTS



    # --- Define Problems ---
    #tile_initial = [1, 2, 3, 0, 4, 6, 7, 5, 8] # Medium unit C*=3
    tile_initial = [8, 6, 7, 2, 5, 4, 3, 0, 1] # harder 3x3 unit C*=31
    #tile_initial = [15, 11, 12, 14, 9, 13, 10, 8, 6, 7, 2, 5, 4, 3, 0, 1] # harderer 4x4 unit C*= >>31 A* ran out of memory @ 48GB
    sliding_tile_unit_cost = SlidingTileProblem(initial_state=tile_initial, 
                                                use_variable_costs=False, 
                                                make_heuristic_inadmissable=make_heuristic_inadmissable,
                                                degradation=tile_degradation)
    sliding_tile_var_cost = SlidingTileProblem(initial_state=tile_initial, 
                                               use_variable_costs=True,
                                               make_heuristic_inadmissable=make_heuristic_inadmissable,
                                               degradation=tile_degradation)

    tile_initial = [8, 6, 7, 2, 5, 4, 3, 0, 1, 10, 11, 9] # harder 4x3 unit C*=32
    sliding_tile_unit_cost43 = SlidingTileProblem(initial_state=tile_initial, 
                                                use_variable_costs=False, 
                                                make_heuristic_inadmissable=False,
                                                degradation=tile_degradation)
    sliding_tile_var_cost43 = SlidingTileProblem(initial_state=tile_initial, 
                                               use_variable_costs=True,
                                               make_heuristic_inadmissable=False,
                                               degradation=tile_degradation)
    sliding_tile_unit_cost43_inadmiss = SlidingTileProblem(initial_state=tile_initial, 
                                               use_variable_costs=False,
                                               make_heuristic_inadmissable=True,
                                               degradation=tile_degradation)
    sliding_tile_unit_cost43_d5 = SlidingTileProblem(initial_state=tile_initial, 
                                                use_variable_costs=False, 
                                                make_heuristic_inadmissable=False,
                                                degradation=5)



    pancake_initial = (8, 3, 5, 1, 6, 4, 2, 7) # C*=8
    pancake_unit_cost = PancakeProblem(initial_state=pancake_initial, 
                                       use_variable_costs=False,
                                       make_heuristic_inadmissable=make_heuristic_inadmissable,
                                       degradation=pancake_degradation)

    pancake_var_cost = PancakeProblem(initial_state=pancake_initial, 
                                      use_variable_costs=True,
                                      make_heuristic_inadmissable=make_heuristic_inadmissable,
                                      degradation=pancake_degradation)

    hanoi_disks = 7 # Optimal cost = 2^7 - 1 = 127
    hanoi_problem = TowersOfHanoiProblem(num_disks=hanoi_disks, initial_peg='A', target_peg='C',
                                         make_heuristic_inadmissable=False, heuristic='3PegStd',
                                         degradation=hanoi_degradation)
    hanoi_problem_3_indmiss = TowersOfHanoiProblem(num_disks=hanoi_disks, initial_peg='A', target_peg='C',
                                         make_heuristic_inadmissable=True, heuristic='3PegStd',
                                         degradation=hanoi_degradation)
    hanoi_problem_3_d5 = TowersOfHanoiProblem(num_disks=hanoi_disks, initial_peg='A', target_peg='C',
                                         make_heuristic_inadmissable=False, heuristic='3PegStd',
                                         degradation=5)
    hanoi_problem_3_infpeg = TowersOfHanoiProblem(num_disks=hanoi_disks, initial_peg='A', target_peg='C',
                                         make_heuristic_inadmissable=False, heuristic='InfinitePegRelaxation',
                                         degradation=hanoi_degradation)
    hanoi_problem_3_infpeg_indmiss = TowersOfHanoiProblem(num_disks=hanoi_disks, initial_peg='A', target_peg='C',
                                         make_heuristic_inadmissable=True, heuristic='InfinitePegRelaxation',
                                         degradation=hanoi_degradation)
    hanoi_problem_3_infpeg_d5 = TowersOfHanoiProblem(num_disks=hanoi_disks, initial_peg='A', target_peg='C',
                                         make_heuristic_inadmissable=False, heuristic='InfinitePegRelaxation',
                                         degradation=5)



    hanoi_problem_4Tower_3peg = TowersOfHanoiProblem(num_disks=hanoi_disks, initial_peg='A', target_peg='D', pegs=['A', 'B', 'C', 'D'],
                                         make_heuristic_inadmissable=False,  heuristic='3PegStd',
                                         degradation=hanoi_degradation)

    hanoi_problem_4Tower_InfPeg = TowersOfHanoiProblem(num_disks=hanoi_disks, initial_peg='A', target_peg='D', pegs=['A', 'B', 'C', 'D'],
                                         make_heuristic_inadmissable=False,  heuristic='InfinitePegRelaxation',
                                         degradation=hanoi_degradation)

    hanoi_problem_4Tower_InfPeg_state2 = TowersOfHanoiProblem(num_disks=hanoi_disks, initial_peg='A', target_peg='D', pegs=['A', 'B', 'C', 'D'],
                                         initial_state=['B', 'B', 'C', 'C', 'D', 'A', 'A'],
                                         make_heuristic_inadmissable=False,  heuristic='InfinitePegRelaxation',
                                         degradation=hanoi_degradation)

    hanoi_problem_4Tower_InfPeg_state3 = TowersOfHanoiProblem(num_disks=hanoi_disks, initial_peg='A', target_peg='D', pegs=['A', 'B', 'C', 'D'],
                                         initial_state=['D', 'D', 'D', 'D', 'D', 'D', 'A'],
                                         make_heuristic_inadmissable=False,  heuristic='InfinitePegRelaxation',
                                         degradation=hanoi_degradation)


    hanoi_problem_4Tower_3peg_inadmiss = TowersOfHanoiProblem(num_disks=hanoi_disks, initial_peg='A', target_peg='D', pegs=['A', 'B', 'C', 'D'],
                                         make_heuristic_inadmissable=True, heuristic='3PegStd',
                                         degradation=hanoi_degradation)

    hanoi_problem_4Tower_InfPeg_inadmiss = TowersOfHanoiProblem(num_disks=hanoi_disks, initial_peg='A', target_peg='D', pegs=['A', 'B', 'C', 'D'],
                                         make_heuristic_inadmissable=True, heuristic='InfinitePegRelaxation',
                                         degradation=hanoi_degradation)
    hanoi_problem_4Tower_InfPeg_d5 = TowersOfHanoiProblem(num_disks=hanoi_disks, initial_peg='A', target_peg='D', pegs=['A', 'B', 'C', 'D'],
                                         make_heuristic_inadmissable=False, heuristic='InfinitePegRelaxation',
                                         degradation=5)
    hanoi_problem_4Tower_InfPeg_12disk = TowersOfHanoiProblem(num_disks=12, initial_peg='A', target_peg='D', pegs=['A', 'B', 'C', 'D'],
                                         make_heuristic_inadmissable=False, heuristic='InfinitePegRelaxation',
                                         degradation=0)


    matrices_base = '/media/tim/dl3storage/gitprojects/searches/problems/matrices/'
    numpy_probs = os.path.join(matrices_base, 'numpy_probs')

    grid_easy_unit = GridProblem(f'{numpy_probs}/matrix_10yX10x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=False, heuristic='manhattan')

    grid_easy_unit_diag_octile = GridProblem(f'{numpy_probs}/matrix_10yX10x.npy',
                    initial_state=None, goal_state=None, cost_multiplier=1,
                    make_heuristic_inadmissable=False, degradation=0,
                    allow_diagonal=True, heuristic='octile')

    grid_easy_unit_diag_octile_d5 = GridProblem(f'{numpy_probs}/matrix_10yX10x.npy',
                    initial_state=None, goal_state=None, cost_multiplier=1,
                    make_heuristic_inadmissable=False, degradation=5,
                    allow_diagonal=True, heuristic='octile')


    grid_easy_unit_diag_manhattan = GridProblem(f'{numpy_probs}/matrix_10yX10x.npy',
                    initial_state=None, goal_state=None, cost_multiplier=1,
                    make_heuristic_inadmissable=False, degradation=0,
                    allow_diagonal=True, heuristic='manhattan')
    
    grid_harder_unit = GridProblem(f'{numpy_probs}/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=False, heuristic='manhattan')

    grid_harder_unit_d500 = GridProblem(f'{numpy_probs}/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=500,
                   allow_diagonal=False, heuristic='manhattan')


    grid_harder_unit_diag_octile = GridProblem(f'{numpy_probs}/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=True, heuristic='octile')

    grid_harder_unit_diag_octile_d5 = GridProblem(f'{numpy_probs}/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=5,
                   allow_diagonal=True, heuristic='octile')


    grid_harder_unit_diag_euc = GridProblem(f'{numpy_probs}/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=True, heuristic='euclidean')

    grid_harder_unit_diag_euc_d5 = GridProblem(f'{numpy_probs}/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=5,
                   allow_diagonal=True, heuristic='euclidean')

    grid_harder_unit_diag_euc_d500 = GridProblem(f'{numpy_probs}/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=500,
                   allow_diagonal=True, heuristic='euclidean')


    grid_harder_unit_diag_che = GridProblem(f'{numpy_probs}/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=True, heuristic='chebyshev')

    grid_harder_unit_diag_che_d5 = GridProblem(f'{numpy_probs}/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=5,
                   allow_diagonal=True, heuristic='chebyshev')


    grid_harder_unit_diag_octile_d0 = GridProblem(f'{numpy_probs}/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=True, degradation=0,
                   allow_diagonal=True, heuristic='octile')

    grid_harder_unit_diag_euc_d0_cm1000 = GridProblem(f'{numpy_probs}/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1000,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=True, heuristic='euclidean')

    grid_harder_unit_diag_euc_d500_cm1000 = GridProblem(f'{numpy_probs}/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1000,
                   make_heuristic_inadmissable=False, degradation=500,
                   allow_diagonal=True, heuristic='euclidean')


    grid_harder1000x1000_unit_diag_mh_d0_cm1 = GridProblem(f'{numpy_probs}/matrix_1000yX1000x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=False, heuristic='manhattan')

    grid_harder1000x1000_unit_diag_mh_d5_cm1 = GridProblem(f'{numpy_probs}/matrix_1000yX1000x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=5,
                   allow_diagonal=False, heuristic='manhattan')


    grid_harder1000x1000_unit_diag_euc_d0_cm1 = GridProblem(f'{numpy_probs}/matrix_1000yX1000x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=True, heuristic='euclidean')

    grid_harder1000x1000_unit_diag_euc_d5_cm1 = GridProblem(f'{numpy_probs}/matrix_1000yX1000x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=5,
                   allow_diagonal=True, heuristic='euclidean')

    grid_harder1000x1000_unit_nodiag_euc_d0_cm1 = GridProblem(f'{numpy_probs}/matrix_1000yX1000x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=False, heuristic='euclidean')

    grid_harder1000x1000_unit_nodiag_euc_d5_cm1 = GridProblem(f'{numpy_probs}/matrix_1000yX1000x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=5,
                   allow_diagonal=False, heuristic='euclidean')


    # Load grid problems from .map files
    grid_scenarios = util.load_scen_file('/media/tim/dl3storage/gitprojects/searches/problems/matrices/dao-scen/brc505d.map.scen')
    
    grid_dao1 = GridProblem(grid_scenarios[2]['map_dir'], 
                            initial_state=[grid_scenarios[2]['start_y'], grid_scenarios[2]['start_x']], 
                            goal_state=[grid_scenarios[2]['goal_y'], grid_scenarios[2]['goal_x']], 
                            cost_multiplier=1,
                            make_heuristic_inadmissable=False, degradation=0,
                            allow_diagonal=True, 
                            heuristic='octile')

    grid_dao2 = GridProblem(grid_scenarios[1509]['map_dir'], 
                            initial_state=[grid_scenarios[1509]['start_y'], grid_scenarios[1509]['start_x']], 
                            goal_state=[grid_scenarios[1509]['goal_y'], grid_scenarios[1509]['goal_x']], 
                            cost_multiplier=1,
                            make_heuristic_inadmissable=False, degradation=0,
                            allow_diagonal=True, 
                            heuristic='octile')

    grid_scenarios = util.load_scen_file('/media/tim/dl3storage/gitprojects/searches/problems/matrices/maze-scen/maze512-32-9.map.scen')
    
    grid_dao3 = GridProblem(grid_scenarios[-2]['map_dir'], 
                            initial_state=[grid_scenarios[-2]['start_y'], grid_scenarios[-2]['start_x']], 
                            goal_state=[grid_scenarios[-2]['goal_y'], grid_scenarios[-2]['goal_x']], 
                            cost_multiplier=1,
                            make_heuristic_inadmissable=False, degradation=0,
                            allow_diagonal=True, 
                            heuristic='octile')

    grid_dao4 = GridProblem(grid_scenarios[-1]['map_dir'], 
                            initial_state=[grid_scenarios[-1]['start_y'], grid_scenarios[-1]['start_x']], 
                            goal_state=[grid_scenarios[-1]['goal_y'], grid_scenarios[-1]['goal_x']], 
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
                            heuristic='octile')



    problems = [
        sliding_tile_unit_cost,
#        sliding_tile_var_cost,
#        sliding_tile_unit_cost43,
#        sliding_tile_var_cost43,
#        sliding_tile_unit_cost43_inadmiss,
#        sliding_tile_unit_cost43_d5,
        pancake_unit_cost,
#        pancake_var_cost,
#        hanoi_problem,
#        hanoi_problem_3_indmiss,
#        hanoi_problem_3_d5,
#        hanoi_problem_3_infpeg,
#        hanoi_problem_3_infpeg_indmiss,
#        hanoi_problem_3_infpeg_d5,
#        hanoi_problem_4Tower_3peg,
         hanoi_problem_4Tower_InfPeg,
#         hanoi_problem_4Tower_InfPeg_state2,
#         hanoi_problem_4Tower_InfPeg_state3,
#        hanoi_problem_4Tower_3peg_inadmiss,
#        hanoi_problem_4Tower_InfPeg_inadmiss,
#        hanoi_problem_4Tower_InfPeg_d5,
#        hanoi_problem_4Tower_InfPeg_12disk,  # uniform costs takes 4mins, generates 16M expansions, only ~11GB ram
#        grid_easy_unit,
#        grid_easy_unit_diag_octile,
#        grid_easy_unit_diag_octile_d5,
#        grid_easy_unit_diag_manhattan,
#        grid_harder_unit,
#        grid_harder_unit_d500,
        grid_harder_unit_diag_octile,
#        grid_harder_unit_diag_octile_d5,
#        grid_harder_unit_diag_euc,
#        grid_harder_unit_diag_euc_d5,
#        grid_harder_unit_diag_euc_d500,
#        grid_harder_unit_diag_che,
#        grid_harder_unit_diag_che_d5,
#        grid_harder_unit_diag_octile_d0,
#        grid_harder_unit_diag_euc_d0_cm1000,
#        grid_harder_unit_diag_euc_d500_cm1000,
#        grid_harder1000x1000_unit_diag_mh_d0_cm1,
#        grid_harder1000x1000_unit_diag_mh_d5_cm1,
#        grid_harder1000x1000_unit_diag_euc_d0_cm1,
#        grid_harder1000x1000_unit_diag_euc_d5_cm1,
#        grid_harder1000x1000_unit_nodiag_euc_d0_cm1,
#        grid_harder1000x1000_unit_nodiag_euc_d5_cm1,
        grid_dao1,
        grid_dao2,
        grid_dao3,
        grid_dao4,
        grid_dao5,
    ]

    # --- Define Algorithms ie give algorithm setups with differing params, unique fn names ---
    run_ucs = generic_search(priority_key='g', tiebreaker1='g', tiebreaker2='NONE', visualise=True)
    run_hucs = generic_search(priority_key='g', tiebreaker1='f', tiebreaker2='NONE', visualise=True)
    run_greedy_bfs = generic_search(priority_key='h', tiebreaker1='g', tiebreaker2='NONE', visualise=True)
    run_astar = generic_search(priority_key='f', tiebreaker1='g', tiebreaker2='NONE', visualise=True)
    run_astar1 = generic_search(priority_key='f', tiebreaker1='-g', tiebreaker2='NONE', visualise=True)
    run_astar2 = generic_search(priority_key='f', tiebreaker1='FIFO', tiebreaker2='NONE', visualise=True)
    run_bidir_astar = bidirectional_a_star_search(tiebreaker1='-g', tiebreaker2='NONE', visualise=True)
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
        run_ucs,
        run_greedy_bfs,
        run_astar1,
        run_bidir_astar,
        run_mcts_standard,
#        "MCTS (H-Select)": run_mcts_h_select, # Add heuristic versions
#        "MCTS (H-Rollout)": run_mcts_h_rollout,
#        "MCTS (H-Both)": run_mcts_h_both,
    ]

    # --- Run Experiments ---
    util.run_experiments(problems, algorithms, args.out_dir, args.out_prefix, seed=args.seed)

    print(f"Finished search comparison at {time.strftime('%Y-%m-%d %H:%M:%S')}")
