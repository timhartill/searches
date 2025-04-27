"""
Dijkstra/Uniform cost (g), Best first (h) ,A* f=g+h, Bidirectional A*, MCTS for Sliding Tile, Pancake, Towers of Hanoi
- This code implements various search algorithms for solving the Sliding Tile, Pancake Sorting, and Towers of Hanoi problems.

Partly generated from Gemini 2.5.
"""
import random
import time
import traceback # For error reporting
import json

import util

# problems
from puzzle import SlidingTileProblem, PancakeProblem, TowersOfHanoiProblem
from spatial import GridProblem

# searches
from search_mcts import heuristic_mcts_search
from search_unidirectional import generic_search
from search_bidirectional import bidirectional_a_star_search

# --- Main Execution Logic ---
if __name__ == "__main__":
    print(f"Running search comparison at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out_dir = '/media/tim/dl3storage/gitprojects/searches/outputs'  #output dir (visualisations in subdirs off matrices)
    out_file_base = f"{out_dir}/search_eval_{time.strftime('%Y%m%d_%H%M%S')}"

    random.seed(42)

    # Problem parameters - set here globally or individually in problem definition section
    make_heuristic_inadmissable = False # Set to True to make all heuristics for all problems inadmissible
    tile_degradation = 0
    pancake_degradation = 4
    hanoi_degradation = 0

    grid_prob = 'matrix_10yX10x.npy'   # w/o diagonal C*=22 allowing diag C* = 15.899
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


    grid_easy_unit = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_10yX10x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=False, heuristic='manhattan')

    grid_easy_unit_diag_octile = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_10yX10x.npy',
                    initial_state=None, goal_state=None, cost_multiplier=1,
                    make_heuristic_inadmissable=False, degradation=0,
                    allow_diagonal=True, heuristic='octile')

    grid_easy_unit_diag_octile_d5 = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_10yX10x.npy',
                    initial_state=None, goal_state=None, cost_multiplier=1,
                    make_heuristic_inadmissable=False, degradation=5,
                    allow_diagonal=True, heuristic='octile')


    grid_easy_unit_diag_manhattan = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_10yX10x.npy',
                    initial_state=None, goal_state=None, cost_multiplier=1,
                    make_heuristic_inadmissable=False, degradation=0,
                    allow_diagonal=True, heuristic='manhattan')
    
    grid_harder_unit = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=False, heuristic='manhattan')

    grid_harder_unit_d500 = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=500,
                   allow_diagonal=False, heuristic='manhattan')


    grid_harder_unit_diag_octile = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=True, heuristic='octile')

    grid_harder_unit_diag_octile_d5 = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=5,
                   allow_diagonal=True, heuristic='octile')


    grid_harder_unit_diag_euc = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=True, heuristic='euclidean')

    grid_harder_unit_diag_euc_d5 = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=5,
                   allow_diagonal=True, heuristic='euclidean')

    grid_harder_unit_diag_euc_d500 = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=500,
                   allow_diagonal=True, heuristic='euclidean')


    grid_harder_unit_diag_che = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=True, heuristic='chebyshev')

    grid_harder_unit_diag_che_d5 = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=5,
                   allow_diagonal=True, heuristic='chebyshev')


    grid_harder_unit_diag_octile_d0 = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=True, degradation=0,
                   allow_diagonal=True, heuristic='octile')

    grid_harder_unit_diag_euc_d0_cm1000 = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1000,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=True, heuristic='euclidean')

    grid_harder_unit_diag_euc_d500_cm1000 = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_20yX100x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1000,
                   make_heuristic_inadmissable=False, degradation=500,
                   allow_diagonal=True, heuristic='euclidean')


    grid_harder1000x1000_unit_diag_mh_d0_cm1 = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_1000yX1000x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=False, heuristic='manhattan')

    grid_harder1000x1000_unit_diag_mh_d5_cm1 = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_1000yX1000x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=5,
                   allow_diagonal=False, heuristic='manhattan')


    grid_harder1000x1000_unit_diag_euc_d0_cm1 = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_1000yX1000x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=True, heuristic='euclidean')

    grid_harder1000x1000_unit_diag_euc_d5_cm1 = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_1000yX1000x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=5,
                   allow_diagonal=True, heuristic='euclidean')

    grid_harder1000x1000_unit_nodiag_euc_d0_cm1 = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_1000yX1000x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=0,
                   allow_diagonal=False, heuristic='euclidean')

    grid_harder1000x1000_unit_nodiag_euc_d5_cm1 = GridProblem('/media/tim/dl3storage/gitprojects/searches/problems/matrices/matrix_1000yX1000x.npy',
                   initial_state=None, goal_state=None, cost_multiplier=1,
                   make_heuristic_inadmissable=False, degradation=5,
                   allow_diagonal=False, heuristic='euclidean')


    problems_to_solve = [
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
    ]

    # --- Define Algorithms ---
    def run_ucs(problem): return generic_search(problem, priority_key='g')
    def run_greedy_bfs(problem): return generic_search(problem, priority_key='h')
    def run_astar(problem): return generic_search(problem, priority_key='f')
    def run_bidir_astar(problem): return bidirectional_a_star_search(problem)
    def run_mcts_standard(problem): 
        # Wrapper for standard MCTS (no heuristic guidance)
        return heuristic_mcts_search(problem, iterations=iterations, max_depth=max_depth, heuristic_weight=0.0, heuristic_rollout=False)
    def run_mcts_h_select(problem): 
        # MCTS with heuristic in selection only
        return heuristic_mcts_search(problem, iterations=iterations, max_depth=max_depth, heuristic_weight=heuristic_weight, heuristic_rollout=False) # Tune weight
    def run_mcts_h_rollout(problem): 
        # MCTS with heuristic in rollout only
        return heuristic_mcts_search(problem, iterations=iterations, max_depth=max_depth, heuristic_weight=0.0, heuristic_rollout=True)
    def run_mcts_h_both(problem): 
        # MCTS with heuristic in both selection and rollout
        return heuristic_mcts_search(problem, iterations=iterations, max_depth=max_depth, heuristic_weight=heuristic_weight, heuristic_rollout=True) # Tune weight


    search_algorithms_runners = {
        "Uniform Cost": run_ucs,
        "Greedy Best-First": run_greedy_bfs,
        "AStar": run_astar,
        "Bidirectional AStar": run_bidir_astar,
        "MCTS (Standard)": run_mcts_standard,
    #    "MCTS (H-Select)": run_mcts_h_select, # Add heuristic versions
    #    "MCTS (H-Rollout)": run_mcts_h_rollout,
    #    "MCTS (H-Both)": run_mcts_h_both,
    }


    # --- Run Experiments ---
    all_results = []
    for problem in problems_to_solve:  # For each problem
        print(f"\n{'=' * 20}\nSolving: {problem}\nInitial State: {problem.initial_state()}\nGoal State:    {problem.goal_state()}\nInitial Heuristic: {problem.heuristic(problem.initial_state())}\n{'-' * 20}")
        problem_results = []
        
        for algo_display_name, algo_func in search_algorithms_runners.items():  # For each algorithm
            print(f"Running {algo_display_name}...")
            result = None
            try:
                result = algo_func(problem) # Call the runner
                
                # Set algorithm name in result consistently
                #if result and 'algorithm' in result: 
                result['algorithm'] = algo_display_name
                
                print(f"{algo_display_name} Done. Time: {result.get('time', -1):.4f}secs Nodes Expanded: {result.get('nodes_expanded', -1)} Path Cost: {result.get('cost', 'N/A')} Length: {len(result['path']) if result['path'] else 'No Path Found'}")

            except Exception as e:
                print(f"!!! ERROR during {algo_display_name} on {problem}: {e}")
                traceback.print_exc() 
                result = { "path": None, "cost": -1, "nodes_expanded": -1, "time": -1, 
                           "algorithm": algo_display_name, "error": str(e)}

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
             # print(f"Path Length: {len(res['path'])}") # Should be sum(unit cost) + 1
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
    util.write_jsonl_to_csv(all_results, csv_file_path, del_keys=['path'])
