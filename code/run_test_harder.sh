#!/bin/bash
# Test script for search runner options

<<comment
search_runner.py usage:

usage:  [-h] [--out_dir OUT_DIR] [--out_prefix OUT_PREFIX] [--in_dir IN_DIR]
        [--seed SEED] [--grid_dir GRID_DIR] [--grid [GRID ...]]
        [--grid_max_per_scen GRID_MAX_PER_SCEN] [--grid_reverse_scen_order]
        [--grid_heur [GRID_HEUR ...]] [--grid_degs [GRID_DEGS ...]]
        [--grid_inadmiss] [--grid_cost_multipier GRID_COST_MULTIPIER]
        [--grid_allow_diag] [--grid_diag_cost GRID_DIAG_COST]
        [--grid_ignore_cstar] [--tiles_dir TILES_DIR] [--tiles TILES]
        [--tiles_max TILES_MAX] [--tiles_heur [TILES_HEUR ...]]
        [--tiles_degs [TILES_DEGS ...]] [--tiles_inadmiss] [--tiles_var_cost]
        [--tiles_ignore_cstar] [--pancakes_dir PANCAKES_DIR]
        [--pancakes PANCAKES] [--pancakes_max PANCAKES_MAX]
        [--pancakes_heur [PANCAKES_HEUR ...]]
        [--pancakes_degs [PANCAKES_DEGS ...]] [--pancakes_inadmiss]
        [--pancakes_var_cost] [--pancakes_ignore_cstar] [--toh_dir TOH_DIR]
        [--toh TOH] [--toh_max TOH_MAX] [--toh_heur [TOH_HEUR ...]]
        [--toh_degs [TOH_DEGS ...]] [--toh_inadmiss] [--toh_ignore_cstar]
        [--algo_timeout ALGO_TIMEOUT]
        [--algo_min_remaining_gb ALGO_MIN_REMAINING_GB] [--algo_visualise]
        [--algo_heur [ALGO_HEUR ...]] [--algo_mcts [ALGO_MCTS ...]]
        [--algo_mcts_iterations ALGO_MCTS_ITERATIONS]
        [--algo_mcts_max_depth ALGO_MCTS_MAX_DEPTH]
        [--algo_mcts_exploration_weight ALGO_MCTS_EXPLORATION_WEIGHT]
        [--algo_mcts_heur_weight ALGO_MCTS_HEUR_WEIGHT]

Search Algorithm Comparison Runner

options:
  -h, --help            show this help message and exit
  --out_dir OUT_DIR     Full path to output directory. CSV and JSON output
                        files will be written here.
  --out_prefix OUT_PREFIX
                        Log, CSV and JSON output file prefix. Date and time
                        will be added to make unique.
  --in_dir IN_DIR       Full path to input directory BASE. Expected subdirs
                        off here are matrices, pancake, tile and toh. matrices
                        should have np-scen and np-map and eg dao-map and dao-
                        scen etc off it.
  --seed SEED           random seed. Reset before running each algorithm on
                        each problem.
  --grid_dir GRID_DIR   Grid subdir off in_dir.
  --grid [GRID ...]     Space-separated list of the domain portion of the grid
                        problems to run eg --grid dao mazes. 'NONE' means
                        don't run. for each domain, this will be expanded to
                        the ...matrices/dao-scen subdir and all .scen files in
                        there will be attempted. Will look for corresponding
                        grids in dao-maps subdir
  --grid_max_per_scen GRID_MAX_PER_SCEN
                        Max number of problems to run from any ONE .scen file.
                        Eg if 21 and 156 .scen files in chosen subdir we will
                        run 21 * 156 problems in total
  --grid_reverse_scen_order
                        If set, reverse the order of entries in each scen file
                        before selection of --grid_max_per_scen entries
                        (noting that higher cstar problems tend to occur later
                        in file i.e. files are ordered by c* buckets of ~10
                        problems)
  --grid_heur [GRID_HEUR ...]
                        grid heuristics. Eg --grid_heur octile euclidean
                        chebyshev manhattan
  --grid_degs [GRID_DEGS ...]
                        grid heuristic degradation(s) to run. Eg 0 1 2 3
  --grid_inadmiss       grid heuristic admissable or inadmissable Eg
                        --grid_inadmiss means make inadmissable heuristic.
  --grid_cost_multipier GRID_COST_MULTIPIER
                        Any number > 1.0 will multiply the unit cost, hence
                        weakening the heuristic which isn't multiplied by this
                        number.
  --grid_allow_diag     Allow diagonal movement in the grid problems. Default
                        is False. Note when enabled this sets the variable
                        cost flag.
  --grid_diag_cost GRID_DIAG_COST
                        Cost of diagonal move before cost multiplication. Some
                        HOG2 grid Cstar calculations in .scen files use 2.0,
                        other .scen diag costs vary. Some papers use 1.5.
                        Heuristic (and correct) estimate remains
                        sqrt(2)=1.4142135623730951.
  --grid_ignore_cstar   If set the cstar in .scen files is not used. This is
                        typically set when using a different diagonal cost
                        than what the cstar in the .scen file was based on, or
                        you are using a cost multiplier other than 1.
  --tiles_dir TILES_DIR
                        Tiles subdir off in_dir.
  --tiles TILES         File name of the sliding tile problems to run eg
                        15_puzzle_korf_std100.csv or 'NONE' to skip. Should be
                        in the tiles subdir.
  --tiles_max TILES_MAX
                        Max number of tile problems to run from the chosen
                        tile file. Eg if 100 and 1000 problems in the file we
                        will run 100 tile problems in total
  --tiles_heur [TILES_HEUR ...]
                        tiles heuristics. Only manhattan implemented. Eg
                        --tiles_heur manhattan
  --tiles_degs [TILES_DEGS ...]
                        tiles heuristic degradation(s) to run. Eg 0 1 2 3
  --tiles_inadmiss      tiles heuristic admissable or inadmissable Eg
                        --tiles_inadmiss means make inadmissable heuristic.
  --tiles_var_cost      When enabled this uses the tile value as the cost
                        rather than 1. Default is false.
  --tiles_ignore_cstar  If set the cstar in .csv files is not used. This is
                        typically set when using variable costs or an
                        inadmissable heuristic.
  --pancakes_dir PANCAKES_DIR
                        Pancakes subdir off in_dir.
  --pancakes PANCAKES   File name of the pancake problems to run or 'NONE' to
                        skip. Should be in the pancake subdir.
  --pancakes_max PANCAKES_MAX
                        Max number of pancake problems to run from the chosen
                        pancake file. Eg if 100 and 1000 problems in the file
                        we will run 100 pancake problems in total
  --pancakes_heur [PANCAKES_HEUR ...]
                        pancakes heuristics. Only symmetric gap implemented.
                        Eg --pancakes_heur symgap
  --pancakes_degs [PANCAKES_DEGS ...]
                        pancakes heuristic degradation(s) to run. Eg 0 1 2 3
  --pancakes_inadmiss   pancakes heuristic admissable or inadmissable Eg
                        --pancakes_inadmiss means make inadmissable heuristic.
  --pancakes_var_cost   When enabled this uses the num pancakes flipped as the
                        cost rather than 1. Default is false.
  --pancakes_ignore_cstar
                        If set the cstar in .csv files is not used. This is
                        typically set when using variable costs or an
                        inadmissable heuristic.
  --toh_dir TOH_DIR     Toh subdir off in_dir.
  --toh TOH             File name of the towers of hanoi problems to run or
                        'NONE' to skip. Should be in the toh subdir.
  --toh_max TOH_MAX     Max number of toh problems to run from the chosen toh
                        file. Eg if 100 and 1000 problems in the file we will
                        run 100 toh problems in total
  --toh_heur [TOH_HEUR ...]
                        toh heuristics. infinitepegrelaxation and 3pegstd
                        implemented. Eg --toh_heur 3pegstd
                        infinitepegrelaxation
  --toh_degs [TOH_DEGS ...]
                        toh heuristic degradation(s) to run. Eg 0 1 2 3
  --toh_inadmiss        toh heuristic admissable or inadmissable Eg
                        --toh_inadmiss means make inadmissable heuristic.
  --toh_ignore_cstar    If set the cstar in .csv files is not used. This is
                        typically set when using variable costs or an
                        inadmissable heuristic.
  --algo_timeout ALGO_TIMEOUT
                        Maximum time in minutes to allow any algorithm to run
                        for. If exceeded, statistics to that point are
                        returned along with status of 'timeout'. Normal return
                        status = 'completed'. It is important to set this
                        value such that no algorithm OOMs on the particular
                        machine running the experiments.
  --algo_min_remaining_gb ALGO_MIN_REMAINING_GB
                        Minimum GB RAM remaining before algorithm is killed.
  --algo_visualise      Output .png files showing nodes expanded, path,
                        meeting point etc for each algorithm and problem type
                        that supports this.
  --algo_heur [ALGO_HEUR ...]
                        which unidirectional and bidirectional heuristic
                        searches to run. Pass NONE to not run any: eg
                        --algo_heur astar uniformcost bestfirst bdastar. Will
                        set priority key to g+h, g and/or h appropriately
  --algo_mcts [ALGO_MCTS ...]
                        which MCTS searches to run. Pass NONE to not run any:
                        eg --algo_mcts NONE
  --algo_mcts_iterations ALGO_MCTS_ITERATIONS
                        Number of MCTS iterations to be run for each MCTS
                        algorithm.
  --algo_mcts_max_depth ALGO_MCTS_MAX_DEPTH
                        Maximum depth of MCTS Tree.
  --algo_mcts_exploration_weight ALGO_MCTS_EXPLORATION_WEIGHT
                        MCTS Exploration Weight
  --algo_mcts_heur_weight ALGO_MCTS_HEUR_WEIGHT
                        MCTS Heuristic Weight (if applicable)
comment

#daotest mazetest
#dao maze
#"8_puzzle_probs2_easytest.csv"
#"11_puzzle_probs1_easytest.csv"
#11_puzzle_probs10_seed42_2025-05-20_17-08-21.csv
#"15_puzzle_probs1_testcstar66.csv"
#"15_puzzle_probs100_korf_std.csv"
#"8_pancake_probs1_easytest.csv"
#"14_pancake_probs2_test.csv"
#"14_pancake_probs100_seed42_2025-05-20_17-21-23.csv"
#"7_toh_3_peg_probs1_easytest.csv"
#"7_toh_4_peg_probs2_easytest.csv"
#"12_toh_4_peg_probs2_test.csv"
#"12_toh_4_peg_probs100_seed42_2025-05-20_17-21-23.csv"

#--algo_heur astar uc huc bfs bd_astar bd_uc bd_huc bd_bfs \
#--algo_mcts mcts_noheur mcts_selectheur mcts_rolloutheur mcts_bothheur \

#        --tiles_ignore_cstar \
#        --tiles_var_cost \
#        --pancakes_ignore_cstar \
#        --pancakes_var_cost \
#        --toh_inadmiss \


python search_runner.py \
        --out_dir "/media/tim/dl3storage/gitprojects/searches/outputs" \
        --out_prefix "search-eval" \
        --in_dir "/media/tim/dl3storage/gitprojects/searches/problems" \
        --seed 42 \
        --grid mazetest \
        --grid_max_per_scen 1 \
        --grid_reverse_scen_order \
        --grid_heur octile \
        --grid_degs 0 3 \
        --grid_cost_multipier 1.0 \
        --grid_allow_diag \
        --grid_diag_cost 1.5 \
        --grid_ignore_cstar \
        --tiles "15_puzzle_probs1_testcstar66.csv" \
        --tiles_max 100 \
        --tiles_heur manhattan \
        --tiles_degs 0 2 \
        --pancakes "14_pancake_probs2_test.csv" \
        --pancakes_max 100 \
        --pancakes_heur symgap \
        --pancakes_degs 0 2 \
        --toh "12_toh_4_peg_probs2_test.csv" \
        --toh_max 100 \
        --toh_heur infinitepegrelaxation pdb_4_10+2 \
        --toh_degs 0 2 \
        --algo_visualise \
        --algo_timeout 120 \
        --algo_min_remaining_gb 5.0 \
        --algo_heur astar bd_astar \
        --algo_mcts NONE


