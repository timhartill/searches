#!/bin/bash

<<comment
Create 100 random 14-pancake and 12-disk 4-peg toh problems respectively 
all with C* added using A*
comment


python search_runner.py \
        --out_dir "/media/tim/dl3storage/gitprojects/searches/outputs" \
        --out_prefix "create-puzzles" \
        --in_dir "/media/tim/dl3storage/gitprojects/searches/problems" \
        --seed 42 \
        --grid NONE \
        --tiles NONE \
        --tiles_heur manhattan \
        --tiles_make_random 0 \
        --tiles_make_size 12 \
        --tiles_add_cstar \
        --pancakes NONE \
        --pancakes_heur symgap \
        --pancakes_make_random 100 \
        --pancakes_make_size 14 \
        --pancakes_add_cstar \
        --toh NONE \
        --toh_heur pdb_4_10+2 \
        --toh_make_random 100 \
        --toh_num_disks 12 \
        --toh_num_pegs 4 \
        --toh_add_cstar \
        --algo_timeout 120 \
        --algo_min_remaining_gb 5.0 \
        --algo_heur NONE \
        --algo_mcts NONE


