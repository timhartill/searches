**Search Arena** 

Here you can find Python-based versions of various unidirectional and bidirectional search algorithms that can be run against standard evaluation domains, namely Sliding Tile, Pancake, Towers of Hanoi, and Pathfinding (Dragon Age Origins grids and various mazes sourced from https://www.movingai.com/benchmarks/grids.html). 


**Setup and Installation**

Tested under Python 3.12 on Ubuntu and CentOS. 

1. Install Python into a virtual environment.
2. Install _numpy_, and _sortedcontainers_.
3. Clone this repository. The project structure is very simple: all code and run scripts are in _/code_, all problems are specified in text files in subdirectories off _/problems_, all outputs will appear in _/outputs_ which will be created dynamically.


**Run a set of algorithms on a set of problems**

1. From a terminal in the _/code_ subdirectory run: _bash run_test_easy.sh_
2. Results and logs will be in _/outputs_.
3. Reviewing _bash run_test_easy.sh_ will give you the idea as to how to run different algorithms on different problems. Basically everything starts in _search_runner.py_.

**Adding Rust Modules (Optional)**

1. pip install _maturin_
2. Run _maturin develop --release_ on the command line to compile the Rust library _.../code/src/lib.rs_ as an importable Python package _rust_utils_. 
3. Add a _--rust_heur_ flag to your run script to run the Rust version of the manhattan heuristic for Sliding Tile. Speeds up 4x4 Sliding Tile problems by 20% or so.
4. Add a _--rust_ flag to your run script to use a Rust version of the node dictionary that stores g, h and the parent for each state. Actually runs a bit slower dues to the calling overhead but saves a modest amount on memory usage.

