**Search Arena** 

Here you can find Python-based versions of various unidirectional and bidirectional search algorithms that can be run against standard evaluation domains, namely Sliding Tile, Pancake, Towers of Hanoi, and Pathfinding (Dragon Age Origins grids and various mazes sourced from https://www.movingai.com/benchmarks/grids.html). 


**Setup and Installation**

Tested under Python 3.12 on Ubuntu and CentOS. 

1. Install Python into a virtual environment.
2. Install numpy, sortedcontainers and maturin
3. This project contains a few Rust utilities. To install Rust, go to https://www.rust-lang.org/tools/install and follow the simple installation instruction found there. 
4. Clone this repository. The project structure is very simple: all code and run scripts are in _/code_, all problems are specified in text files in subdirectories off _/problems_, all outputs will appear in _/outputs_ which will be created dynamically.
6. If running on a platform other than Ubuntu, compile the Rust utilies by running _maturin develop_ from the _/code_ subdirectory.


**Run a set of algorithms on a set of problems**

1. From a terminal in the _/code_ subdirectory run: _bash run_test_easy.sh_
2. Results and logs will be in _/outputs_.
3. Reviewing _bash run_test_easy.sh_ will give you the idea as to how to run different algorithms on different problems. Basically everything starts in _search_runner.py_.



