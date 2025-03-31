"""
Dijkstra, A*, Bidirectional A*, MCTS for Sliding Tile, Pancake, Towers of Hanoi
- This code implements various search algorithms for solving the Sliding Tile, Pancake Sorting, and Towers of Hanoi problems.

Mostly generated from Gemini 2.5, with some modifications and fixes.
"""
import heapq
import math
import random
import time
import collections
from abc import ABC, abstractmethod # Can be used for formal interface if desired

# --- SlidingTileProblem Class (Keep as is) ---
class SlidingTileProblem:
    """Implements the Sliding Tile puzzle problem interface."""
    def __init__(self, initial_state, goal_state=None):
        self.initial_state_tuple = tuple(initial_state)
        self.n = int(math.sqrt(len(initial_state)))
        if self.n * self.n != len(initial_state):
            raise ValueError("Invalid state length for a square puzzle.")
        if goal_state:
            self.goal_state_tuple = tuple(goal_state)
        else:
            sorted_list = list(range(1, self.n * self.n)) + [0]
            self.goal_state_tuple = tuple(sorted_list)
        self._goal_positions = {tile: i for i, tile in enumerate(self.goal_state_tuple)}
        self._str_repr = f"SlidingTile-{self.n}x{self.n}"
    def initial_state(self): return self.initial_state_tuple
    def goal_state(self): return self.goal_state_tuple
    def is_goal(self, state): return state == self.goal_state_tuple
    def get_neighbors(self, state):
        neighbors = []
        blank_index = state.index(0)
        row, col = divmod(blank_index, self.n)
        moves = {'up': (row - 1, col), 'down': (row + 1, col), 'left': (row, col - 1), 'right': (row, col + 1)}
        for move_dir, (new_row, new_col) in moves.items():
            if 0 <= new_row < self.n and 0 <= new_col < self.n:
                new_blank_index = new_row * self.n + new_col
                new_state_list = list(state)
                new_state_list[blank_index], new_state_list[new_blank_index] = new_state_list[new_blank_index], new_state_list[blank_index]
                neighbors.append(tuple(new_state_list))
        return neighbors
    def heuristic(self, state):
        distance = 0
        for i, tile in enumerate(state):
            if tile != 0:
                current_row, current_col = divmod(i, self.n)
                goal_index = self._goal_positions.get(tile)
                if goal_index is None: continue
                goal_row, goal_col = divmod(goal_index, self.n)
                distance += abs(current_row - goal_row) + abs(current_col - goal_col)
        return distance
    def get_cost(self, state1, state2): return 1
    def __str__(self): return self._str_repr

# --- PancakeProblem Class (Keep as is) ---
class PancakeProblem:
    """Implements the Pancake Sorting problem interface."""
    def __init__(self, initial_state, goal_state=None):
        self.initial_state_tuple = tuple(initial_state)
        self.n = len(initial_state)
        if goal_state:
             self.goal_state_tuple = tuple(goal_state)
        else:
             self.goal_state_tuple = tuple(sorted(initial_state))
        self._str_repr = f"Pancake-{self.n}"
    def initial_state(self): return self.initial_state_tuple
    def goal_state(self): return self.goal_state_tuple
    def is_goal(self, state): return state == self.goal_state_tuple
    def get_neighbors(self, state):
        neighbors = []
        for k in range(2, self.n + 1):
            flipped_part = state[:k][::-1]
            rest_part = state[k:]
            neighbors.append(flipped_part + rest_part)
        return neighbors
    def heuristic(self, state):
        gaps = 0
        for i in range(self.n - 1):
            if abs(state[i] - state[i+1]) > 1:
                gaps += 1
        # Optional: Add 1 if largest element not at bottom (variant)
        # if self.n > 0 and state[-1] != self.n: # Assuming 1..N
        #    gaps += 1
        return gaps
    def get_cost(self, state1, state2): return 1
    def __str__(self): return self._str_repr

# --- Towers of Hanoi Problem Implementation ---

class TowersOfHanoiProblem:
    """Implements the Towers of Hanoi problem interface."""

    def __init__(self, num_disks, initial_peg='A', target_peg='C'):
        if num_disks < 1:
            raise ValueError("Number of disks must be at least 1.")
        self.num_disks = num_disks
        self.pegs = ['A', 'B', 'C']
        if initial_peg not in self.pegs or target_peg not in self.pegs or initial_peg == target_peg:
            raise ValueError(f"Invalid initial or target peg. Must be one of {self.pegs} and different.")

        self.initial_peg = initial_peg
        self.target_peg = target_peg
        # Determine auxiliary peg
        self.aux_peg = next(p for p in self.pegs if p != initial_peg and p != target_peg)

        # State representation: tuple where index is disk number (0=smallest, N-1=largest)
        # and value is the peg ('A', 'B', or 'C') the disk is on.
        self._initial_state = tuple([initial_peg] * num_disks)
        self._goal_state = tuple([target_peg] * num_disks)
        self._str_repr = f"TowersOfHanoi-{num_disks}"

    def initial_state(self):
        return self._initial_state

    def goal_state(self):
        return self._goal_state

    def is_goal(self, state):
        """Checks if the state is the goal state."""
        return state == self._goal_state

    def _get_peg_tops(self, state):
        """Helper to find the smallest (topmost) disk on each peg."""
        peg_tops = {p: None for p in self.pegs}
        # Store the disk index found for each peg
        peg_top_disk_index = {p: float('inf') for p in self.pegs} 
        
        for disk_index, peg in enumerate(state):
             if disk_index < peg_top_disk_index[peg]:
                  peg_top_disk_index[peg] = disk_index
                  peg_tops[peg] = disk_index # Store the index of the top disk
                  
        # Correct the value to be None if no disk found (inf index)
        for peg in self.pegs:
             if peg_top_disk_index[peg] == float('inf'):
                  peg_tops[peg] = None
                  
        return peg_tops


    def get_neighbors(self, state):
        """Generates valid neighbor states by moving one disk."""
        neighbors = []
        peg_tops = self._get_peg_tops(state)

        for source_peg in self.pegs:
            disk_to_move = peg_tops[source_peg] # Index of the disk on top of source_peg

            # Skip if source peg is empty
            if disk_to_move is None:
                continue

            for dest_peg in self.pegs:
                if source_peg == dest_peg:
                    continue

                top_disk_on_dest = peg_tops[dest_peg] # Index of the disk on top of dest_peg

                # Check validity: Move is valid if dest is empty OR
                # moving disk is smaller than the top disk on dest.
                # Smaller disk index means smaller disk.
                if top_disk_on_dest is None or disk_to_move < top_disk_on_dest:
                    # Create the new state
                    new_state_list = list(state)
                    new_state_list[disk_to_move] = dest_peg # Move the disk
                    neighbors.append(tuple(new_state_list))

        return neighbors

    def heuristic(self, state):
        """
        Calculates an admissible heuristic for Towers of Hanoi.
        Iterates from the largest disk down. If disk k is not on the peg
        it should be on (relative to disk k+1 or the final target),
        add 2^k to the heuristic value.
        """
        h_value = 0
        # The peg where disk k *should* be based on the position of larger disks
        current_target_peg = self.target_peg

        # Iterate from largest disk (N-1) down to smallest (0)
        for k in range(self.num_disks - 1, -1, -1):
            if state[k] == current_target_peg:
                # Disk k is on the correct peg relative to the disk below it (or target).
                # The target for the next smaller disk (k-1) remains the same.
                pass
            else:
                # Disk k is NOT on its target peg relative to the disk below it.
                # It (and all smaller disks) must move. Add lower bound cost 2^k.
                # Note: 2^k = pow(2, k)
                h_value += pow(2, k)
                # The next smaller disk (k-1) must now end up on the *third* peg
                # (neither the current peg of disk k, nor the peg disk k should have been on).
                # Update the target for disk k-1.
                current_target_peg = next(p for p in self.pegs if p != state[k] and p != current_target_peg)

        return h_value

    def get_cost(self, state1, state2):
        """Cost of each move is 1."""
        return 1

    def __str__(self):
        return self._str_repr


# --- Path Reconstruction Functions (Keep as is) ---
def reconstruct_path(came_from, start_state, goal_state):
    # ... (implementation from previous version) ...
    path = []
    current = goal_state
    start_node = start_state 
    if current == start_node: return [start_node]
    limit = 2**(len(start_state) + 5) # Safety limit for Hanoi
    count = 0
    while current != start_node:
        path.append(current)
        parent = came_from.get(current)
        if parent is None:
             print(f"Error: State {current} not found in came_from map during reconstruction.")
             return None 
        current = parent
        count += 1
        if count > limit: # Prevent infinite loops
            print("Error: Path reconstruction exceeded limit.")
            return None
    path.append(start_node)
    return path[::-1] 
    
def reconstruct_bidirectional_path(came_from_fwd, came_from_bwd, start_state, goal_state, meeting_node):
    # ... (implementation from previous version) ...
    path1 = []
    curr = meeting_node
    limit = 2**(len(start_state) + 5) # Safety limit
    count = 0
    while curr is not None: 
        path1.append(curr)
        curr = came_from_fwd.get(curr)
        count += 1
        if count > limit: print("Error: Path fwd reconstruction exceeded limit."); return None
    path1.reverse() 
    path2 = []
    curr = came_from_bwd.get(meeting_node) 
    count = 0
    while curr is not None: 
         path2.append(curr)
         curr = came_from_bwd.get(curr)
         count += 1
         if count > limit: print("Error: Path bwd reconstruction exceeded limit."); return None
    return path1 + path2


# --- Search Algorithms (Keep generic versions using 'problem' object) ---
# dijkstra_search(problem) - OK
# a_star_search(problem) - OK
# bidirectional_a_star_search(problem) - OK (using closed sets fix)
# MCTSNode class - OK
# mcts_search(problem, ...) - OK (using fix for UnboundLocalError)
# --- Include the finalized versions of these functions from previous steps ---
def dijkstra_search(problem):
    start_time = time.time()
    start_node = problem.initial_state()
    frontier = [(0, start_node)] 
    heapq.heapify(frontier)
    came_from = {start_node: None}
    cost_so_far = {start_node: 0}
    nodes_expanded = 0
    closed_set = set() # Also useful for Dijkstra variants/robustness
    while frontier:
        current_cost, current_state = heapq.heappop(frontier)
        nodes_expanded += 1
        if current_state in closed_set: continue # Already processed optimal path
        closed_set.add(current_state)
        # if current_cost > cost_so_far.get(current_state, float('inf')): continue # Alternative check        
        if problem.is_goal(current_state):
            end_time = time.time()
            path = reconstruct_path(came_from, start_node, current_state)
            return {"path": path, "cost": current_cost, "nodes_expanded": nodes_expanded, "time": end_time - start_time, "algorithm": "Dijkstra"}
        for neighbor_state in problem.get_neighbors(current_state):
            if neighbor_state in closed_set: continue # Skip if already processed
            new_cost = current_cost + problem.get_cost(current_state, neighbor_state)
            if new_cost < cost_so_far.get(neighbor_state, float('inf')):
                cost_so_far[neighbor_state] = new_cost
                priority = new_cost
                heapq.heappush(frontier, (priority, neighbor_state))
                came_from[neighbor_state] = current_state
    end_time = time.time()
    return {"path": None, "cost": -1, "nodes_expanded": nodes_expanded, "time": end_time - start_time, "algorithm": "Dijkstra"}


def a_star_search(problem):
    """Performs A* search on a generic problem using a closed set."""
    start_time = time.time()
    start_node = problem.initial_state()
    
    # Priority queue stores (f_score, state)
    h_initial = problem.heuristic(start_node)
    frontier = [(h_initial, start_node)] 
    heapq.heapify(frontier)

    # Data structures to track paths and costs
    came_from = {start_node: None}
    g_score = {start_node: 0} # Cost from start to node

    # Set to store nodes that have been expanded (popped and processed)
    closed_set = set()

    nodes_expanded = 0 # Counter for nodes popped from frontier

    while frontier:
        # Get node with the lowest f_score
        current_f_score, current_state = heapq.heappop(frontier)
        nodes_expanded += 1 

        # --- Check if already processed ---
        # If we've already processed this node via an optimal path, skip it.
        if current_state in closed_set:
            continue

        # --- Goal Check ---
        # Check for goal only after confirming it's not in the closed set
        if problem.is_goal(current_state):
            end_time = time.time()
            path = reconstruct_path(came_from, start_node, current_state)
            final_g_score = g_score.get(current_state) # Cost is the g_score of the goal
            # final_g_score will exist if path is found correctly
            return {
                "path": path,
                "cost": final_g_score,
                "nodes_expanded": nodes_expanded,
                "time": end_time - start_time,
                "algorithm": "A* (Closed Set)" # Indicate variant used
            }

        # --- Mark as processed ---
        # Add to closed set only after goal check and closed set check pass
        closed_set.add(current_state)

        # --- Explore Neighbors ---
        # Retrieve the g_score for the current state (must exist)
        current_g_score = g_score.get(current_state) 
        # A robustness check, though theoretically current_state should always be in g_score here
        if current_g_score is None: 
            print(f"Warning: Popped state {current_state} not found in g_score map!")
            continue 

        for neighbor_state in problem.get_neighbors(current_state):
            # Calculate tentative g_score for the neighbor
            cost = problem.get_cost(current_state, neighbor_state)
            tentative_g_score = current_g_score + cost

            # Check if this path to the neighbor is better than any known path
            if tentative_g_score < g_score.get(neighbor_state, float('inf')):
                # This path is better! Record it.
                came_from[neighbor_state] = current_state
                g_score[neighbor_state] = tentative_g_score
                
                # Calculate f_score and add to the frontier
                h_score = problem.heuristic(neighbor_state)
                f_score = tentative_g_score + h_score
                heapq.heappush(frontier, (f_score, neighbor_state))

    # If the loop finishes without finding the goal
    end_time = time.time()
    return {
        "path": None, 
        "cost": -1, 
        "nodes_expanded": nodes_expanded, 
        "time": end_time - start_time, 
        "algorithm": "A* (Closed Set)"
    }


def bidirectional_a_star_search(problem):
    start_time = time.time()
    start_node = problem.initial_state()
    goal_node = problem.goal_state()
    if problem.is_goal(start_node): return {"path": [start_node], "cost": 0, "nodes_expanded": 0, "time": 0, "algorithm": "Bidirectional A*"}
    h_start = problem.heuristic(start_node)
    frontier_fwd = [(h_start, start_node)] 
    came_from_fwd = {start_node: None}
    g_score_fwd = {start_node: 0}
    closed_fwd = set() 
    h_goal = problem.heuristic(goal_node) 
    frontier_bwd = [(h_goal, goal_node)] 
    came_from_bwd = {goal_node: None}
    g_score_bwd = {goal_node: 0} 
    closed_bwd = set() 
    nodes_expanded = 0
    best_path_cost = float('inf') 
    meeting_node = None
    while frontier_fwd and frontier_bwd:
        # --- Forward Step ---
        if frontier_fwd:
            _, current_state_fwd = heapq.heappop(frontier_fwd) # f_score ignored after pop, use g_score
            if current_state_fwd in closed_fwd: continue
            current_g_fwd = g_score_fwd.get(current_state_fwd, float('inf'))
            if current_g_fwd >= best_path_cost : continue             
            closed_fwd.add(current_state_fwd)
            nodes_expanded += 1
            if current_state_fwd in g_score_bwd: 
                current_path_cost = current_g_fwd + g_score_bwd[current_state_fwd]
                if current_path_cost < best_path_cost:
                    best_path_cost = current_path_cost
                    meeting_node = current_state_fwd
            for neighbor_state in problem.get_neighbors(current_state_fwd):
                if neighbor_state in closed_fwd: continue
                cost = problem.get_cost(current_state_fwd, neighbor_state)
                tentative_g_score = current_g_fwd + cost
                if tentative_g_score < g_score_fwd.get(neighbor_state, float('inf')):
                    came_from_fwd[neighbor_state] = current_state_fwd
                    g_score_fwd[neighbor_state] = tentative_g_score
                    h_score = problem.heuristic(neighbor_state)
                    f_score = tentative_g_score + h_score
                    if f_score < best_path_cost: heapq.heappush(frontier_fwd, (f_score, neighbor_state))
        # --- Backward Step ---
        if frontier_bwd:
            _, current_state_bwd = heapq.heappop(frontier_bwd) # f_score ignored after pop
            if current_state_bwd in closed_bwd: continue
            current_g_bwd = g_score_bwd.get(current_state_bwd, float('inf'))
            if current_g_bwd >= best_path_cost: continue                  
            closed_bwd.add(current_state_bwd) 
            nodes_expanded += 1
            if current_state_bwd in g_score_fwd: 
                current_path_cost = g_score_fwd[current_state_bwd] + current_g_bwd
                if current_path_cost < best_path_cost:
                    best_path_cost = current_path_cost
                    meeting_node = current_state_bwd
            for neighbor_state in problem.get_neighbors(current_state_bwd):
                if neighbor_state in closed_bwd: continue
                cost = problem.get_cost(current_state_bwd, neighbor_state) 
                tentative_g_score = current_g_bwd + cost 
                if tentative_g_score < g_score_bwd.get(neighbor_state, float('inf')):
                    came_from_bwd[neighbor_state] = current_state_bwd 
                    g_score_bwd[neighbor_state] = tentative_g_score
                    h_score = problem.heuristic(neighbor_state) 
                    f_score = tentative_g_score + h_score
                    if f_score < best_path_cost: heapq.heappush(frontier_bwd, (f_score, neighbor_state))
    end_time = time.time()
    if meeting_node:
        path = reconstruct_bidirectional_path(came_from_fwd, came_from_bwd, start_node, goal_node, meeting_node)
        final_cost = len(path) - 1 if path else -1
        if path and final_cost != best_path_cost : print(f"Warning: Bidirectional cost mismatch! Path={final_cost}, U={best_path_cost}")
        final_cost = best_path_cost # Trust calculated cost U
        return {"path": path, "cost": final_cost if path else -1, "nodes_expanded": nodes_expanded, "time": end_time - start_time, "algorithm": "Bidirectional A*"}
    else:
        return {"path": None, "cost": -1, "nodes_expanded": nodes_expanded, "time": end_time - start_time, "algorithm": "Bidirectional A*"}

class MCTSNode:
    def __init__(self, state, parent=None, problem=None): 
        self.state = state
        self.parent = parent
        self.problem = problem 
        self.children = []
        self.visits = 0
        self.value = 0.0  
        self._untried_actions = problem.get_neighbors(state) if problem else []
        random.shuffle(self._untried_actions)
    def is_fully_expanded(self): return len(self._untried_actions) == 0
    def best_child(self, exploration_weight=1.41): 
        if not self.children: return None
        if self.visits == 0: return random.choice(self.children) 
        log_total_visits = math.log(self.visits)
        def ucb1(node):
            if node.visits == 0: return float('inf')
            exploitation = node.value / node.visits
            exploration = exploration_weight * math.sqrt(log_total_visits / node.visits)
            return exploitation + exploration
        best_node = max(self.children, key=lambda node: ucb1(node) + random.uniform(0, 1e-6))
        return best_node
    def expand(self):
        if not self._untried_actions: return None 
        action_state = self._untried_actions.pop()
        child_node = MCTSNode(action_state, parent=self, problem=self.problem)
        self.children.append(child_node)
        return child_node
    def is_terminal(self): return self.problem.is_goal(self.state) if self.problem else False

def mcts_search(problem, iterations=1000, max_depth=50):
    start_time = time.time()
    start_node = problem.initial_state()
    root = MCTSNode(state=start_node, problem=problem)
    if root.is_terminal(): return {"path": [root.state], "cost": 0, "nodes_expanded": 1, "time": time.time()-start_time, "algorithm": "MCTS", "iterations": 0}
    for i in range(iterations):
        node = root
        path_to_leaf = [node]
        # 1. Selection
        while not node.is_terminal() and node.is_fully_expanded():
            selected_child = node.best_child() 
            if selected_child is None: 
                 print(f"Warning: MCTS Selection encountered None best_child for fully expanded node {node.state}. Stopping iteration.")
                 break 
            node = selected_child 
            path_to_leaf.append(node)
        # 2. Expansion
        expanded_node = None 
        if not node.is_terminal() and not node.is_fully_expanded():
            expanded_node = node.expand() 
            if expanded_node: 
                 node = expanded_node 
                 path_to_leaf.append(node)
        # 3. Simulation
        current_state = node.state 
        reward = 0
        sim_depth = 0
        sim_path_cost = 0 
        sim_history = {current_state} 
        while not problem.is_goal(current_state) and sim_depth < max_depth:
            neighbors = problem.get_neighbors(current_state)
            valid_neighbors = [n for n in neighbors if n not in sim_history]
            if not valid_neighbors: break 
            current_state = random.choice(valid_neighbors)
            sim_history.add(current_state)
            sim_depth += 1
            sim_path_cost += 1 
        if problem.is_goal(current_state): reward = 1000.0 / (1 + sim_path_cost) 
        else: reward = -problem.heuristic(current_state) 
        # 4. Backpropagation
        for node_in_path in reversed(path_to_leaf):
            node_in_path.visits += 1
            node_in_path.value += reward 
    end_time = time.time()
    # --- Extracting Path from MCTS Tree ---
    goal_node_in_tree = None
    min_cost_in_tree = float('inf')
    queue = collections.deque([(root, 0)]) 
    visited_in_tree = {root.state: 0}
    nodes_explored_in_tree = 0
    best_path_found = None
    while queue:
         current_node, current_cost = queue.popleft()
         nodes_explored_in_tree += 1
         if current_node.is_terminal():
             if current_cost < min_cost_in_tree:
                  min_cost_in_tree = current_cost
                  goal_node_in_tree = current_node
                  path = []
                  temp = goal_node_in_tree
                  while temp is not None: path.append(temp.state); temp = temp.parent
                  best_path_found = path[::-1]
         for child in current_node.children:
             cost_step = problem.get_cost(current_node.state, child.state)
             new_cost = current_cost + cost_step
             if child.state not in visited_in_tree or new_cost < visited_in_tree[child.state]:
                  visited_in_tree[child.state] = new_cost
                  queue.append((child, new_cost))
    if best_path_found:
        return {"path": best_path_found, "cost": min_cost_in_tree, "nodes_expanded": nodes_explored_in_tree, "time": end_time - start_time, "algorithm": "MCTS", "iterations": iterations, "tree_root_visits": root.visits }
    else:
         best_first_move_node = root.best_child(exploration_weight=0) 
         return {"path": None, "cost": -1, "nodes_expanded": nodes_explored_in_tree, "time": end_time - start_time, "algorithm": "MCTS", "iterations": iterations, "best_next_state_estimate": best_first_move_node.state if best_first_move_node else None, "tree_root_visits": root.visits}


# --- Main Execution Logic ---

if __name__ == "__main__":
    # --- Define Problems ---

    # Sliding Tile (8-Puzzle) - Harder
    tile_initial = [1, 2, 3, 0, 4, 6, 7, 5, 8] # Medium
    #tile_initial = [8, 6, 7, 2, 5, 4, 3, 0, 1] # harder
    tile_goal    = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    sliding_tile_problem = SlidingTileProblem(initial_state=tile_initial, goal_state=tile_goal)

    # Pancake Sorting (8-pancake)
    pancake_initial = (8, 3, 5, 1, 6, 4, 2, 7)
    pancake_problem = PancakeProblem(initial_state=pancake_initial)

    # Towers of Hanoi (Adjust N - smaller N for faster testing)
    # N=3: Optimal 7 moves
    # N=4: Optimal 15 moves
    # N=7: Optimal 127 moves
    # N=10: Optimal 1023 moves (May start to get slow for Dijkstra/MCTS)
    hanoi_disks = 7 
    hanoi_problem = TowersOfHanoiProblem(num_disks=hanoi_disks, initial_peg='A', target_peg='C')

    problems_to_solve = [
        sliding_tile_problem,
        pancake_problem,
        hanoi_problem
    ]

    search_algorithms = [
        dijkstra_search,
        a_star_search,
        bidirectional_a_star_search,
        mcts_search
    ]

    # --- Run Experiments ---
    all_results = []

    for problem in problems_to_solve:
        print(f"\n" + "=" * 20)
        print(f"Solving: {problem}")
        print(f"Initial State: {problem.initial_state()}")
        print(f"Goal State:    {problem.goal_state()}")
        print(f"Initial Heuristic: {problem.heuristic(problem.initial_state())}")
        print("-" * 20)

        problem_results = []
        for algorithm_func in search_algorithms:
            algo_name = algorithm_func.__name__
            print(f"Running {algo_name}...")

            result = None
            try:
                if algo_name == 'mcts_search':
                    # Adjust iterations/depth based on problem complexity maybe?
                    # Hanoi benefits from more iterations due to large branching factor near start
                    iterations = 100000
                    depth = 100
                    #if isinstance(problem, TowersOfHanoiProblem):
                    #    iterations = 20000 # More for Hanoi?
                    #    depth = problem.num_disks * 3 # Generous depth limit

                    result = algorithm_func(problem, iterations=iterations, max_depth=depth)
                else:
                    result = algorithm_func(problem)
                
                print(f"{algo_name} Done.")

            except Exception as e:
                print(f"!!! ERROR during {algo_name} on {problem}: {e}")
                import traceback
                traceback.print_exc() # Print detailed traceback
                result = { # Create a dummy result indicating failure
                     "path": None, "cost": -1, "nodes_expanded": -1, 
                     "time": -1, "algorithm": algo_name, "error": str(e)
                }


            if result: # Only process if no exception or dummy created
                 result['problem'] = str(problem) # Add problem identifier
                 problem_results.append(result)
                 all_results.append(result) # Also add to overall list


        # --- Print Results for this Problem ---
        print("\n" + "=" * 10 + f" Results for {problem} " + "=" * 10)
        for res in problem_results:
            print(f"\nAlgorithm: {res['algorithm']}")
            if 'iterations' in res: print(f"MCTS Iterations: {res['iterations']}")
            if 'tree_root_visits' in res: print(f"MCTS Root Visits: {res['tree_root_visits']}")
            print(f"Time Taken: {res['time']:.4f} seconds")
            print(f"Nodes Expanded/Explored: {res['nodes_expanded']}")
            print(f"Path Found: {'Yes' if res['path'] else 'No'}")
            if res['path']:
                print(f"Path Cost: {res['cost']}")
            else:
                 print("Path Cost: N/A")
                 if 'best_next_state_estimate' in res and res['best_next_state_estimate']:
                       print(f"MCTS Best Next State Estimate: {res['best_next_state_estimate']}")
                 if 'error' in res:
                       print(f"ERROR during run: {res['error']}") # Display error if one occurred

        print("=" * (20 + len(str(problem)) + 14))


    print("\n" + "*"*15 + " Overall Summary " + "*"*15)
    for res in all_results:
         status = f"Cost: {res['cost']}" if res['path'] else "No Path Found"
         if 'error' in res: status = f"ERROR: {res['error']}"
         print(f"- Problem: {res.get('problem','N/A')}, Algorithm: {res.get('algorithm','N/A')}, Time: {res.get('time',-1):.4f}s, Nodes: {res.get('nodes_expanded',-1)}, Status: {status}")
        