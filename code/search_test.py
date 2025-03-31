""" Different searches on Sliding Tile Puzzle (8-puzzle) using Dijkstra, A*, Bidirectional A*, and MCTS.
"""


import heapq
import math
import random
import time
import collections

# --- Puzzle Representation and Logic ---

class SlidingTilePuzzle:
    """Represents the sliding tile puzzle state and operations."""

    def __init__(self, initial_state, goal_state):
        self.initial_state = tuple(initial_state)
        self.goal_state = tuple(goal_state)
        self.n = int(math.sqrt(len(initial_state)))
        if self.n * self.n != len(initial_state):
            raise ValueError("Invalid state length for a square puzzle.")
        
        # Pre-compute goal positions for faster heuristic calculation
        self.goal_positions = {tile: i for i, tile in enumerate(goal_state)}

    def get_neighbors(self, state):
        """Generates valid neighbor states by moving the blank tile."""
        neighbors = []
        blank_index = state.index(0)
        row, col = divmod(blank_index, self.n)

        moves = {
            'up': (row - 1, col),
            'down': (row + 1, col),
            'left': (row, col - 1),
            'right': (row, col + 1)
        }

        for move_dir, (new_row, new_col) in moves.items():
            if 0 <= new_row < self.n and 0 <= new_col < self.n:
                new_blank_index = new_row * self.n + new_col
                new_state_list = list(state)
                # Swap blank tile with the target tile
                new_state_list[blank_index], new_state_list[new_blank_index] = \
                    new_state_list[new_blank_index], new_state_list[blank_index]
                neighbors.append(tuple(new_state_list))
        return neighbors

    def manhattan_distance(self, state):
        """Calculates the Manhattan distance heuristic."""
        distance = 0
        for i, tile in enumerate(state):
            if tile != 0: # Ignore the blank tile
                current_row, current_col = divmod(i, self.n)
                goal_index = self.goal_positions[tile]
                goal_row, goal_col = divmod(goal_index, self.n)
                distance += abs(current_row - goal_row) + abs(current_col - goal_col)
        return distance

    def is_goal(self, state):
        """Checks if the state is the goal state."""
        return state == self.goal_state

# --- Path Reconstruction ---

def reconstruct_path(came_from, start_state, goal_state):
    """Reconstructs the path from start to goal."""
    path = []
    current = goal_state
    while current != start_state:
        path.append(current)
        # Check if current state is in came_from before accessing
        if current not in came_from:
             print(f"Error: State {current} not found in came_from map during reconstruction.")
             return None # Or raise an error
        current = came_from[current]
    path.append(start_state)
    return path[::-1] # Return path from start to goal

def reconstruct_bidirectional_path(came_from_fwd, came_from_bwd, start_state, goal_state, meeting_node):
    """Reconstructs path for bidirectional search."""
    path_fwd = []
    current = meeting_node
    while current != start_state:
        path_fwd.append(current)
        current = came_from_fwd[current]
    path_fwd.append(start_state)
    path_fwd.reverse()

    path_bwd = []
    current = meeting_node
    while current != goal_state:
         # Skip the meeting node itself in the backward path
        prev = came_from_bwd[current]
        path_bwd.append(prev)
        current = prev
    # path_bwd already goes from meeting_node's predecessor towards goal_state
    
    # Combine paths: path_fwd goes start -> meeting_node, path_bwd goes meeting_node_pred -> goal
    # Need to reconstruct backward path correctly: goal -> ... -> meeting_node_pred
    
    # Let's re-trace backward path correctly
    path_bwd_correct = []
    current = came_from_bwd[meeting_node] # Start from the node *before* the meeting node in backward search
    while current != goal_state:
         path_bwd_correct.append(current)
         current = came_from_bwd[current]
    path_bwd_correct.append(goal_state)
    
    # Combine path_fwd (start...meeting) and path_bwd_correct (node_after_meeting...goal)
    # The came_from_bwd stores parent *towards* the goal state in the backward search tree
    # So reconstructing from meeting node using came_from_bwd goes away from goal
    
    # Let's rethink reconstruction:
    # Path 1: start -> ... -> came_from_fwd[meeting_node] -> meeting_node
    # Path 2: goal -> ... -> came_from_bwd[meeting_node] -> meeting_node
    
    path1 = []
    curr = meeting_node
    while curr is not None:
        path1.append(curr)
        curr = came_from_fwd.get(curr) # Use .get for safety
    path1.reverse() # Now start -> meeting_node

    path2 = []
    curr = came_from_bwd.get(meeting_node) # Start from parent in backward search
    while curr is not None:
         path2.append(curr)
         curr = came_from_bwd.get(curr)
    # path2 is now parent_of_meeting_node -> ... -> goal

    return path1 + path2 # Combine path from start and path from meeting node's successor (in reverse search) to goal


# --- Dijkstra's Algorithm ---

def dijkstra_search(puzzle):
    """Performs Dijkstra's search (Uniform Cost Search).
    Priority is determined solely by cost_so_far (g(n)).
    Explores states layer by layer based on cost from the start. 
    Guaranteed to find the shortest path in terms of number of moves (since cost is uniform).
    """
    start_time = time.time()
    
    # Priority queue stores (cost, state)
    frontier = [(0, puzzle.initial_state)] 
    heapq.heapify(frontier)
    
    came_from = {puzzle.initial_state: None}
    cost_so_far = {puzzle.initial_state: 0}
    nodes_expanded = 0

    while frontier:
        current_cost, current_state = heapq.heappop(frontier)
        nodes_expanded += 1

        if puzzle.is_goal(current_state):
            end_time = time.time()
            path = reconstruct_path(came_from, puzzle.initial_state, current_state)
            return {
                "path": path,
                "cost": current_cost,
                "nodes_expanded": nodes_expanded,
                "time": end_time - start_time,
                "algorithm": "Dijkstra"
            }

        # Optimization: If we found a shorter path already, skip
        if current_cost > cost_so_far.get(current_state, float('inf')):
             continue

        for neighbor_state in puzzle.get_neighbors(current_state):
            new_cost = current_cost + 1 # Cost of each move is 1
            if neighbor_state not in cost_so_far or new_cost < cost_so_far[neighbor_state]:
                cost_so_far[neighbor_state] = new_cost
                priority = new_cost # Dijkstra uses only g(n)
                heapq.heappush(frontier, (priority, neighbor_state))
                came_from[neighbor_state] = current_state

    end_time = time.time()
    return {"path": None, "cost": -1, "nodes_expanded": nodes_expanded, "time": end_time - start_time, "algorithm": "Dijkstra"} # Goal not found


# --- A* Search ---

def a_star_search(puzzle):
    """Performs A* search."""
    start_time = time.time()
    
    # Priority queue stores (f_score, state)
    # f_score = g_score + h_score
    h_initial = puzzle.manhattan_distance(puzzle.initial_state)
    frontier = [(h_initial, puzzle.initial_state)] # f = 0 + h
    heapq.heapify(frontier)

    came_from = {puzzle.initial_state: None}
    g_score = {puzzle.initial_state: 0} # Cost from start
    nodes_expanded = 0

    while frontier:
        current_f_score, current_state = heapq.heappop(frontier)
        nodes_expanded += 1

        if puzzle.is_goal(current_state):
            end_time = time.time()
            path = reconstruct_path(came_from, puzzle.initial_state, current_state)
            return {
                "path": path,
                "cost": g_score[current_state],
                "nodes_expanded": nodes_expanded,
                "time": end_time - start_time,
                "algorithm": "A*"
            }

        # Optimization: If we pulled a state with a higher f-score than its current g-score + h, skip
        # (This check isn't strictly necessary if heuristic is consistent, but good practice)
        # A better check: if current_state already processed via lower g_score path
        # We handle this implicitly by checking g_score before adding/updating neighbors


        for neighbor_state in puzzle.get_neighbors(current_state):
            tentative_g_score = g_score[current_state] + 1 # Cost of each move is 1

            if tentative_g_score < g_score.get(neighbor_state, float('inf')):
                # This path to neighbor is better than any previous one. Record it.
                came_from[neighbor_state] = current_state
                g_score[neighbor_state] = tentative_g_score
                h_score = puzzle.manhattan_distance(neighbor_state)
                f_score = tentative_g_score + h_score
                heapq.heappush(frontier, (f_score, neighbor_state))

    end_time = time.time()
    return {"path": None, "cost": -1, "nodes_expanded": nodes_expanded, "time": end_time - start_time, "algorithm": "A*"} # Goal not found


# --- Bidirectional Heuristic Search (Bidirectional A*) ---
# NOTE: Correct implementation, especially termination condition, can be complex.
# This is a simplified conceptual illustration.
# "Lower bound pairs" might refer to specific pruning techniques (like MM algorithm)
# which are not implemented here for simplicity.

def bidirectional_a_star_search(puzzle):
    """Performs Bidirectional A* search (conceptual)."""
    start_time = time.time()

    # --- Forward Search Structures ---
    frontier_fwd = [(puzzle.manhattan_distance(puzzle.initial_state), puzzle.initial_state)] # (f, state)
    came_from_fwd = {puzzle.initial_state: None}
    g_score_fwd = {puzzle.initial_state: 0}
    visited_fwd = {puzzle.initial_state} # Track nodes fully processed

    # --- Backward Search Structures ---
    # Heuristic for backward search estimates cost from state to initial_state
    # For Manhattan distance, h(n -> goal) == h(goal -> n) usually, but let's define explicitly
    # We actually need h(state -> start) for backward search priority queue.
    # But A* typically uses h(state -> *target*), so backward search uses h(state -> initial_state)
    # Let's use standard h(state->goal) for priority, consistent with A*.
    h_goal = 0 # h(goal->goal)=0, but backward starts at goal
    frontier_bwd = [(puzzle.manhattan_distance(puzzle.goal_state), puzzle.goal_state)] # (f, state) - h estimates cost TO GOAL
    came_from_bwd = {puzzle.goal_state: None}
    g_score_bwd = {puzzle.goal_state: 0} # Cost from GOAL
    visited_bwd = {puzzle.goal_state} # Track nodes fully processed

    nodes_expanded = 0
    best_path_cost = float('inf')
    meeting_node = None

    # --- Main Loop ---
    while frontier_fwd and frontier_bwd:
        
        # --- Forward Step ---
        if frontier_fwd: # Check if not empty
            # Peek minimum f-score without popping yet for termination check later
            f_min_fwd, _ = frontier_fwd[0] # Smallest f-score in forward queue
            
            # Termination Check (Simplified): Stop if sum of lowest costs exceeds best path
            # A more robust check involves the heuristics. A common one:
            # if f_min_fwd + f_min_bwd >= best_path_cost: break 
            # where f_min_bwd is min f-score in backward queue.
            # A simpler check: If lowest g_fwd + g_bwd >= best_path_cost
            # Let's use a basic check for illustration: stop if frontiers meet costlier than best path
            # We need g_score of the nodes with f_min_fwd/f_min_bwd for better check.
            
            # If the node we are about to expand from forward search has been reached
            # by backward search, its path cost is g_fwd + g_bwd.
            # Check if combined cost might be better than current best
            
            current_f_fwd, current_state_fwd = heapq.heappop(frontier_fwd)
            nodes_expanded += 1
            
            # Check for meeting point
            if current_state_fwd in g_score_bwd:
                current_path_cost = g_score_fwd[current_state_fwd] + g_score_bwd[current_state_fwd]
                if current_path_cost < best_path_cost:
                    best_path_cost = current_path_cost
                    meeting_node = current_state_fwd
                    # Don't stop yet, a shorter path might be found via another meeting point

            # Expand forward
            if g_score_fwd[current_state_fwd] >= best_path_cost: # Pruning based on current best
                 continue # No need to expand if already worse than best complete path


            for neighbor_state in puzzle.get_neighbors(current_state_fwd):
                tentative_g_score = g_score_fwd[current_state_fwd] + 1
                if tentative_g_score < g_score_fwd.get(neighbor_state, float('inf')):
                    came_from_fwd[neighbor_state] = current_state_fwd
                    g_score_fwd[neighbor_state] = tentative_g_score
                    h_score = puzzle.manhattan_distance(neighbor_state)
                    f_score = tentative_g_score + h_score
                    heapq.heappush(frontier_fwd, (f_score, neighbor_state))
                    visited_fwd.add(neighbor_state)


        # --- Backward Step ---
        if frontier_bwd: # Check if not empty
             
            # Termination check element (similar reasoning)
            
            current_f_bwd, current_state_bwd = heapq.heappop(frontier_bwd)
            nodes_expanded += 1

            # Check for meeting point
            if current_state_bwd in g_score_fwd:
                current_path_cost = g_score_fwd[current_state_bwd] + g_score_bwd[current_state_bwd]
                if current_path_cost < best_path_cost:
                    best_path_cost = current_path_cost
                    meeting_node = current_state_bwd
            
            if g_score_bwd[current_state_bwd] >= best_path_cost: # Pruning
                 continue


            # Expand backward (neighbors lead *towards* the start state conceptually)
            for neighbor_state in puzzle.get_neighbors(current_state_bwd):
                 # Note: get_neighbors gives states reachable *from* current_state_bwd
                 # In backward search, these neighbors are potential *parents* towards goal
                 tentative_g_score = g_score_bwd[current_state_bwd] + 1
                 if tentative_g_score < g_score_bwd.get(neighbor_state, float('inf')):
                     came_from_bwd[neighbor_state] = current_state_bwd # Parent towards goal
                     g_score_bwd[neighbor_state] = tentative_g_score
                     # Heuristic estimates cost from neighbor to START state
                     # Using h(neighbor -> goal) for consistency in A* priority
                     h_score = puzzle.manhattan_distance(neighbor_state)
                     f_score = tentative_g_score + h_score
                     heapq.heappush(frontier_bwd, (f_score, neighbor_state))
                     visited_bwd.add(neighbor_state)
        
        # --- More Robust Termination Condition (Conceptual) ---
        # Check if the sum of the minimum g-scores at the intersection of frontiers 
        # exceeds the best path found so far. Or check f-scores:
        if frontier_fwd and frontier_bwd:
            min_f_fwd, _ = frontier_fwd[0]
            min_f_bwd, _ = frontier_bwd[0]
            # A common condition: stop when min_f_fwd + min_f_bwd >= best_path_cost (requires consistent heuristic)
            # Simpler: Check minimum g-scores at intersection
            potential_min_cost = float('inf')
            intersection = visited_fwd.intersection(visited_bwd)
            if intersection:
                 potential_min_cost = min(g_score_fwd[n] + g_score_bwd[n] for n in intersection)

            if potential_min_cost >= best_path_cost and meeting_node is not None:
                 break # Likely found the optimal path


    end_time = time.time()

    if meeting_node:
        # Path reconstruction needs careful joining at meeting_node
        # Path: initial -> ... -> came_from_fwd[meeting_node] -> meeting_node
        # Path: goal -> ... -> came_from_bwd[meeting_node] -> meeting_node
        # Combine them correctly.
        
        # Simpler reconstruction for demo:
        path = reconstruct_bidirectional_path(came_from_fwd, came_from_bwd, puzzle.initial_state, puzzle.goal_state, meeting_node)

        # Verify cost matches path length if path reconstruction is complex
        actual_cost = len(path) - 1 if path else -1 

        return {
            "path": path,
            "cost": best_path_cost, # Or actual_cost if verified
            "nodes_expanded": nodes_expanded,
            "time": end_time - start_time,
            "algorithm": "Bidirectional A*"
        }
    else:
        return {"path": None, "cost": -1, "nodes_expanded": nodes_expanded, "time": end_time - start_time, "algorithm": "Bidirectional A*"}


# --- Monte Carlo Tree Search (MCTS) ---
# MCTS is less suited for finding the *optimal* shortest path in deterministic
# problems like this compared to A*. It's more for games or large state spaces.
# This implementation demonstrates the MCTS structure but may not be efficient
# or guarantee optimality for the sliding tile puzzle.

class MCTSNode:
    def __init__(self, state, parent=None, puzzle=None):
        self.state = state
        self.parent = parent
        self.puzzle = puzzle # Need puzzle logic for neighbors and goal check
        self.children = []
        self.visits = 0
        self.value = 0.0  # Total reward (e.g., negative path length, or 1 for win, 0 for loss)
        self.untried_actions = puzzle.get_neighbors(state) if puzzle else []
        random.shuffle(self.untried_actions) # Shuffle for random exploration

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, exploration_weight=1.41): # UCB1
        if not self.children:
             return None # Should not happen if called after expansion or on non-terminal

        log_total_visits = math.log(self.visits)

        def ucb1(node):
            if node.visits == 0:
                return float('inf') # Prioritize unvisited nodes
            exploitation = node.value / node.visits
            exploration = exploration_weight * math.sqrt(log_total_visits / node.visits)
            return exploitation + exploration

        return max(self.children, key=ucb1)

    def expand(self):
        if not self.untried_actions:
             print("Warning: Expanding a node with no untried actions.")
             return None # Should not happen if called correctly

        action_state = self.untried_actions.pop()
        child_node = MCTSNode(action_state, parent=self, puzzle=self.puzzle)
        self.children.append(child_node)
        return child_node

    def is_terminal(self):
         return self.puzzle.is_goal(self.state) if self.puzzle else False


def mcts_search(puzzle, iterations=1000, max_depth=50):
    """Performs Monte Carlo Tree Search."""
    start_time = time.time()
    root = MCTSNode(state=puzzle.initial_state, puzzle=puzzle)

    if root.is_terminal():
         return {"path": [root.state], "cost": 0, "nodes_expanded": 1, "time": time.time()-start_time, "algorithm": "MCTS", "iterations": 0}

    for i in range(iterations):
        node = root
        path_to_leaf = [node] # Keep track for backpropagation

        # 1. Selection: Traverse the tree using UCB1
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()
            if node is None: # Should not happen in theory unless stuck
                print("Warning: Selection reached a dead end.")
                break
            path_to_leaf.append(node)
        
        if node is None: continue # Skip if selection failed

        # 2. Expansion: If the selected node is not terminal and not fully expanded
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()
            if node:
                 path_to_leaf.append(node)

        # 3. Simulation (Rollout): From the new/selected node, simulate randomly
        current_state = node.state
        depth = 0
        sim_path_len = 0
        while not puzzle.is_goal(current_state) and depth < max_depth:
            neighbors = puzzle.get_neighbors(current_state)
            if not neighbors: break # No moves possible
            current_state = random.choice(neighbors)
            depth += 1
            sim_path_len += 1

        # Determine reward: Higher is better. Use negative path length?
        # Or simple win/loss. Let's use win=1, loss=0 for simplicity here.
        # A reward based on closeness (heuristic) might guide better.
        reward = 0
        if puzzle.is_goal(current_state):
             reward = 1 # Found goal in simulation
        # Alternative reward: reward = -sim_path_len (penalize long paths)

        # 4. Backpropagation: Update visit counts and value up the tree
        for node_in_path in reversed(path_to_leaf):
            node_in_path.visits += 1
            node_in_path.value += reward # Add reward (win/loss or neg length)

    end_time = time.time()

    # Choose the best move from the root based on visits or value
    # For pathfinding, we want the path itself.
    # If the goal was found *during selection/expansion*, we might have a path.
    
    # Let's find if the goal exists in the constructed tree
    goal_node = None
    queue = collections.deque([root])
    visited_states = {root.state}
    best_path_in_tree = None
    min_cost_in_tree = float('inf')

    while queue:
        current_node = queue.popleft()
        if current_node.is_terminal():
             # Reconstruct path from this goal node found in the tree
             path = []
             temp = current_node
             while temp is not None:
                 path.append(temp.state)
                 temp = temp.parent
             path.reverse()
             cost = len(path) - 1
             if cost < min_cost_in_tree:
                  min_cost_in_tree = cost
                  best_path_in_tree = path
             # Don't necessarily stop, might find shorter path in tree later

        for child in current_node.children:
             if child.state not in visited_states:
                  visited_states.add(child.state)
                  queue.append(child)


    # Count total nodes created in the tree (approx expansion count)
    nodes_created = len(visited_states) # Rough measure

    if best_path_in_tree:
        return {
            "path": best_path_in_tree,
            "cost": min_cost_in_tree,
            "nodes_expanded": nodes_created, # More accurately 'nodes explored/created'
            "time": end_time - start_time,
            "algorithm": "MCTS",
            "iterations": iterations
        }
    else:
        # If goal not found in tree, return info about best first move (most visited child)
         best_first_move_node = root.best_child(exploration_weight=0) # Choose best based purely on exploitation/visits
         return {
            "path": None, # Goal not reached in explored tree
            "cost": -1,
            "nodes_expanded": nodes_created,
            "time": end_time - start_time,
            "algorithm": "MCTS",
            "iterations": iterations,
            "best_next_state_estimate": best_first_move_node.state if best_first_move_node else None
        }


# --- Example Usage ---

if __name__ == "__main__":
    # Define initial and goal states (8-puzzle)
    # Example: A moderately difficult 8-puzzle
    initial = [1, 2, 3, 
               0, 4, 6, 
               7, 5, 8]
    # initial = [8, 6, 7, 2, 5, 4, 3, 0, 1] # Harder one
    goal    = [1, 2, 3, 
               4, 5, 6, 
               7, 8, 0]

    puzzle = SlidingTilePuzzle(initial_state=initial, goal_state=goal)

    print(f"Initial State: {puzzle.initial_state}")
    print(f"Goal State:    {puzzle.goal_state}")
    print("-" * 30)

    # --- Run Algorithms ---
    results = []
    
    print("Running Dijkstra...")
    result_dijkstra = dijkstra_search(puzzle)
    results.append(result_dijkstra)
    print("Dijkstra Done.")

    print("Running A*...")
    result_astar = a_star_search(puzzle)
    results.append(result_astar)
    print("A* Done.")

    print("Running Bidirectional A*...")
    # Be aware: Bidirectional can be tricky, reconstruction might need debugging
    # Ensure the logic handles meeting points and path merging correctly.
    result_bidir = bidirectional_a_star_search(puzzle) 
    results.append(result_bidir)
    print("Bidirectional A* Done.")

    print("Running MCTS...")
    # MCTS might be slow and might not find the optimal path or even a path
    # depending on iterations and simulation depth.
    result_mcts = mcts_search(puzzle, iterations=5000, max_depth=60) # Adjust iterations/depth
    results.append(result_mcts)
    print("MCTS Done.")

    # --- Print Results ---
    print("\n" + "=" * 10 + " Search Results " + "=" * 10)
    for res in results:
        print(f"\nAlgorithm: {res['algorithm']}")
        if 'iterations' in res: print(f"Iterations: {res['iterations']}")
        print(f"Time Taken: {res['time']:.4f} seconds")
        print(f"Nodes Expanded/Explored: {res['nodes_expanded']}")
        print(f"Path Found: {'Yes' if res['path'] else 'No'}")
        if res['path']:
            print(f"Path Cost: {res['cost']}")
            # print(f"Path Length: {len(res['path'])}") # Should be cost + 1
            print("Path:", res['path']) # Uncomment to see the full path states
        else:
             print("Path Cost: N/A")
             if 'best_next_state_estimate' in res:
                   print(f"MCTS Best Next State Estimate: {res['best_next_state_estimate']}")