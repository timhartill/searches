import heapq
import math
import random
import time
import collections
from abc import ABC, abstractmethod

# --- Generic Problem Interface (Conceptual via Duck Typing) ---
# Search algorithms will expect objects with these methods:
# - initial_state() -> state
# - goal_state() -> state
# - is_goal(state) -> bool
# - get_neighbors(state) -> list[state]
# - heuristic(state) -> number (used by A*, Bidirectional, MCTS reward potentially)
# - get_cost(state1, state2) -> number (Assuming cost is 1 for these problems)
# - __str__() -> string (For nice printing)

# --- Sliding Tile Puzzle Implementation ---

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
            # Default goal for n*n-1 puzzle
            sorted_list = list(range(1, self.n * self.n)) + [0]
            self.goal_state_tuple = tuple(sorted_list)

        # Pre-compute goal positions for faster heuristic calculation
        self._goal_positions = {tile: i for i, tile in enumerate(self.goal_state_tuple)}
        self._str_repr = f"SlidingTile-{self.n}x{self.n}"

    def initial_state(self):
        return self.initial_state_tuple

    def goal_state(self):
        return self.goal_state_tuple

    def is_goal(self, state):
        """Checks if the state is the goal state."""
        return state == self.goal_state_tuple

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

    def heuristic(self, state):
        """Calculates the Manhattan distance heuristic."""
        distance = 0
        for i, tile in enumerate(state):
            if tile != 0: # Ignore the blank tile
                current_row, current_col = divmod(i, self.n)
                goal_index = self._goal_positions.get(tile)
                if goal_index is None: # Should not happen if state is valid
                     continue
                goal_row, goal_col = divmod(goal_index, self.n)
                distance += abs(current_row - goal_row) + abs(current_col - goal_col)
        return distance

    def get_cost(self, state1, state2):
        """Cost of moving between adjacent states is always 1."""
        return 1

    def __str__(self):
        return self._str_repr

# --- Pancake Sorting Problem Implementation ---

class PancakeProblem:
    """Implements the Pancake Sorting problem interface."""

    def __init__(self, initial_state, goal_state=None):
        self.initial_state_tuple = tuple(initial_state)
        self.n = len(initial_state)
        if goal_state:
             self.goal_state_tuple = tuple(goal_state)
        else:
             # Default goal is sorted stack
             self.goal_state_tuple = tuple(sorted(initial_state))

        self._str_repr = f"Pancake-{self.n}"


    def initial_state(self):
        return self.initial_state_tuple

    def goal_state(self):
        return self.goal_state_tuple

    def is_goal(self, state):
        """Checks if the state is the goal state."""
        return state == self.goal_state_tuple

    def get_neighbors(self, state):
        """Generates neighbor states by flipping the top k pancakes."""
        neighbors = []
        # Flipping top k pancakes (k=2 to N)
        for k in range(2, self.n + 1):
            # Slice the top k, reverse it, and concatenate with the rest
            flipped_part = state[:k][::-1]
            rest_part = state[k:]
            neighbors.append(flipped_part + rest_part)
        return neighbors

    def heuristic(self, state):
        """Calculates the Symmetrical Gap Heuristic."""
        # The number of pairs of adjacent pancakes |p_i - p_{i+1}| > 1
        # This indicates a 'break' in the desired sorted sequence.
        # Each flip can resolve at most one such gap.
        gaps = 0
        for i in range(self.n - 1):
            if abs(state[i] - state[i+1]) > 1:
                gaps += 1
        
        # Also consider if the largest pancake is not at the bottom 
        # (implicit gap with the "plate"). Some definitions include this.
        # Let's stick to the standard adjacent gap check for now.
        # If the largest pancake N is not in the last position, it needs at least one flip.
        # A common variant adds 1 if state[n-1] != n (where n is the largest pancake value)
        if self.n > 0 and state[-1] != max(state): # Assuming pancakes are 1 to N or similar range
            # Check if the largest pancake (value N if 1..N) is at the bottom
             if state[-1] != self.n: # Assuming pancakes are 1..N
                   pass # Standard gap doesn't always add this explicitly, let's omit for now
                   # gaps += 1 # Uncomment this line for a potentially stronger variant

        return gaps


    def get_cost(self, state1, state2):
        """Cost of each flip is 1."""
        return 1

    def __str__(self):
        return self._str_repr


# --- Path Reconstruction (Remains the same) ---

def reconstruct_path(came_from, start_state, goal_state):
    """Reconstructs the path from start to goal."""
    path = []
    current = goal_state
    # Ensure start_state is hashable if used as dict key directly
    start_node = start_state # Assuming states are hashable tuples
    
    if current == start_node:
        return [start_node]

    while current != start_node:
        path.append(current)
        parent = came_from.get(current)
        if parent is None:
             # This can happen if goal was reached but not via came_from chain?
             # Or if start/goal are same and loop condition was weird.
             print(f"Error: State {current} not found in came_from map during reconstruction, or loop error.")
             print(f"Came from keys: {list(came_from.keys())[:10]}...") # Debug
             print(f"Start state: {start_node}")
             # If current is start_node, loop shouldn't run. If goal != start, parent must exist.
             # Check if start node was added correctly.
             if start_node not in came_from and start_node != goal_state :
                  print("Start node itself is not in came_from (unless it's the goal).")

             # Check if goal node is reachable *at all* based on came_from keys
             if goal_state not in came_from and goal_state != start_node:
                  print(f"Goal state {goal_state} is not even a key in came_from.")

             # Let's assume the start node should always be the root with parent None.
             # If path reconstruction fails, maybe goal wasn't truly reached via the recorded path.
             return None # Indicate failure

        current = parent
        # Safety break
        if len(path) > 10000: # Arbitrary limit to prevent infinite loops
            print("Error: Path reconstruction exceeded limit.")
            return None

    path.append(start_node)
    return path[::-1] # Return path from start to goal

def reconstruct_bidirectional_path(came_from_fwd, came_from_bwd, start_state, goal_state, meeting_node):
    """Reconstructs path for bidirectional search."""
    # Path 1: start -> ... -> came_from_fwd[meeting_node] -> meeting_node
    path1 = []
    curr = meeting_node
    while curr is not None: # Trace back using forward parents
        path1.append(curr)
        curr = came_from_fwd.get(curr) # Use .get for safety if start has no parent entry
    path1.reverse() # Now path1 is start -> ... -> meeting_node

    # Path 2: meeting_node -> came_from_bwd[meeting_node] -> ... -> goal
    # The came_from_bwd stores parents *towards* the goal state in the backward search tree
    # So reconstructing from meeting_node using came_from_bwd traces *away* from goal.
    # We need the path segment from the *successor* of meeting_node (in backward path) to goal.
    
    path2 = []
    # Start from the node *after* meeting_node in the path towards the goal
    # This parent is stored in came_from_bwd[meeting_node]
    curr = came_from_bwd.get(meeting_node)
    while curr is not None: # Trace back using backward parents (away from meeting node, towards goal)
         path2.append(curr)
         curr = came_from_bwd.get(curr)
    # path2 now contains [parent_of_meeting_node_in_bwd_search, ..., goal_state]

    # Combine path1 (start...meeting) and path2 (node_after_meeting...goal)
    return path1 + path2


# --- Search Algorithms (Updated for Generic Problem) ---

def dijkstra_search(problem):
    """Performs Dijkstra's search (Uniform Cost Search) on a generic problem."""
    start_time = time.time()
    start_node = problem.initial_state()
    goal_node = problem.goal_state()

    # Priority queue stores (cost, state)
    frontier = [(0, start_node)]
    heapq.heapify(frontier)

    came_from = {start_node: None}
    cost_so_far = {start_node: 0}
    nodes_expanded = 0

    while frontier:
        current_cost, current_state = heapq.heappop(frontier)

        # Optimization: If we found a shorter path already, skip
        # Use get() with a default of infinity
        if current_cost > cost_so_far.get(current_state, float('inf')):
             continue

        nodes_expanded += 1 # Count node expansion accurately here

        if problem.is_goal(current_state):
            end_time = time.time()
            path = reconstruct_path(came_from, start_node, current_state)
            return {
                "path": path,
                "cost": current_cost,
                "nodes_expanded": nodes_expanded,
                "time": end_time - start_time,
                "algorithm": "Dijkstra"
            }

        for neighbor_state in problem.get_neighbors(current_state):
            new_cost = current_cost + problem.get_cost(current_state, neighbor_state) # Use problem's cost
            if new_cost < cost_so_far.get(neighbor_state, float('inf')):
                cost_so_far[neighbor_state] = new_cost
                priority = new_cost # Dijkstra uses only g(n)
                heapq.heappush(frontier, (priority, neighbor_state))
                came_from[neighbor_state] = current_state

    end_time = time.time()
    # Goal not found
    return {"path": None, "cost": -1, "nodes_expanded": nodes_expanded, "time": end_time - start_time, "algorithm": "Dijkstra"}


def a_star_search(problem):
    """Performs A* search on a generic problem."""
    start_time = time.time()
    start_node = problem.initial_state()
    goal_node = problem.goal_state()

    # Priority queue stores (f_score, state)
    h_initial = problem.heuristic(start_node)
    frontier = [(h_initial, start_node)] # f = g + h = 0 + h
    heapq.heapify(frontier)

    came_from = {start_node: None}
    g_score = {start_node: 0} # Cost from start
    nodes_expanded = 0

    while frontier:
        current_f_score, current_state = heapq.heappop(frontier)

        # Check if already found a better path to this state
        if current_state in g_score and g_score[current_state] < (current_f_score - problem.heuristic(current_state)):
             # This check isn't perfect, as heuristic might change f-score order.
             # A better check might involve a 'closed' set or checking g_score directly.
             # Let's rely on the check *before adding* to frontier for optimality.
             # If we pop a node, we consider it the optimal path found *so far*.
             pass # Keep processing

        nodes_expanded += 1

        if problem.is_goal(current_state):
            end_time = time.time()
            path = reconstruct_path(came_from, start_node, current_state)
            # Ensure g_score exists for goal if reached
            final_g_score = g_score.get(current_state, -1) 
            if path and final_g_score == -1: # Should not happen if path exists
                final_g_score = len(path) -1 # Estimate cost from path len
                
            return {
                "path": path,
                "cost": final_g_score,
                "nodes_expanded": nodes_expanded,
                "time": end_time - start_time,
                "algorithm": "A*"
            }

        # Explore neighbors
        current_g_score = g_score.get(current_state, float('inf')) # Should exist if popped

        for neighbor_state in problem.get_neighbors(current_state):
            cost = problem.get_cost(current_state, neighbor_state)
            tentative_g_score = current_g_score + cost

            if tentative_g_score < g_score.get(neighbor_state, float('inf')):
                # This path to neighbor is better. Record it.
                came_from[neighbor_state] = current_state
                g_score[neighbor_state] = tentative_g_score
                h_score = problem.heuristic(neighbor_state)
                f_score = tentative_g_score + h_score
                heapq.heappush(frontier, (f_score, neighbor_state))

    end_time = time.time()
    return {"path": None, "cost": -1, "nodes_expanded": nodes_expanded, "time": end_time - start_time, "algorithm": "A*"} # Goal not found


def bidirectional_a_star_search(problem):
    """Performs Bidirectional A* search on a generic problem using closed sets."""
    start_time = time.time()
    start_node = problem.initial_state()
    goal_node = problem.goal_state()

    if problem.is_goal(start_node):
        return {"path": [start_node], "cost": 0, "nodes_expanded": 0, "time": 0, "algorithm": "Bidirectional A*"}

    # --- Forward Search Structures ---
    h_start = problem.heuristic(start_node)
    frontier_fwd = [(h_start, start_node)] # (f = g+h, state)
    came_from_fwd = {start_node: None}
    g_score_fwd = {start_node: 0}
    closed_fwd = set() # Track expanded nodes in forward search

    # --- Backward Search Structures ---
    h_goal = problem.heuristic(goal_node) # Should be 0 if heuristic consistent
    frontier_bwd = [(h_goal, goal_node)] # (f = g_bwd + h(n->goal), state)
    came_from_bwd = {goal_node: None}
    g_score_bwd = {goal_node: 0} # Cost FROM GOAL
    closed_bwd = set() # Track expanded nodes in backward search

    nodes_expanded = 0
    best_path_cost = float('inf') # Upper bound U
    meeting_node = None

    # --- Main Loop ---
    while frontier_fwd and frontier_bwd:

        # --- Potential Termination Check (can be refined) ---
        # Check if fronts potentially crossed optimal path cost
        if meeting_node is not None:
             if frontier_fwd and frontier_bwd:
                  f_min_fwd, _ = frontier_fwd[0]
                  f_min_bwd, _ = frontier_bwd[0]
                  # Heuristic-based check (requires consistent heuristic)
                  # if f_min_fwd >= best_path_cost or f_min_bwd >= best_path_cost:
                  #      break
                  # A potentially safer check based on g-scores (if available)
                  # g_min_fwd = # need state associated with f_min_fwd
                  # g_min_bwd = # need state associated with f_min_bwd
                  # if g_min_fwd + g_min_bwd >= best_path_cost: # Estimate minimum possible path cost through frontiers
                  #      break 
             # Simple check: if both queues only contain states worse than best_path_cost? Less efficient.
             pass # Rely on pruning and finding meeting node for this implementation


        # --- Forward Step ---
        if frontier_fwd:
            current_f_fwd, current_state_fwd = heapq.heappop(frontier_fwd)

            # Check if already expanded via an optimal path
            if current_state_fwd in closed_fwd:
                continue
            
            # Check if current path cost is already too high (pruning)
            # Note: g_score might not be fully up-to-date here if duplicates existed, but useful
            if g_score_fwd.get(current_state_fwd, float('inf')) >= best_path_cost :
                 continue
                 
            closed_fwd.add(current_state_fwd) # Mark as expanded
            nodes_expanded += 1


            # Check for meeting point *after* confirming not closed
            if current_state_fwd in g_score_bwd: # Found by backward search (check if backward search *reached* it)
                current_path_cost = g_score_fwd[current_state_fwd] + g_score_bwd[current_state_fwd]
                if current_path_cost < best_path_cost:
                    best_path_cost = current_path_cost
                    meeting_node = current_state_fwd
                    # print(f"  Fwd Step: New best path potential: cost {best_path_cost} via {meeting_node}") # Debug


            # Expand forward neighbors
            for neighbor_state in problem.get_neighbors(current_state_fwd):
                # Ensure neighbor hasn't already been closed/expanded in forward search
                if neighbor_state in closed_fwd:
                     continue
                     
                cost = problem.get_cost(current_state_fwd, neighbor_state)
                tentative_g_score = g_score_fwd[current_state_fwd] + cost

                if tentative_g_score < g_score_fwd.get(neighbor_state, float('inf')):
                    came_from_fwd[neighbor_state] = current_state_fwd
                    g_score_fwd[neighbor_state] = tentative_g_score
                    h_score = problem.heuristic(neighbor_state)
                    f_score = tentative_g_score + h_score

                    # Add to frontier only if potentially better than best path found
                    # And if not already closed (redundant check here, but safe)
                    if f_score < best_path_cost:
                         heapq.heappush(frontier_fwd, (f_score, neighbor_state))


        # --- Backward Step ---
        if frontier_bwd:
            current_f_bwd, current_state_bwd = heapq.heappop(frontier_bwd)

            if current_state_bwd in closed_bwd:
                continue

            if g_score_bwd.get(current_state_bwd, float('inf')) >= best_path_cost:
                  continue
                  
            closed_bwd.add(current_state_bwd) # Mark as expanded in backward search
            nodes_expanded += 1


            # Check for meeting point
            if current_state_bwd in g_score_fwd: # Found by forward search
                current_path_cost = g_score_fwd[current_state_bwd] + g_score_bwd[current_state_bwd]
                if current_path_cost < best_path_cost:
                    best_path_cost = current_path_cost
                    meeting_node = current_state_bwd
                    # print(f"  Bwd Step: New best path potential: cost {best_path_cost} via {meeting_node}") # Debug

            # Expand backward neighbors
            for neighbor_state in problem.get_neighbors(current_state_bwd):
                if neighbor_state in closed_bwd:
                    continue
                    
                cost = problem.get_cost(current_state_bwd, neighbor_state) # Cost is symmetric here
                tentative_g_score = g_score_bwd[current_state_bwd] + cost # g_bwd grows from goal

                if tentative_g_score < g_score_bwd.get(neighbor_state, float('inf')):
                    came_from_bwd[neighbor_state] = current_state_bwd # Parent towards goal
                    g_score_bwd[neighbor_state] = tentative_g_score

                    h_score = problem.heuristic(neighbor_state) # h(n->goal)
                    f_score = tentative_g_score + h_score

                    if f_score < best_path_cost:
                         heapq.heappush(frontier_bwd, (f_score, neighbor_state))

    # --- End of Loop ---
    end_time = time.time()

    if meeting_node:
        path = reconstruct_bidirectional_path(came_from_fwd, came_from_bwd, start_node, goal_node, meeting_node)
        
        # Verify path cost if possible
        actual_cost = -1
        if path:
             actual_cost = len(path) - 1 
             # If path reconstruction worked, its length should match best_path_cost
             final_cost = actual_cost # Use length from reconstructed path
             if final_cost != best_path_cost:
                  print(f"Warning: Bidirectional path cost mismatch! Path len={final_cost}, best_path_cost={best_path_cost}")
                  # Could indicate issue in reconstruction or termination logic letting suboptimal path be recorded
                  final_cost = best_path_cost # Trust the calculated cost potentially
        else:
             final_cost = -1 # Path reconstruction failed


        return {
            "path": path,
            "cost": final_cost if path else -1, # Report cost only if path found
            "nodes_expanded": nodes_expanded,
            "time": end_time - start_time,
            "algorithm": "Bidirectional A*"
        }
    else:
        # No meeting point found
        return {"path": None, "cost": -1, "nodes_expanded": nodes_expanded, "time": end_time - start_time, "algorithm": "Bidirectional A*"}

# --- Monte Carlo Tree Search (Updated for Generic Problem) ---

class MCTSNode:
    def __init__(self, state, parent=None, problem=None): # Takes generic problem
        self.state = state
        self.parent = parent
        self.problem = problem # Store the problem instance
        self.children = []
        self.visits = 0
        self.value = 0.0  # Can be avg reward, total reward, etc. Let's use total.
        # Use problem methods for neighbors and goal check
        self._untried_actions = problem.get_neighbors(state) if problem else []
        random.shuffle(self._untried_actions)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, exploration_weight=1.41): # UCB1
        if not self.children: return None
        if self.visits == 0: return random.choice(self.children) # Avoid log(0)

        log_total_visits = math.log(self.visits)

        def ucb1(node):
            if node.visits == 0:
                # Encourage exploration of unvisited nodes strongly
                return float('inf')
            # Average value (reward / visits)
            exploitation = node.value / node.visits
            exploration = exploration_weight * math.sqrt(log_total_visits / node.visits)
            return exploitation + exploration

        # Add small random noise to break ties consistently
        best_node = max(self.children, key=lambda node: ucb1(node) + random.uniform(0, 1e-6))
        return best_node


    def expand(self):
        if not self._untried_actions: return None # Cannot expand
        action_state = self._untried_actions.pop()
        # Pass the problem instance to the child node
        child_node = MCTSNode(action_state, parent=self, problem=self.problem)
        self.children.append(child_node)
        return child_node

    def is_terminal(self):
         # Use the problem's goal check
         return self.problem.is_goal(self.state) if self.problem else False


def mcts_search(problem, iterations=1000, max_depth=50):
    """Performs Monte Carlo Tree Search on a generic problem."""
    start_time = time.time()
    start_node = problem.initial_state()
    # Pass the problem instance to the root node
    root = MCTSNode(state=start_node, problem=problem)

    if root.is_terminal():
         return {"path": [root.state], "cost": 0, "nodes_expanded": 1, "time": time.time()-start_time, "algorithm": "MCTS", "iterations": 0}

    for i in range(iterations):
        node = root
        path_to_leaf = [node]

        # 1. Selection: Traverse down the tree using UCB1
        # Loop while the current node is not terminal AND fully expanded
        while not node.is_terminal() and node.is_fully_expanded():
            # Find the best child to move to
            # Use a temporary variable name to avoid confusion if loop breaks
            selected_child = node.best_child() 
            
            if selected_child is None: 
                 # This case *shouldn't* really happen if node is fully_expanded
                 # and has children, unless best_child() logic or tree structure is flawed.
                 # Break defensively to avoid errors in this iteration.
                 print(f"Warning: MCTS Selection encountered None best_child for fully expanded node {node.state}. Stopping iteration.")
                 break # Stop traversing down this path for this iteration

            # If a valid child was selected, update node and continue selection
            node = selected_child 
            path_to_leaf.append(node)
        
        # *** The problematic 'if best_child is None...' check is REMOVED from here ***
        # After the loop, 'node' holds the leaf node selected by the policy.
        # This node is either terminal OR it's not fully expanded.
            
        # 2. Expansion: If the selected node is not terminal and not fully expanded
        # We attempt to expand this 'node'
        expanded_node = None # Keep track if expansion happened
        if not node.is_terminal() and not node.is_fully_expanded():
            expanded_node = node.expand() # Attempt to expand the selected node
            if expanded_node: 
                 # If expansion was successful, the simulation starts from the new child
                 node = expanded_node 
                 path_to_leaf.append(node)
            # If expansion failed (e.g., node became fully expanded concurrently? Rare. Or expand() bug?), 
            # simulation will start from the 'node' selected before the expand attempt.
            
        # 3. Simulation (Rollout): Start from 'node' 
        # ('node' is either the leaf selected by Selection or the newly expanded node)
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
            # Assuming cost 1 per step in simulation for reward calculation
            sim_path_cost += 1 # Use problem.get_cost if needed

        # Determine reward
        if problem.is_goal(current_state):
            reward = 1000.0 / (1 + sim_path_cost) 
        else:
            reward = -problem.heuristic(current_state) 
        
        # 4. Backpropagation: Update statistics up the tree
        for node_in_path in reversed(path_to_leaf):
            node_in_path.visits += 1
            node_in_path.value += reward # Accumulate reward

    # --- End of Iterations ---
    end_time = time.time()

    # --- Extracting Path from MCTS Tree (remains the same) ---
    # ... (BFS search on tree to find best path) ...
    
    # (Code for BFS on tree and returning results is unchanged)
    # ... (rest of the function) ...

    # --- Placeholder for the rest of the function's return logic ---
    # (This part needs to be included from your previous working code)
    goal_node_in_tree = None
    min_cost_in_tree = float('inf')
    queue = collections.deque([(root, 0)]) # Node, cost_from_root
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
                  while temp is not None:
                      path.append(temp.state)
                      temp = temp.parent
                  best_path_found = path[::-1]

         for child in current_node.children:
             # Use problem.get_cost here if costs aren't uniform
             cost_step = 1 
             # cost_step = problem.get_cost(current_node.state, child.state)
             new_cost = current_cost + cost_step
             if child.state not in visited_in_tree or new_cost < visited_in_tree[child.state]:
                  visited_in_tree[child.state] = new_cost
                  queue.append((child, new_cost))

    if best_path_found:
        return {
            "path": best_path_found,
            "cost": min_cost_in_tree,
            "nodes_expanded": nodes_explored_in_tree, 
            "time": end_time - start_time,
            "algorithm": "MCTS",
            "iterations": iterations,
            "tree_root_visits": root.visits 
        }
    else:
         best_first_move_node = root.best_child(exploration_weight=0) 
         return {
            "path": None, 
            "cost": -1,
            "nodes_expanded": nodes_explored_in_tree,
            "time": end_time - start_time,
            "algorithm": "MCTS",
            "iterations": iterations,
            "best_next_state_estimate": best_first_move_node.state if best_first_move_node else None,
            "tree_root_visits": root.visits
        }

# --- Main Execution Logic ---

if __name__ == "__main__":

    # input params
    iterations = 100000   # for MCTS
    max_depth = 100      # for MCTS


    # --- Define Problems ---

    # Sliding Tile (8-Puzzle)
    # tile_initial = [1, 2, 3, 0, 4, 6, 7, 5, 8] # Medium
    tile_initial = [8, 6, 7, 2, 5, 4, 3, 0, 1] # Harder
    tile_goal    = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    sliding_tile_problem = SlidingTileProblem(initial_state=tile_initial, goal_state=tile_goal)

    # Pancake Sorting
    # pancake_initial = (3, 1, 2) # Simple 3-pancake
    # pancake_initial = (1, 6, 2, 5, 3, 4) # 6-pancake example
    pancake_initial = (8, 3, 5, 1, 6, 4, 2, 7) # 8-pancake example
    pancake_problem = PancakeProblem(initial_state=pancake_initial) # Goal auto-detected as sorted

    problems_to_solve = [
        sliding_tile_problem,
        pancake_problem
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
            
            # Special handling for MCTS iterations if needed
            if algo_name == 'mcts_search':
                 # Lower iterations for potentially faster testing, increase for better results
                 result = algorithm_func(problem, iterations=iterations, max_depth=max_depth) 
            else:
                 result = algorithm_func(problem)
            
            print(f"{algo_name} Done.")
            result['problem'] = str(problem) # Add problem identifier to result
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
                # print(f"Path Length: {len(res['path'])}") # Should be cost + 1
                # print("Path States:", res['path']) # Uncomment to see states
            else:
                 print("Path Cost: N/A")
                 if 'best_next_state_estimate' in res:
                       print(f"MCTS Best Next State Estimate: {res['best_next_state_estimate']}")
        print("=" * (20 + len(str(problem)) + 14))


    print("\n" + "*"*15 + " Overall Summary " + "*"*15)
    # You could add more summary logic here if needed, e.g., comparing times/nodes across problems/algorithms
    for res in all_results:
         print(f"- Problem: {res['problem']}, Algorithm: {res['algorithm']}, Time: {res['time']:.4f}s, Nodes: {res['nodes_expanded']}, Cost: {res['cost']}")