"""
Dijkstra/Uniform cost (g), Best first (h) ,A* f=g+h, Bidirectional A*, MCTS for Sliding Tile, Pancake, Towers of Hanoi
- This code implements various search algorithms for solving the Sliding Tile, Pancake Sorting, and Towers of Hanoi problems.

Mostly generated from Gemini 2.5, with some modifications and fixes.
"""
import heapq
import math
import random
import time
import collections
from abc import ABC, abstractmethod # Can be used for formal interface if desired
import traceback # For error reporting

# --- Path Reconstruction Functions ---

def reconstruct_path(came_from, start_state, goal_state):
    """Reconstructs the path from start to goal."""
    path = []
    current = goal_state
    start_node = start_state 
    if current == start_node: return [start_node]
    
    limit = 100000 # Generic large limit
    if isinstance(start_state, tuple) and hasattr(start_state, '__len__'): 
        limit = max(limit, 2**(len(start_state) + 6)) 

    count = 0
    while current != start_node:
        path.append(current)
        parent = came_from.get(current)
        if parent is None:
             current_str = str(current)[:100] + ('...' if len(str(current)) > 100 else '')
             print(f"Error: State {current_str} not found in came_from map during reconstruction.")
             return None 
        current = parent
        count += 1
        if count > limit: 
            print(f"Error: Path reconstruction exceeded limit ({limit}).")
            return None
            
    path.append(start_node)
    return path[::-1] 
    
def reconstruct_bidirectional_path(came_from_fwd, came_from_bwd, start_state, goal_state, meeting_node):
    """Reconstructs path for bidirectional search."""
    path1 = []
    curr = meeting_node
    
    limit = 100000 
    if isinstance(start_state, tuple) and hasattr(start_state, '__len__'):
         limit = max(limit, 2**(len(start_state) + 6))

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


# --- Problem Class Definitions ---

# --- SlidingTileProblem (Corrected Formatting & use_variable_costs flag) ---
class SlidingTileProblem:
    """
    Implements the Sliding Tile puzzle problem interface.
    Can use uniform cost (1) or variable cost (value of tile moved).
    """
    def __init__(self, initial_state, goal_state=None, 
                 use_variable_costs=False, make_heuristic_inadmissable=False,
                 degradation=0):
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
        self.use_variable_costs = use_variable_costs
        self.optimality_guaranteed = True
        self.make_heuristic_inadmissable = make_heuristic_inadmissable
        if make_heuristic_inadmissable:
            self.h_multiplier = len(initial_state) * (degradation+10)
            self.optimality_guaranteed = False
        else:
            self.h_multiplier = 1    
        self.degradation = degradation    
        cost_type = "VarCost" if use_variable_costs else "UnitCost"
        self._str_repr = f"SlidingTile-{self.n}x{self.n}-{cost_type}-d{degradation}-a{not make_heuristic_inadmissable}"

    def initial_state(self): 
        return self.initial_state_tuple
        
    def goal_state(self): 
        return self.goal_state_tuple
        
    def is_goal(self, state): 
        return state == self.goal_state_tuple

    def get_neighbors(self, state):
        """Returns list of tuples: (neighbor_state, moved_tile_value) from state
        where moved_tile_value is the value (label) of the tile that was moved to the blank space.
        """
        neighbors = []
        blank_index = state.index(0)    # index of blank in state list 
        row, col = divmod(blank_index, self.n)
        moves = {'up': (row - 1, col), 'down': (row + 1, col), 'left': (row, col - 1), 'right': (row, col + 1)}
        for move_dir, (new_row, new_col) in moves.items():
            if 0 <= new_row < self.n and 0 <= new_col < self.n:  # if valid move
                new_blank_index = new_row * self.n + new_col  # index of blank if move this direction
                new_state_list = list(state)
                moved_tile_value = new_state_list[new_blank_index] 
                # swap blank with the tile in the new position:
                new_state_list[blank_index], new_state_list[new_blank_index] = new_state_list[new_blank_index], new_state_list[blank_index]
                neighbors.append((tuple(new_state_list), moved_tile_value)) 
        return neighbors 

    def get_cost(self, state1, state2, move_info=None):
        """
        Returns cost of move. If use_variable_costs is True, cost is the
        value of the tile moved (min 1). Otherwise, cost is 1.
        """
        if not self.use_variable_costs:
            return 1 
            
        if move_info is not None:
            moved_tile_value = move_info
            return max(1, moved_tile_value) 
        else: 
            blank1_idx, blank2_idx, moved_tile = -1, -1, -1
            # Ensure state1 and state2 are tuples/sequences before len()
            if not hasattr(state1, '__len__') or not hasattr(state2, '__len__') or len(state1) != len(state2):
                print(f"Warning: Invalid states for cost calculation fallback: {state1}, {state2}")
                return 1 # Fallback cost
            for i in range(len(state1)):
                if state1[i] == 0: blank1_idx = i
                if state2[i] == 0: blank2_idx = i
            if blank1_idx != -1 and blank2_idx != -1 :
                 moved_tile = state2[blank1_idx] 
                 return max(1, moved_tile)
            else:
                 print(f"Warning: Could not determine moved tile between {state1} and {state2}")
                 return 1 

    def heuristic(self, state):
        """
        Calculates the Manhattan distance heuristic (number of steps).
        NOTE: This heuristic counts steps (cost=1). If variable costs are used,
        its effectiveness will decrease but still admissable since var costs >= unit costs.
        """
        distance = 0
        multiplier = 1
        ignored_tiles = set(range(self.degradation + 1))
        for i, tile in enumerate(state):
            if tile not in ignored_tiles:
                current_pos = divmod(i, self.n)
                goal_idx = self._goal_positions.get(tile)
                if goal_idx is None: continue 
                goal_pos = divmod(goal_idx, self.n)
                if self.make_heuristic_inadmissable:
                    multiplier = i
                distance += multiplier * (abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1]))
        return distance * self.h_multiplier
        
    def __str__(self): 
        return self._str_repr

# --- PancakeProblem (Corrected Formatting & use_variable_costs flag) ---
class PancakeProblem:
    """
    Implements the Pancake Sorting problem interface.
    Can use uniform cost (1) or variable cost (k = pancakes flipped).
    """
    def __init__(self, initial_state, goal_state=None, 
                 use_variable_costs=False, make_heuristic_inadmissable=False,
                 degradation=0):
        self.initial_state_tuple=tuple(initial_state); self.n=len(initial_state)
        if goal_state: self.goal_state_tuple=tuple(goal_state) 
        else: self.goal_state_tuple=tuple(sorted(initial_state))
        self.use_variable_costs = use_variable_costs
        self.optimality_guaranteed = not use_variable_costs
        self.make_heuristic_inadmissable = make_heuristic_inadmissable
        if make_heuristic_inadmissable:
            self.h_multiplier = len(initial_state) * (degradation+10)
            self.optimality_guaranteed = False
        else:
            self.h_multiplier = 1    
        self.degradation = degradation
        cost_type = "VarCost" if use_variable_costs else "UnitCost"
        self._str_repr = f"Pancake-{self.n}-{cost_type}-d{degradation}-a{not make_heuristic_inadmissable}"
        
    def initial_state(self): 
        return self.initial_state_tuple
        
    def goal_state(self): 
        return self.goal_state_tuple
        
    def is_goal(self, state): 
        return state == self.goal_state_tuple

    def get_neighbors(self, state):
        """Returns list of tuples: (neighbor_state, k_flipped)"""
        neighbors = []
        for k in range(2, self.n + 1):
            if k > 1: 
                flipped_part = state[:k][::-1]
                rest_part = state[k:]
                neighbors.append((flipped_part + rest_part, k)) 
        return neighbors 

    def get_cost(self, state1, state2, move_info=None):
        """
        Returns cost of move. If use_variable_costs is True, cost is k 
        (number of pancakes flipped). Otherwise, cost is 1.
        """
        if not self.use_variable_costs:
            return 1
            
        if move_info is not None:
            k_flipped = move_info
            return k_flipped 
        else: 
            if not hasattr(state1, '__len__') or not hasattr(state2, '__len__') or len(state1) != len(state2): return 1 
            n=len(state1)
            if state1 == state2: return 0 
            
            diff_k = n 
            for k in range(n):
                 if state1[k] != state2[k]:
                      diff_k = k + 1 
                      break

            found_k = 1 
            for k_try in range(max(2, diff_k), n + 1):
                 if state1[:k_try][::-1] == state2[:k_try] and state1[k_try:] == state2[k_try:]:
                      found_k = k_try; break
                 elif k_try == n and state1[:k_try][::-1] == state2[:k_try]: 
                      found_k = k_try; break

            # Verify diff_k if loop didn't find larger k
            if found_k == 1 and diff_k >= 2:
                if state1[:diff_k][::-1] == state2[:diff_k] and state1[diff_k:] == state2[diff_k:]:
                    found_k = diff_k
                else:
                    print(f"Warning: Could not determine k flipped between {state1} and {state2}")
            elif found_k == 1 and diff_k < 2: # Should only happen if states are same, handled above
                 print(f"Warning: Unexpected state comparison in pancake get_cost fallback.")

            return found_k


    def heuristic(self, state):
        """
        Calculates the Gap Heuristic (number of adjacent non-consecutive pairs).
        NOTE: This counts number of "breaks". If variable costs (cost=k) are used,
        this heuristic likely becomes non-admissible as one flip (cost k) can fix
        at most 2 gaps.
        """
        return sum(1 for i in range(self.n-1) if abs(state[i]-state[i+1]) > 1) * self.h_multiplier
        
    def __str__(self): 
        return self._str_repr

# --- TowersOfHanoiProblem (Corrected Formatting) ---
class TowersOfHanoiProblem:
    """Implements the Towers of Hanoi problem interface. Cost is always 1."""
    def __init__(self, num_disks, initial_peg='A', target_peg='C', 
                 make_heuristic_inadmissable=False, degradation=0): 
        self.use_variable_costs = False # Cost is always 1
        self.optimality_guaranteed = not self.use_variable_costs
        self.make_heuristic_inadmissable = make_heuristic_inadmissable
        if make_heuristic_inadmissable:
            self.h_multiplier = num_disks * (degradation+10)
            self.optimality_guaranteed = False
        else:
            self.h_multiplier = 1    
        self.degradation = degradation    
        assert num_disks >= 1, "Number of disks must be at least 1."
        self.num_disks = num_disks
        self.pegs=['A','B','C']
        assert initial_peg in self.pegs and target_peg in self.pegs and initial_peg != target_peg, \
            f"Invalid initial or target peg. Must be one of {self.pegs} and different."
        self.initial_peg = initial_peg
        self.target_peg = target_peg
        self.aux_peg=next(p for p in self.pegs if p!=initial_peg and p!=target_peg)
        self._initial_state=tuple([initial_peg]*num_disks)
        self._goal_state=tuple([target_peg]*num_disks)
        self._str_repr=f"TowersOfHanoi-{num_disks}-d{degradation}-a{not make_heuristic_inadmissable}"

    def initial_state(self): 
        return self._initial_state
        
    def goal_state(self): 
        return self._goal_state
        
    def is_goal(self, state): 
        return state == self._goal_state

    def _get_peg_tops(self, state):
        """Helper to find the smallest (topmost) disk index on each peg."""
        peg_tops = {p: None for p in self.pegs}
        # Use infinity to correctly find the minimum disk index
        peg_top_disk_index = {p: float('inf') for p in self.pegs} 
        
        for disk_index, peg in enumerate(state):
             if disk_index < peg_top_disk_index[peg]:
                  peg_top_disk_index[peg] = disk_index
                  
        # Update peg_tops with the actual disk index found, or None
        for peg in self.pegs:
             if peg_top_disk_index[peg] != float('inf'):
                  peg_tops[peg] = peg_top_disk_index[peg] 
                  
        return peg_tops

    def get_neighbors(self, state): 
        """Returns list of tuples: (neighbor_state, cost=1)"""
        nbs=[]; pts=self._get_peg_tops(state); 
        for sp in self.pegs:
            dtm = pts[sp] # Disk To Move index (top disk on source)
            if dtm is not None: # If source peg is not empty
                for dp in self.pegs: # Destination Peg
                    if sp != dp: # Cannot move to same peg
                        tdod = pts[dp] # Top Disk On Destination index
                        # Check move validity: dest empty OR moving disk < disk on dest
                        if tdod is None or dtm < tdod:
                            nsl = list(state)
                            nsl[dtm] = dp # Move disk dtm to peg dp
                            nbs.append((tuple(nsl), 1)) # Append (new_state, cost)
        return nbs

    def heuristic(self, state): 
        """Calculates the standard admissible heuristic for Towers of Hanoi."""
        h=0 
        ctp=self.target_peg # current target peg for disk k
        for k in range(self.num_disks-1,-1,-1): # Iterate largest disk (N-1) down to smallest (0)
            if state[k] != ctp: 
                # If disk k is not where it should be relative to disk k+1 (or final target)
                # it and all smaller disks must move. Lower bound cost is 2^k.
                h += pow(2,k)
                # The new target for disk k-1 becomes the 'third' peg 
                # (neither where k is, nor where k should have been)
                ctp = next(p for p in self.pegs if p!=state[k] and p!=ctp)
            # else: disk k is on the correct peg (ctp), so target for k-1 remains ctp.
        return h * self.h_multiplier

    def get_cost(self, state1, state2, move_info=None): 
        """Cost is always 1 for Towers of Hanoi."""
        return 1
        
    def __str__(self): 
        return self._str_repr


# --- Generic Unidirectional Search Function ---
def generic_search(problem, priority_key='f'):
    """
    Performs a generic best-first search using a closed set.
    Priority can be based on 'g', 'h', or 'f' = g+h. Handles variable costs.
    """
    if priority_key not in ['g', 'h', 'f']: raise ValueError("priority_key must be 'g', 'h', or 'f'")
    algo_name_map = {'g': "Uniform Cost", 'h': "Greedy Best-First", 'f': "A*"}
    algorithm_name = algo_name_map[priority_key] + " (Generic)"
    optimality_guaranteed = (priority_key == 'g') or (priority_key=='f' and problem.optimality_guaranteed)

    start_time = time.time(); start_node = problem.initial_state()
    h_initial = problem.heuristic(start_node) if priority_key in ['h', 'f'] else 0
    initial_g = 0
    if priority_key == 'g': initial_priority = initial_g
    elif priority_key == 'h': initial_priority = h_initial
    else: initial_priority = initial_g + h_initial # 'f'

    frontier = [(initial_priority, start_node)] 
    heapq.heapify(frontier)
    came_from = {start_node: None} 
    g_score = {start_node: initial_g}
    closed_set = set()
    nodes_expanded = 0

    while frontier:
        current_priority, current_state = heapq.heappop(frontier)
        
        # Optimization: If current_state's g_score is worse than recorded, skip
        # This can happen with duplicate states in the queue with different priorities
        #if current_state in g_score and g_score[current_state] < current_priority - (problem.heuristic(current_state) if priority_key != 'g' else 0):
             # Check g_score derived from priority vs stored g_score if applicable
             # Let's rely on the closed set check primarily
             #pass 

        nodes_expanded += 1 
        if current_state in closed_set: continue
        closed_set.add(current_state) # Add after popping and checking

        if problem.is_goal(current_state):
            end_time = time.time()
            path = reconstruct_path(came_from, start_node, current_state)
            final_g_score = g_score.get(current_state)
            return {"path": path, "cost": final_g_score, "nodes_expanded": nodes_expanded, "time": end_time - start_time, "algorithm": algorithm_name, "optimal": optimality_guaranteed }

        current_g_score = g_score.get(current_state)
        if current_g_score is None: continue # Should have g_score if reached here

        for neighbor_info in problem.get_neighbors(current_state):
            # Handle cases where get_neighbors might return just state or (state, move_info)
            if isinstance(neighbor_info, tuple) and len(neighbor_info) >= 1:
                 neighbor_state = neighbor_info[0]
                 move_info = neighbor_info[1] if len(neighbor_info) > 1 else None
            else:
                 neighbor_state = neighbor_info
                 move_info = None

            if neighbor_state in closed_set: continue

            cost = problem.get_cost(current_state, neighbor_state, move_info)
            tentative_g_score = current_g_score + cost

            if tentative_g_score < g_score.get(neighbor_state, float('inf')):
                came_from[neighbor_state] = current_state 
                g_score[neighbor_state] = tentative_g_score
                priority = tentative_g_score # Default for 'g'
                if priority_key in ['h', 'f']:
                    h_score = problem.heuristic(neighbor_state)
                    if priority_key == 'h': priority = h_score
                    elif priority_key == 'f': priority = tentative_g_score + h_score
                heapq.heappush(frontier, (priority, neighbor_state))

    end_time = time.time()
    return {"path": None, "cost": -1, "nodes_expanded": nodes_expanded, "time": end_time - start_time, "algorithm": algorithm_name, "optimal": False }


# --- Bidirectional A* (Updated for variable cost) ---
def bidirectional_a_star_search(problem):
    """Performs Bidirectional A* search. Handles variable costs."""
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
            _, current_state_fwd = heapq.heappop(frontier_fwd)
            if current_state_fwd in closed_fwd: continue
            current_g_fwd = g_score_fwd.get(current_state_fwd, float('inf'))
            if current_g_fwd >= best_path_cost : continue             
            closed_fwd.add(current_state_fwd); nodes_expanded += 1
            if current_state_fwd in g_score_bwd: 
                current_path_cost = current_g_fwd + g_score_bwd[current_state_fwd]
                if current_path_cost < best_path_cost: best_path_cost = current_path_cost; meeting_node = current_state_fwd
            
            for neighbor_info in problem.get_neighbors(current_state_fwd):
                neighbor_state, move_info = (neighbor_info if isinstance(neighbor_info, tuple) and len(neighbor_info)==2 else (neighbor_info, None))
                if neighbor_state in closed_fwd: continue
                cost = problem.get_cost(current_state_fwd, neighbor_state, move_info) 
                tentative_g_score = current_g_fwd + cost
                if tentative_g_score < g_score_fwd.get(neighbor_state, float('inf')):
                    came_from_fwd[neighbor_state] = current_state_fwd; g_score_fwd[neighbor_state] = tentative_g_score
                    h_score = problem.heuristic(neighbor_state); f_score = tentative_g_score + h_score
                    if f_score < best_path_cost: heapq.heappush(frontier_fwd, (f_score, neighbor_state))
        
        # --- Backward Step ---
        if frontier_bwd:
            _, current_state_bwd = heapq.heappop(frontier_bwd)
            if current_state_bwd in closed_bwd: continue
            current_g_bwd = g_score_bwd.get(current_state_bwd, float('inf'))
            if current_g_bwd + problem.heuristic(current_state_bwd) >= best_path_cost: continue 
            closed_bwd.add(current_state_bwd); nodes_expanded += 1
            if current_state_bwd in g_score_fwd: 
                current_path_cost = g_score_fwd[current_state_bwd] + current_g_bwd
                if current_path_cost < best_path_cost: best_path_cost = current_path_cost; meeting_node = current_state_bwd

            for neighbor_info in problem.get_neighbors(current_state_bwd):
                neighbor_state, move_info = (neighbor_info if isinstance(neighbor_info, tuple) and len(neighbor_info)==2 else (neighbor_info, None))
                if neighbor_state in closed_bwd: continue
                cost = problem.get_cost(current_state_bwd, neighbor_state, move_info) 
                tentative_g_score = current_g_bwd + cost 
                if tentative_g_score < g_score_bwd.get(neighbor_state, float('inf')):
                    came_from_bwd[neighbor_state] = current_state_bwd 
                    g_score_bwd[neighbor_state] = tentative_g_score
                    h_score = problem.heuristic(neighbor_state); f_score = tentative_g_score + h_score
                    if f_score < best_path_cost: heapq.heappush(frontier_bwd, (f_score, neighbor_state))
        
    end_time = time.time()
    if meeting_node:
        path = reconstruct_bidirectional_path(came_from_fwd, came_from_bwd, start_node, goal_node, meeting_node)
        final_cost = -1; recalculated_cost = 0; cost_mismatch = False
        if path:
             try:
                 for i in range(len(path) - 1):
                     recalculated_cost += problem.get_cost(path[i], path[i+1]) # Use fallback cost
                 final_cost = recalculated_cost
                 if abs(final_cost - best_path_cost) > 1e-6 : cost_mismatch = True
             except Exception as e:
                 print(f"Error recalculating bidirectional path cost: {e}"); final_cost = best_path_cost 
        if cost_mismatch: print(f"Warning: Bidirectional cost mismatch! PathRecalc={final_cost}, SearchCost={best_path_cost}")
        final_reported_cost = best_path_cost # Report cost found by search
        return {"path": path, "cost": final_reported_cost if path else -1, "nodes_expanded": nodes_expanded, "time": end_time - start_time, "algorithm": "Bidirectional A*"}
    else:
        return {"path": None, "cost": -1, "nodes_expanded": nodes_expanded, "time": end_time - start_time, "algorithm": "Bidirectional A*"}


# --- MCTSNode and mcts_search (Updated for variable cost) ---
class MCTSNode:
    def __init__(self, state, parent=None, problem=None): 
        self.state = state
        self.parent = parent    # parent MCTSNode
        self.problem = problem  # problem instance, can access problem methods like get_neighbors, heuristic
        self.children = []      # list of child MCTSNode
        self.visits = 0         # number of visits to this node
        self.value = 0.0        # current value of this node
        # self.heuristic = problem.heuristic(state) if problem else 0 # heuristic value of this node
        # Store neighbor states for expansion, original info might be needed for cost later if not recalculated
        self._untried_actions_info = problem.get_neighbors(state) if problem else []  # list of tuples of (neighbor states, 'moved' item = "action")
        self._untried_states = [info[0] if isinstance(info, tuple) else info for info in self._untried_actions_info] # list of neighbour states
        random.shuffle(self._untried_states)

    def best_child(self, exploration_weight=1.41): 
        """ Selects the best child node randomly if previously unvisited or deterministically based on UCB1 (Upper Confidence Bound) formula. """
        if not self.children: return None
        if self.visits <= 0: # Ensure visits is positive for log
             # If root or unvisited node, pick randomly or based on prior?
             # TODO: consider using heuristic for unvisited nodes either max or sampled
             return random.choice(self.children) if self.children else None 
        log_total_visits = math.log(self.visits)

        def ucb1(node): # Upper Confidence Bound 
            #TODO modify to include heuristic?
            if node.visits == 0: return float('inf') # Prioritize unvisited children [and/or use heuristic?]
            exploitation = node.value / node.visits  # higher value=higher exploitation score but the more visits the lower the exploitation score
            exploration = exploration_weight * math.sqrt(log_total_visits / node.visits) # sqrt(log(parent_visits) / curr_child_visits)  1 visit = 0, 2=0.59*1.41, 10=0.48, 100=0.21, 1000=0.08
            return exploitation + exploration

        # Add small random noise to break ties consistently
        #TODO: consider using heuristic for unvisited nodes either max or sampled
        best_node = max(self.children, key=lambda node: ucb1(node) + random.uniform(0, 1e-6))
        return best_node
    
    def expand(self):
        """ Expand single node by removing one untried state and creating a child node from it. """
        if not self._untried_states: return None 
        action_state = self._untried_states.pop() 
        child_node = MCTSNode(action_state, parent=self, problem=self.problem)
        self.children.append(child_node)
        return child_node

    def is_fully_expanded(self): return len(self._untried_states) == 0

    def is_terminal(self): return self.problem.is_goal(self.state) if self.problem else False


def mcts_search(problem, iterations=100000, max_depth=150):
    """Performs MCTS. Handles variable costs in simulation and final path cost."""
    start_time = time.time() 
    start_node = problem.initial_state()
    root = MCTSNode(state=start_node, problem=problem)
    if root.is_terminal(): return {"path": [root.state], "cost": 0, "nodes_expanded": 1, "time": time.time()-start_time, "algorithm": "MCTS", "iterations": 0}
    
    for i in range(iterations):
        node = root 
        path_to_leaf = [node]
        # 1. Selection
        while not node.is_terminal() and node.is_fully_expanded():  # traverse while not at goal and all children of this node are expanded
            selected_child = node.best_child() # Select best child randomly if unvisited or based on UCB1 if visited
            if selected_child is None: break 
            node = selected_child
            path_to_leaf.append(node)
        # 2. Expansion
        if not node.is_terminal() and not node.is_fully_expanded():  # from last node, expand one untried state
            expanded_node = node.expand() 
            if expanded_node: 
                node = expanded_node
                path_to_leaf.append(node)
        # 3. Simulation: randomly move from current state to goal state or max depth
        current_state = node.state
        reward = 0
        sim_depth = 0
        sim_path_cost = 0 
        sim_history = {current_state}   # like closed set
        while not problem.is_goal(current_state) and sim_depth < max_depth:
            neighbors_info = problem.get_neighbors(current_state) # list of tuples (neighbor_state, move_info) or list of neighbour states
            valid_neighbors_info = []
            for info in neighbors_info:
                if isinstance(info, tuple):
                    next_s, move_info = info
                else:
                    next_s = info
                    move_info = None  # Assign None if move info is not provided

                if next_s not in sim_history:
                    valid_neighbors_info.append((next_s, move_info))

            if not valid_neighbors_info:
                break

            next_state, move_info = random.choice(valid_neighbors_info) # Randomly select one of the valid neighbors
            sim_path_cost += problem.get_cost(current_state, next_state, move_info) 
            current_state = next_state
            sim_history.add(current_state) 
            sim_depth += 1
            
        if problem.is_goal(current_state): 
             reward = 1000.0 / (1 + sim_path_cost) if sim_path_cost >= 0 else 1000.0
        else: 
            reward = -problem.heuristic(current_state)  # or -1.0 * (1 + sim_path_cost) if sim_path_cost >= 0 else -1.0 
        # 4. Backpropagation
        for node_in_path in reversed(path_to_leaf):
            if node_in_path.visits <= 0: node_in_path.visits = 0 # Ensure not negative
            node_in_path.visits += 1 
            node_in_path.value += reward 

    end_time = time.time()
    
    # --- Extracting Path and Cost from MCTS Tree (using BFS and get_cost) ---
    #goal_node_in_tree = None
    min_cost_in_tree = float('inf')
    queue = collections.deque([(root, [root.state], 0)]) 
    visited_in_tree = {root.state: 0} 
    nodes_explored_in_tree = 0; best_path_found_list = None

    while queue:
         current_node, current_path_list, current_cost = queue.popleft()
         nodes_explored_in_tree += 1

         if current_node.is_terminal():
             if current_cost < min_cost_in_tree:
                  min_cost_in_tree = current_cost
                  best_path_found_list = current_path_list 
         
         # Only explore children if potentially better path exists
         if current_cost >= min_cost_in_tree: continue 

         for child in current_node.children:
             cost_step = problem.get_cost(current_node.state, child.state) # Use fallback cost calc
             new_cost = current_cost + cost_step
             
             if new_cost < visited_in_tree.get(child.state, float('inf')):
                  visited_in_tree[child.state] = new_cost
                  new_path_list = current_path_list + [child.state]
                  # Add only if potentially better than best complete path found so far
                  if new_cost < min_cost_in_tree: 
                      queue.append((child, new_path_list, new_cost))

    if best_path_found_list:
        return {"path": best_path_found_list, "cost": min_cost_in_tree, "nodes_expanded": nodes_explored_in_tree, "time": end_time - start_time, "algorithm": "MCTS", "iterations": iterations, "tree_root_visits": root.visits }
    else:
         best_first_move_node = root.best_child(exploration_weight=0) 
         return {"path": None, "cost": -1, "nodes_expanded": nodes_explored_in_tree, "time": end_time - start_time, "algorithm": "MCTS", "iterations": iterations, "best_next_state_estimate": best_first_move_node.state if best_first_move_node else None, "tree_root_visits": root.visits}


# --- Heuristic MCTS Implementation ---

class HeuristicMCTSNode:  #(MCTSNode): # Inherit from previous MCTSNode if needed, or define fully
    """ MCTS Node extended for heuristic guidance. """
    def __init__(self, state, parent=None, problem=None):
        # Basic MCTS Node attributes
        self.state = state
        self.parent = parent
        self.problem = problem 
        self.children = []
        self.visits = 0
        self.value = 0.0 # Accumulated reward (e.g., -cost, win/loss, -heuristic)
        
        # Heuristic value (cache if expensive to compute)
        self._heuristic_value = None 
        
        # Manage untried actions/states
        self._untried_actions_info = problem.get_neighbors(state) if problem else []
        # Store states for expansion control
        self._untried_states = [info[0] if isinstance(info, tuple) else info for info in self._untried_actions_info]
        random.shuffle(self._untried_states)

    def get_heuristic(self):
        """ Calculates or retrieves the cached heuristic value for the node's state. """
        if self._heuristic_value is None and self.problem:
            self._heuristic_value = self.problem.heuristic(self.state)
        return self._heuristic_value if self._heuristic_value is not None else float('inf') # Default if no problem/heuristic

    def is_fully_expanded(self): 
        return len(self._untried_states) == 0

    def expand(self):
        """ Expands the node by creating one child node from untried states. """
        if not self._untried_states: return None 
        # could be random or based on heuristic; applying heuristics in selection and/or rollout considered more impactful
        action_state = self._untried_states.pop() 
        # Create a new node of the same type (HeuristicMCTSNode)
        child_node = HeuristicMCTSNode(action_state, parent=self, problem=self.problem) 
        self.children.append(child_node)
        return child_node

    def best_child(self, exploration_weight=1.41, heuristic_weight=0.0, epsilon=1e-6):
        """ Selects the best child using UCB1 potentially modified by heuristic. """
        if not self.children: return None
        
        # Ensure parent visits is positive for log calculation
        parent_visits = self.visits if self.visits > 0 else 1 
        log_total_visits = math.log(parent_visits)

        best_score = -float('inf')
        best_children = [] # Handle ties

        for child in self.children:
            if child.visits == 0:
                # Prioritize unvisited children slightly differently if using heuristic
                # Can give them a high initial score or use heuristic directly?
                # Let's give a very high score, potentially modified by heuristic later if desired
                score = float('inf')
                if heuristic_weight > 0:
                    h_val = child.get_heuristic()
                    # Scale heuristic bonus inversely: bonus = weight / (1 + h)
                    score = heuristic_weight / (epsilon + h_val) 
            else:
                # UCB1 components
                exploitation = child.value / child.visits # Average reward
                exploration = exploration_weight * math.sqrt(log_total_visits / child.visits)
                
                # Heuristic component (lower heuristic is better -> higher score)
                heuristic_term = 0
                if heuristic_weight > 0:
                    h_val = child.get_heuristic()
                    # Scale heuristic bonus inversely: bonus = weight / (1 + h)
                    heuristic_term = heuristic_weight / (epsilon + h_val) # Add epsilon to prevent div by zero if h=0
                    # Alternative scaling: Normalize heuristic? Requires knowing range.
                
                score = exploitation + exploration + heuristic_term

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        # Break ties randomly
        return random.choice(best_children) if best_children else None

    def is_terminal(self): 
        return self.problem.is_goal(self.state) if self.problem else False


def heuristic_mcts_search(problem, 
                          iterations=100000, 
                          max_depth=150, 
                          exploration_weight=1.41, 
                          heuristic_weight=0.0, # Controls heuristic influence in selection
                          heuristic_rollout=False, # Controls heuristic use in simulation
                          epsilon=1e-6, # Small value for division stability
                         ):
    """
    Performs MCTS search, optionally using heuristic guidance in selection
    and/or simulation phases.
    """
    start_time = time.time()
    start_node = problem.initial_state()
    # Use the HeuristicMCTSNode
    root = HeuristicMCTSNode(state=start_node, problem=problem) 

    if root.is_terminal(): 
        return {"path": [root.state], "cost": 0, "nodes_expanded": 1, "time": time.time()-start_time, 
                "algorithm": "Heuristic MCTS", "iterations": 0, "h_weight": heuristic_weight, "h_rollout": heuristic_rollout}


    for i in range(iterations):
        node = root
        path_to_leaf = [node]
        
        # 1. Selection (using potentially heuristic-guided best_child)
        while not node.is_terminal() and node.is_fully_expanded():
            selected_child = node.best_child(exploration_weight=exploration_weight, 
                                             heuristic_weight=heuristic_weight, 
                                             epsilon=epsilon) 
            if selected_child is None: break 
            node = selected_child
            path_to_leaf.append(node)
            
        # 2. Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            expanded_node = node.expand() 
            if expanded_node: 
                node = expanded_node # Move to the new node
                path_to_leaf.append(node)
                
        # 3. Simulation (Rollout) - Potentially heuristic-guided
        current_state = node.state
        reward = 0
        sim_depth = 0
        sim_path_cost = 0 
        sim_history = {current_state} 

        while not problem.is_goal(current_state) and sim_depth < max_depth:
            neighbors_info = problem.get_neighbors(current_state)
            valid_neighbors_info = []
            for info in neighbors_info:
                if isinstance(info, tuple):
                    next_s, move_info = info
                else:
                    next_s = info
                    move_info = None  # Assign None if move info is not provided

                if next_s not in sim_history:
                    valid_neighbors_info.append((next_s, move_info))

            if not valid_neighbors_info:            # Dead end in simulation
                break
            
            next_state = None
            move_info = None

            if heuristic_rollout and valid_neighbors_info:
                # Heuristic-biased rollout: Choose neighbor probabilistically based on h value
                neighbor_states = [ni[0] for ni in valid_neighbors_info]
                heuristics = [problem.heuristic(ns) for ns in neighbor_states]
                
                # Calculate weights (lower heuristic -> higher weight)
                weights = [1.0 / (epsilon + h) for h in heuristics]
                total_weight = sum(weights)
                
                if total_weight > 0:
                    probabilities = [w / total_weight for w in weights]
                    # Choose based on calculated probabilities
                    chosen_index = random.choices(range(len(valid_neighbors_info)), weights=probabilities, k=1)[0]
                    next_state, move_info = valid_neighbors_info[chosen_index]
                else: 
                    # Fallback if all weights are zero (e.g., all heuristics infinite?)
                    next_state, move_info = random.choice(valid_neighbors_info)
            
            elif valid_neighbors_info: # Standard random rollout
                next_state, move_info = random.choice(valid_neighbors_info)

            else: # Should not happen if break condition works
                 break

            if next_state is None: break # Safety check

            sim_path_cost += problem.get_cost(current_state, next_state, move_info) 
            current_state = next_state
            sim_history.add(current_state)
            sim_depth += 1
            
        # Calculate reward based on simulation outcome
        if problem.is_goal(current_state): 
             # Higher reward for lower cost paths found in simulation
             reward = 1000000.0 / (1 + sim_path_cost) if sim_path_cost >= 0 else 1000000.0
        else: 
             # Use negative heuristic of final state as penalty
             reward = -problem.heuristic(current_state) 
             # Alternative: Fixed penalty for not reaching goal, or -sim_path_cost
             
        # 4. Backpropagation
        for node_in_path in reversed(path_to_leaf):
            # Ensure visits starts correctly for UCB calculation later
            if node_in_path.visits <= 0: node_in_path.visits = 0
            node_in_path.visits += 1
            node_in_path.value += reward 


    end_time = time.time()
    
    # --- Extracting Best Path Found in Tree ---
    # Use BFS starting from root to find the best path based on cost
    min_cost_in_tree = float('inf')
    queue = collections.deque([(root, [root.state], 0)]) # Node, path_list, cost_so_far
    visited_in_tree = {root.state: 0} 
    nodes_explored_in_tree = 0
    best_path_found_list = None

    while queue:
         current_node, current_path_list, current_cost = queue.popleft()
         nodes_explored_in_tree += 1

         if current_node.is_terminal():
             if current_cost < min_cost_in_tree:
                  min_cost_in_tree = current_cost
                  best_path_found_list = current_path_list 
         
         if current_cost >= min_cost_in_tree: continue # Pruning BFS

         for child in current_node.children:
             cost_step = problem.get_cost(current_node.state, child.state) 
             new_cost = current_cost + cost_step
             
             if new_cost < visited_in_tree.get(child.state, float('inf')):
                  # Check cost before adding to prevent cycles/redundancy? Or trust closed set?
                  # Let's add if cheaper or not visited in this BFS path extraction phase
                  visited_in_tree[child.state] = new_cost
                  new_path_list = current_path_list + [child.state]
                  if new_cost < min_cost_in_tree: # Only explore if potentially better
                       queue.append((child, new_path_list, new_cost))

    # Prepare results dictionary
    algo_name = f"Heuristic MCTS (SelW={heuristic_weight:.2f}, Rollout={heuristic_rollout})"
    if best_path_found_list:
        return {"path": best_path_found_list, "cost": min_cost_in_tree, "nodes_expanded": nodes_explored_in_tree, 
                "time": end_time - start_time, "algorithm": algo_name, "iterations": iterations, 
                "tree_root_visits": root.visits, "h_weight": heuristic_weight, "h_rollout": heuristic_rollout }
    else:
         # If goal not found, return best guess for next move from root (greedy exploitation)
         best_first_move_node = root.best_child(exploration_weight=0, heuristic_weight=0) # Pure exploitation
         return {"path": None, "cost": -1, "nodes_expanded": nodes_explored_in_tree, 
                 "time": end_time - start_time, "algorithm": algo_name, "iterations": iterations, 
                 "best_next_state_estimate": best_first_move_node.state if best_first_move_node else None, 
                 "tree_root_visits": root.visits, "h_weight": heuristic_weight, "h_rollout": heuristic_rollout}




# --- Main Execution Logic ---

if __name__ == "__main__":
    print(f"Running search comparison at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    random.seed(42)

    iterations = 100            # MCTS  1000000 finds near-optimal paths in 8-puzzle and occasionally pancake
    max_depth = 150             # MCTS
    heuristic_weight = 100.0    # MCTS
    make_heuristic_inadmissable = False # Set to True to make heuristic inadmissible
    tile_degradation = 0
    pancake_degradation = 0
    hanoi_degradation = 0

    # --- Define Problems ---
    tile_initial = [1, 2, 3, 0, 4, 6, 7, 5, 8] # Medium C*=3
    #tile_initial = [8, 6, 7, 2, 5, 4, 3, 0, 1] # harder C*=32
    sliding_tile_unit_cost = SlidingTileProblem(initial_state=tile_initial, 
                                                use_variable_costs=False, 
                                                make_heuristic_inadmissable=make_heuristic_inadmissable,
                                                degradation=tile_degradation)
    sliding_tile_var_cost = SlidingTileProblem(initial_state=tile_initial, 
                                               use_variable_costs=True,
                                               make_heuristic_inadmissable=make_heuristic_inadmissable,
                                               degradation=tile_degradation)

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
                                         make_heuristic_inadmissable=make_heuristic_inadmissable,
                                         degradation=hanoi_degradation)


    problems_to_solve = [
        sliding_tile_unit_cost,
        sliding_tile_var_cost,
        pancake_unit_cost,
        pancake_var_cost,
        hanoi_problem
    ]

    # --- Define Algorithms ---


    def run_ucs(problem): return generic_search(problem, priority_key='g')
    def run_greedy_bfs(problem): return generic_search(problem, priority_key='h')
    def run_astar(problem): return generic_search(problem, priority_key='f')
    def run_bidir_astar(problem): return bidirectional_a_star_search(problem)
#    def run_mcts(problem, iterations=100000, max_depth=150): 
#         default_iterations = 100000
#         default_depth = 150
         #if isinstance(problem, TowersOfHanoiProblem): 
         #     default_iterations = 20000; default_depth = problem.num_disks * 3 
#         final_iterations = iterations if iterations is not None else default_iterations
#         final_depth = max_depth if max_depth is not None else default_depth
#         return mcts_search(problem, iterations=final_iterations, max_depth=final_depth)
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
        "A*": run_astar,
        "Bidirectional A*": run_bidir_astar,
#        "MCTS": run_mcts,                      # Original MCTS, no heuristic options 
        "MCTS (Standard)": run_mcts_standard,
        "MCTS (H-Select)": run_mcts_h_select, # Add heuristic versions
        "MCTS (H-Rollout)": run_mcts_h_rollout,
        "MCTS (H-Both)": run_mcts_h_both,
    }


    # --- Run Experiments ---
    all_results = []
    for problem in problems_to_solve:
        print(f"\n{'=' * 20}\nSolving: {problem}\nInitial State: {problem.initial_state()}\nGoal State:    {problem.goal_state()}\nInitial Heuristic: {problem.heuristic(problem.initial_state())}\n{'-' * 20}")
        problem_results = []
        
        for algo_display_name, algo_func in search_algorithms_runners.items():
            print(f"Running {algo_display_name}...")
            result = None
            try:
                result = algo_func(problem) # Call the runner
                
                # Set algorithm name in result consistently
                if result and 'algorithm' in result: 
                    if "Generic" in result['algorithm']:
                         result['algorithm'] = f"{algo_display_name} (Generic)"
                    else: 
                         result['algorithm'] = algo_display_name
                
                print(f"{algo_display_name} Done.")

            except Exception as e:
                print(f"!!! ERROR during {algo_display_name} on {problem}: {e}")
                traceback.print_exc() 
                result = { "path": None, "cost": -1, "nodes_expanded": -1, "time": -1, 
                           "algorithm": algo_display_name, "error": str(e)}

            if result: 
                 result['problem'] = str(problem); problem_results.append(result); all_results.append(result)

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
         algo_name = res.get('algorithm','N/A').replace(" (Generic)", "") 
         print(f"- Problem: {res.get('problem','N/A')}, Algorithm: {algo_name}, Time: {res.get('time',-1):.4f}s, Nodes: {res.get('nodes_expanded',-1)}, Status: {status} {optimal_note}")
