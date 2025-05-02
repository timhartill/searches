"""
BiDirectional Searches

Bidirectional A*

"""
import time

import util


# --- Bidirectional A* (Updated for variable cost) ---
class bidirectional_a_star_search:
    """Performs Bidirectional A* search. Handles variable costs."""
    def __init__(self, tiebreaker1='-g', tiebreaker2='NONE', visualise=True, visualise_dirname = ''):
        self.visualise = visualise
        self.visualise_dirname = visualise_dirname
        self.tiebreaker1 = tiebreaker1  # see calc_tiebreak_val for options
        self.tiebreaker2 = tiebreaker2
        self._str_repr = f"Bidirectional AStar-pf-tb1{tiebreaker1}-tb2{tiebreaker2}"


    def search(self, problem):
        optimality_guaranteed = problem.optimality_guaranteed

        start_time = time.time()
        start_node = problem.initial_state()
        goal_node = problem.goal_state()
        if problem.is_goal(start_node): return {"path": [start_node], "cost": 0, "nodes_expanded": 0, "nodes_expanded_below_cstar": 0,
                                                "time": 0, "optimal": optimality_guaranteed, 'visual': 'no file', "max_heap_size": 0}

        h_start = problem.heuristic(start_node)
        frontier_fwd = util.PriorityQueue(tiebreaker1=self.tiebreaker1, tiebreaker2=self.tiebreaker2)
        frontier_fwd.push(start_node, h_start, 0) # Push with priority and tiebreaker
        came_from_fwd = {start_node: None}
        g_score_fwd = {start_node: 0}
        closed_fwd = set() 

        h_goal = problem.heuristic(goal_node, backward=True)
        frontier_bwd = util.PriorityQueue(tiebreaker1=self.tiebreaker1, tiebreaker2=self.tiebreaker2)
        frontier_bwd.push(goal_node, h_goal, 0) # Push with priority and tiebreaker
        came_from_bwd = {goal_node: None}
        g_score_bwd = {goal_node: 0}
        closed_bwd = set() 

        nodes_expanded = 0
        meeting_node = None
        max_heap_size_combined = 0
        C = -1.0        # Current lowest cost on either frontier
        U = float('inf') # Current lowest cost of path found

        if hasattr(problem, "cstar"):
            cstar = problem.cstar
        else:
            cstar = None
        nodes_expanded_below_cstar = 0


        while not frontier_fwd.isEmpty() and not frontier_bwd.isEmpty():
            C = min(frontier_fwd.peek(priority_only=True), 
                    frontier_bwd.peek(priority_only=True))
            
            if C >= U: # If the estimated lowest cost path on frontier is greater cost than the best path found, stop
                print(f"1. Termination condition U ({U}) >= C ({C}) met.")  # In practice this condition isnt triggered because of the optimization below in expansion: if f_score < U ...
                break

            # --- Forward Step ---
            if frontier_fwd:
                current_state_fwd = frontier_fwd.pop(item_only=True)   # item, priority, tiebreaker1, tiebreaker2
                if current_state_fwd in closed_fwd: 
                    continue
                current_g_fwd = g_score_fwd.get(current_state_fwd, float('inf'))
                if current_g_fwd  + problem.heuristic(current_state_fwd, backward=False) >= U: 
                    continue  
                closed_fwd.add(current_state_fwd)
                nodes_expanded += 1
                if cstar and current_g_fwd < cstar:
                    nodes_expanded_below_cstar += 1

                if current_state_fwd in g_score_bwd: 
                    current_path_cost = current_g_fwd + g_score_bwd[current_state_fwd]
                    if current_path_cost < U:   
                        U = current_path_cost
                        meeting_node = current_state_fwd
                        #break   #NOTE: if break here tend to get optimal or nearly optimal paths with far fewer node expansions than A*
                    #else:  # finds nonoptimal paths
                    #    print(f"2. Terminating as current path cost {current_path_cost} >= U {U}.")
                    #    break    
                
                for neighbor_info in problem.get_neighbors(current_state_fwd):
                    # Handle cases where get_neighbors might return just state or (state, move_info)
                    if isinstance(neighbor_info, tuple) and len(neighbor_info) >= 1:
                        neighbor_state = neighbor_info[0]
                        move_info = neighbor_info[1] if len(neighbor_info) > 1 else None
                    else:
                        neighbor_state = neighbor_info
                        move_info = None
                    if neighbor_state in closed_fwd: continue
                    cost = problem.get_cost(current_state_fwd, neighbor_state, move_info) 
                    tentative_g_score = current_g_fwd + cost
                    if tentative_g_score < g_score_fwd.get(neighbor_state, float('inf')):
                        came_from_fwd[neighbor_state] = current_state_fwd 
                        g_score_fwd[neighbor_state] = tentative_g_score
                        h_score = problem.heuristic(neighbor_state) 
                        f_score = tentative_g_score + h_score
                        if f_score < U: 
                            frontier_fwd.push(neighbor_state, f_score, 
                                              self.calc_tiebreak(g=tentative_g_score, 
                                                                 h=h_score, 
                                                                 count_tb1=frontier_fwd.count_tb1))  # Use -g score as tiebreaker to prefer higher g_score
            
            # --- Backward Step ---
            if frontier_bwd:
                current_state_bwd = frontier_bwd.pop(item_only=True)   # item, priority, tiebreaker1, tiebreaker2
                if current_state_bwd in closed_bwd: 
                    continue
                current_g_bwd = g_score_bwd.get(current_state_bwd, float('inf'))
                if current_g_bwd + problem.heuristic(current_state_bwd, backward=True) >= U: 
                    continue 
                closed_bwd.add(current_state_bwd)
                nodes_expanded += 1
                if cstar and current_g_bwd < cstar:
                    nodes_expanded_below_cstar += 1

                if current_state_bwd in g_score_fwd: 
                    current_path_cost = g_score_fwd[current_state_bwd] + current_g_bwd
                    if current_path_cost < U: 
                        U = current_path_cost
                        meeting_node = current_state_bwd
                        #break  #NOTE: if break here tend to get optimal or nearly optimal paths with far fewer node expansions than A*
                    #else: #finds non-optimal paths
                    #    print(f"2. Terminating as current path cost {current_path_cost} >= U {U}.")
                    #    break    

                for neighbor_info in problem.get_neighbors(current_state_bwd):
                    # Handle cases where get_neighbors might return just state or (state, move_info)
                    if isinstance(neighbor_info, tuple) and len(neighbor_info) >= 1:
                        neighbor_state = neighbor_info[0]
                        move_info = neighbor_info[1] if len(neighbor_info) > 1 else None
                    else:
                        neighbor_state = neighbor_info
                        move_info = None

                    if neighbor_state in closed_bwd: continue
                    cost = problem.get_cost(current_state_bwd, neighbor_state, move_info) 
                    tentative_g_score = current_g_bwd + cost 
                    if tentative_g_score < g_score_bwd.get(neighbor_state, float('inf')):
                        came_from_bwd[neighbor_state] = current_state_bwd 
                        g_score_bwd[neighbor_state] = tentative_g_score
                        h_score = problem.heuristic(neighbor_state, backward=True)
                        f_score = tentative_g_score + h_score
                        if f_score < U: 
                            frontier_bwd.push(neighbor_state, f_score, 
                                              self.calc_tiebreak(g=tentative_g_score, 
                                                                 h=h_score, 
                                                                 count_tb1=frontier_fwd.count_tb1))  # Use -g score as tiebreaker to prefer higher g_score

            if frontier_fwd.max_heap_size + frontier_bwd.max_heap_size > max_heap_size_combined:
                max_heap_size_combined = frontier_fwd.max_heap_size + frontier_bwd.max_heap_size
            
        end_time = time.time()
        image_file = 'no file'   
        if meeting_node:
            path = reconstruct_bidirectional_path(came_from_fwd, came_from_bwd, start_node, goal_node, meeting_node)
            if self.visualise and hasattr(problem, 'visualise'):
                image_file = problem.visualise(path=path, path_type=self._str_repr, 
                                            meeting_node=meeting_node, visited_fwd=closed_fwd, visited_bwd=closed_bwd, 
                                            visualise_dirname=self.visualise_dirname)
                if not image_file: image_file = 'no file'
            final_cost = -1
            recalculated_cost = 0
            cost_mismatch = False
            if path:  # check path reconstruction - NOT NEEDED post debugging!!
                try:
                    for i in range(len(path) - 1):
                        recalculated_cost += problem.get_cost(path[i], path[i+1]) # Use fallback cost
                    final_cost = recalculated_cost
                    if abs(final_cost - U) > 1e-6: 
                        cost_mismatch = True
                except Exception as e:
                    print(f"Error recalculating bidirectional path cost: {e}")
                    final_cost = U 
            if cost_mismatch: 
                print(f"Warning: Bidirectional cost mismatch! PathRecalc={final_cost}, SearchCost={U}")
            final_reported_cost = U # Report cost found by search
            return {"path": path, "cost": final_reported_cost if path else -1, 
                    "nodes_expanded": nodes_expanded,  "nodes_expanded_below_cstar": nodes_expanded_below_cstar,
                    "time": end_time - start_time, "optimal": optimality_guaranteed, "visual": image_file,
                    "max_heap_size": max_heap_size_combined}
        else:
            return {"path": None, "cost": -1, "nodes_expanded": nodes_expanded, "nodes_expanded_below_cstar": nodes_expanded_below_cstar,
                    "time": end_time - start_time, "optimal": optimality_guaranteed, "visual": image_file,
                    "max_heap_size": max_heap_size_combined} # No path found


    def calc_tiebreak(self, g, h, count_tb1):
        """Calculates the tiebreaker value based on the type and values of g and h"""
        if self.tiebreaker1 == 'g':
            return g
        elif self.tiebreaker1 == '-g':  # higher g popped first
            return -g
        elif self.tiebreaker1 == 'h':
            return h
        elif self.tiebreaker1 == 'f':
            return g + h
        elif self.tiebreaker1 in ['FIFO', 'LIFO']:
            return count_tb1
        elif self.tiebreaker1 == 'NONE':
            return 0
        else:
            raise ValueError(f"Invalid tiebreaker1: {self.tiebreaker1}")


    def __str__(self): # enable str(object) to return algo name
        return self._str_repr


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


