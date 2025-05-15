"""
Bidirectional Searches

Bidirectional A*, Bi dir Uniform Cost and Bi dir Best-First

"""
import time
import util
import data_structures

algo_name_map = {'g': "BiDirUniformCost", 'h': "BiDirGreedyBestFirst", 'f': "BiDirAstar"} 



# --- Bidirectional generic search eg bd A* (Updated for variable cost) ---
class bd_generic_search:
    """ Performs Bidirectional generic search ie bd a*, bd uniform cost or bd best first.
    Priority can be based on 'g', 'h', or 'f' = g+h. Handles variable costs.
    if visualise is True and problem supports it, will output visualisation to a subdir off the problem input dir.
    """
    def __init__(self, priority_key='f', tiebreaker1='-g', tiebreaker2='NONE', 
                 visualise=True, visualise_dirname = '', min_ram=2.0, timeout=30.0):
        if priority_key not in algo_name_map: raise ValueError(f"priority_key must be in {algo_name_map}")
        self.timeout = timeout
        self.min_ram = min_ram
        self.visualise = visualise
        self.visualise_dirname = visualise_dirname
        self.priority_key = priority_key
        self.tiebreaker1 = tiebreaker1  # see calc_tiebreak_val for options
        self.tiebreaker2 = tiebreaker2
        self._str_repr = f"{algo_name_map[self.priority_key]}-p{self.priority_key}-tb1{tiebreaker1}-tb2{tiebreaker2}"


    def search(self, problem):
        #optimality_guaranteed = problem.optimality_guaranteed
        optimality_guaranteed = (self.priority_key == 'g') or (self.priority_key=='f' and problem.optimality_guaranteed)

        start_time = time.time()
        start_node = problem.initial_state()
        goal_node = problem.goal_state()
        if problem.is_goal(start_node): return {"path": [start_node], "cost": 0, "nodes_expanded": 0, "nodes_expanded_below_cstar": 0,
                                                "time": 0, "optimal": optimality_guaranteed, 'visual': 'no file', "max_heap_size": 0}

        h_initial = problem.heuristic(start_node)
        frontier_fwd = data_structures.PriorityQueue(priority_key=self.priority_key, 
                                                     tiebreaker1=self.tiebreaker1, tiebreaker2=self.tiebreaker2)
        frontier_fwd.push(start_node, frontier_fwd.calc_priority(0, h_initial), 0) # Push with priority and tiebreaker
        came_from_fwd = {start_node: None}
        g_score_fwd = {start_node: 0}
        closed_fwd = set() 

        h_goal = problem.heuristic(goal_node, backward=True)
        frontier_bwd = data_structures.PriorityQueue(priority_key=self.priority_key, 
                                                     tiebreaker1=self.tiebreaker1, tiebreaker2=self.tiebreaker2)
        frontier_bwd.push(goal_node, frontier_bwd.calc_priority(0, h_goal), 0) # Push with priority and tiebreaker
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
        #nodes_expanded_below_cstar = 0
        i = 0
        checkmem = 1000
        status = ""
        cond_count = 0

        while not frontier_fwd.isEmpty() and not frontier_bwd.isEmpty():
            if frontier_fwd.max_heap_size + frontier_bwd.max_heap_size > max_heap_size_combined:
                max_heap_size_combined = frontier_fwd.max_heap_size + frontier_bwd.max_heap_size
            if (time.time()-start_time)/60.0 > self.timeout:
                status += f"Timeout after {(time.time()-start_time)/60:.4f} mins."
                break
            if i % checkmem == 0 and util.get_available_ram() < self.min_ram:
                status += f"Out of RAM ({util.get_available_ram():.4f}GB remaining)."
                break
            i += 1

            C = min(frontier_fwd.peek(priority_only=True), 
                    frontier_bwd.peek(priority_only=True))
            
            if C >= U: # If the estimated lowest cost path on frontier is greater cost than the best path found, stop
                status += f"Completed. Termination condition C ({C}) >= U ({U}) met."
                break

            # --- Forward Step ---
            if not frontier_fwd.isEmpty() and C == frontier_fwd.peek(priority_only=True):
                current_state_fwd = frontier_fwd.pop(item_only=True)   # item, priority, tiebreaker1, tiebreaker2
                if current_state_fwd in closed_fwd: 
                    continue
                closed_fwd.add(current_state_fwd)
                current_g_fwd = g_score_fwd.get(current_state_fwd, float('inf'))
                nodes_expanded += 1
                #if cstar and current_g_fwd < cstar:
                #    nodes_expanded_below_cstar += 1
                #if current_g_fwd >= U:  # + problem.heuristic(current_state_fwd, backward=False) >= U: #TODO remove?
                #    cond_count += 1
                #    continue

                if current_state_fwd in g_score_bwd: 
                    current_path_cost = current_g_fwd + g_score_bwd[current_state_fwd]
                    if current_path_cost < U:
                        U = current_path_cost
                        meeting_node = current_state_fwd
                        if self.priority_key == 'h':  # BFS is not optimal so may as well end as soon as a path found
                            status += f"Terminating BFS as path found. U:{U}."
                            break
                        continue # TODO No sense expanding?
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
                        #f_score = tentative_g_score + h_score
                        #if f_score < U: # Enabling this caused ~ 30% fewer node expansions but would be cheating for uc and might cause inadmissable heuristics to make the search fail
                        frontier_fwd.push(  neighbor_state, 
                                            frontier_fwd.calc_priority(g=tentative_g_score, h=h_score), 
                                            frontier_fwd.calc_tiebreak1(g=tentative_g_score, h=h_score))  # Use -g score as tiebreaker to prefer higher g_score
            
            # --- Backward Step ---
            if not frontier_bwd.isEmpty() and C == frontier_bwd.peek(priority_only=True):
                current_state_bwd = frontier_bwd.pop(item_only=True)   # item, priority, tiebreaker1, tiebreaker2
                if current_state_bwd in closed_bwd: 
                    continue
                current_g_bwd = g_score_bwd.get(current_state_bwd, float('inf'))
                closed_bwd.add(current_state_bwd)
                nodes_expanded += 1
                #if cstar and current_g_bwd < cstar:
                #    nodes_expanded_below_cstar += 1
                #if current_g_bwd >= U: #+ problem.heuristic(current_state_bwd, backward=True) >= U: #TODO remove
                #    cond_count += 1
                #    continue 

                if current_state_bwd in g_score_fwd: 
                    current_path_cost = g_score_fwd[current_state_bwd] + current_g_bwd
                    if current_path_cost < U: 
                        U = current_path_cost
                        meeting_node = current_state_bwd
                        if self.priority_key == 'h':  # BFS is not optimal so may as well end as soon as a path found
                            status += f"Terminating BFS as path found. U:{U}."
                            break
                        continue    #TODO "continue" here as pointless expanding the node after it found a path?
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
                        #f_score = tentative_g_score + h_score
                        #if f_score < U: # Enabling this caused ~ 30% fewer node expansions but would be cheating for uc and might also cause inadmissable heuristics to fail
                        frontier_bwd.push(  neighbor_state, 
                                            frontier_bwd.calc_priority(g=tentative_g_score, h=h_score), 
                                            frontier_bwd.calc_tiebreak1(g=tentative_g_score, h=h_score))  # Use -g score as tiebreaker to prefer higher g_score

            
        end_time = time.time()
        if not status:
            status = "Completed."
        if cond_count > 0:
            status += f" {cond_count} dup nodes skipped."    
        print(status)    
        image_file = 'no file'   
        nodes_expanded_below_cstar = 0
        print(f"Calculating count of nodes below {U} from forward closed set |{len(closed_fwd)}|..")
        for state in closed_fwd:
            if g_score_fwd.get(state, 0) < U:
                nodes_expanded_below_cstar += 1
        print(f"Calculating count of nodes below {U} from backward closed set |{len(closed_bwd)}|..")
        for state in closed_bwd:
            if g_score_bwd.get(state, 0) < U:
                nodes_expanded_below_cstar += 1
        if meeting_node:
            path = reconstruct_bidirectional_path(came_from_fwd, came_from_bwd, start_node, goal_node, meeting_node)
            if not path:
                status += " Path too long to reconstruct."
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
                    "max_heap_len": max_heap_size_combined, 
                    "closed_set_len": len(closed_fwd)+len(closed_bwd), 
                    "g_score_len": len(g_score_fwd)+len(g_score_bwd),
                    "came_from_len": len(came_from_fwd)+len(came_from_bwd),
                    "status": status}

        status += " No path found."
        return {"path": None, "cost": -1, "nodes_expanded": nodes_expanded, "nodes_expanded_below_cstar": nodes_expanded_below_cstar,
                "time": end_time - start_time, "optimal": optimality_guaranteed, "visual": image_file,
                "max_heap_len": max_heap_size_combined, 
                "closed_set_len": len(closed_fwd)+len(closed_bwd), 
                "g_score_len": len(g_score_fwd)+len(g_score_bwd),
                "came_from_len": len(came_from_fwd)+len(came_from_bwd),
                "status": status} # No path found


    def __str__(self): # enable str(object) to return algo name
        return self._str_repr


def reconstruct_bidirectional_path(came_from_fwd, came_from_bwd, start_state, goal_state, meeting_node):
    """Reconstructs path for bidirectional search."""
    path1 = []
    curr = meeting_node
    
    limit = 100000 

    count = 0
    while curr is not None: 
        path1.append(tuple(curr))
        curr = came_from_fwd.get(curr)
        count += 1
        if count > limit: print("Error: Path fwd reconstruction exceeded limit."); return None
    path1.reverse() 
    
    path2 = []
    curr = came_from_bwd.get(meeting_node) 
    count = 0
    while curr is not None: 
         path2.append(tuple(curr))
         curr = came_from_bwd.get(curr)
         count += 1
         if count > limit: print("Error: Path bwd reconstruction exceeded limit."); return None
         
    return path1 + path2


