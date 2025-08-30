"""
Generic Bidirectional Searches

Bidirectional A*, Bi dir Uniform Cost and Bi dir Best-First

"""
import time
import random
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
                 visualise=True, visualise_dirname = '', min_ram=2.0, timeout=30.0, rust=False, bpmx1=False):
        if priority_key not in algo_name_map: raise ValueError(f"priority_key must be in {algo_name_map}")
        """
        priority_key: 'g', 'h', or 'f' = g+h. Determines the priority of the nodes in the search.
        visualise: If True, will output a visualisation of the search to a subdir off the output dir.
        tiebreaker1/2: 1st and 2nd level Tiebreaker for the priority queue. Can be eg 'g', 'FIFO', 'LIFO', or 'NONE' for no tiebreaker = heap ordering.
        min_ram: Minimum RAM in GB to keep available during search. If RAM goes below this, the search will (sometimes) stop but in practice Python may sometimes grab all mem and the os will kill the process before this condition fires.
        timeout: Timeout in minutes for the search. If the search takes longer than this, it will stop.
        """
        if rust:
            print(f"Rust usage is not implemented for bd_generic_search so using Python. To disable this message add --no_rust to the calling script but note that rust will then be disabled for everything.")
        if bpmx1:
            print(f"bpmx1 from Felner et al 2011 is not implemented for bidirectional search.")
        self.timeout = timeout
        self.min_ram = min_ram
        self.visualise = visualise
        self.visualise_dirname = visualise_dirname
        self.priority_key = priority_key
        self.tiebreaker1 = tiebreaker1  # see calc_tiebreak_val for options
        self.tiebreaker2 = tiebreaker2
        self._str_repr = f"{algo_name_map[self.priority_key]}-p{self.priority_key}-tb1{tiebreaker1}-tb2{tiebreaker2}"


    def search(self, problem):
        """ Run the search on a problem instance and return dict of results."""
        optimality_guaranteed = (self.priority_key == 'g') or (self.priority_key=='f' and problem.optimality_guaranteed)

        start_time = time.time()
        start_node = problem.initial_state()
        goal_node = problem.goal_state()
        if problem.is_goal(start_node): return {"path": [start_node], "cost": 0, "nodes_expanded": 0, "nodes_expanded_below_cstar": 0,
                                                "time": 0, "optimal": optimality_guaranteed, 'visual': 'no file', "max_heap_size": 0}

        h_initial = problem.heuristic(start_node)
        frontier_fwd = data_structures.PriorityQueue(priority_key=self.priority_key, 
                                                     tiebreaker1=self.tiebreaker1, tiebreaker2=self.tiebreaker2)
        frontier_fwd.push(start_node, 
                          frontier_fwd.calc_priority(0, h_initial), 0) # Push with priority and tiebreaker
        came_from_fwd = {start_node: None}
        g_score_fwd = {start_node: 0}
        closed_fwd = set() 

        h_goal = problem.heuristic(goal_node, backward=True)
        frontier_bwd = data_structures.PriorityQueue(priority_key=self.priority_key, 
                                                     tiebreaker1=self.tiebreaker1, tiebreaker2=self.tiebreaker2)
        frontier_bwd.push(goal_node, 
                          frontier_bwd.calc_priority(0, h_goal), 0) # Push with priority and tiebreaker
        came_from_bwd = {goal_node: None}
        g_score_bwd = {goal_node: 0}
        closed_bwd = set() 

        nodes_expanded = 0
        C_nonmono = -1  # C forced monotonic for counting nodes_expanded_below_cstar correctly in the case of non-monotonic C
        C = -1.0        # Current lowest cost on either frontier
        U = float('inf') # Current lowest cost of path found in either direction
        if hasattr(problem, "cstar"):
            cstar = problem.cstar
        else:
            cstar = None
        nodes_expanded_below_cstar = 0
        nodes_expanded_below_cstar_auto = 0
        c_count_dict = {}
        meeting_node = None
        max_heap_size_combined = 0
        i = 0
        checkmem = 1000
        status = ""
        stale_count = 0
        found_goal_count = 0
        U_update_count = 0
        h_consistent = True  # optionally check the consistency of the heuristic if running A* (not exhaustive)
        h_admissable = True  # optionally check the admissability of the heuristic if running A* and cstar is supplied (not exhaustive)
        priority_diminished = 0
        start_ram = util.get_available_ram()
        min_ram = start_ram

        while not frontier_fwd.isEmpty() and not frontier_bwd.isEmpty():
            if frontier_fwd.max_heap_size + frontier_bwd.max_heap_size > max_heap_size_combined:
                max_heap_size_combined = frontier_fwd.max_heap_size + frontier_bwd.max_heap_size
            if (time.time()-start_time)/60.0 > self.timeout:
                status += f"Timeout after {(time.time()-start_time)/60:.4f} mins."
                break
            if i % checkmem == 0:
                min_ram = min(min_ram, util.get_available_ram())
                if min_ram < self.min_ram:
                    status += f"Out of RAM ({min_ram:.4f}GB remaining)."
                    break
            i += 1

            current_priority = min( frontier_fwd.peek(priority_only=True), 
                                    frontier_bwd.peek(priority_only=True) )

            C = current_priority  # C = max(C, current_priority) <- this 'max' works empirically but concerned it *could* fail for inconsistent heuristic where priority diminishes
            C_nonmono = max(C_nonmono, current_priority)
            if current_priority + 1e-6 < C_nonmono:  # This can happen with inconsistent heuristic which causes a state to be re-visited with a smaller priority
                #print(f" Current priority {current_priority} is less than C {C}.")
                priority_diminished += 1

            if C >= U: # If the estimated lowest cost path on frontier is greater cost than the best path found, stop
                status += f"Completed. Termination condition C ({C}) >= U ({U}) met."
                break

            # --- Forward Step ---
            if not frontier_fwd.isEmpty() and current_priority == frontier_fwd.peek(priority_only=True):
                current_state_fwd = frontier_fwd.pop(item_only=True)   # item, priority, tiebreaker1, tiebreaker2
                current_g_fwd = g_score_fwd.get(current_state_fwd)
                if self.priority_key == 'g': 
                    current_h = 0
                else: 
                    current_h = problem.heuristic(current_state_fwd)
                    if self.priority_key == 'f' and h_admissable:
                        if cstar and current_g_fwd + current_h > cstar + 1e-6:
                            status += f" Inadmissable heuristic detected (Fwd)."
                            h_admissable = False
                # left the check for stale entries, but PriorityQueue now removes duplicates internally s.t. only non-stale entries are popped..
                if current_g_fwd + 1e-6 < current_priority - current_h:
                    stale_count += 1
                    continue

                #if current_state_fwd in closed_fwd: continue   # we don't need a closed set in this implementation
                #closed_fwd.add(current_state_fwd) 

                if current_state_fwd in g_score_bwd: 
                    current_path_cost = g_score_bwd[current_state_fwd] + current_g_fwd
                    found_goal_count += 1
                    if current_path_cost < U:
                        U = current_path_cost
                        meeting_node = current_state_fwd
                        U_update_count += 1
                        if self.priority_key == 'h':  # BFS is not optimal so may as well end as soon as a path found
                            status += f"Terminating BFS as path found (fwd). U:{U}."
                            break
                        #break   #NOTE: if break here tend to get optimal or nearly optimal paths with far fewer node expansions than A*
                    #else:  # finds nonoptimal paths
                    #    print(f"2. Terminating as current path cost {current_path_cost} >= U {U}.")
                    #    break    

                nodes_expanded += 1
                if cstar and C_nonmono < cstar:
                    nodes_expanded_below_cstar += 1
                if self.priority_key != 'h':
                    if c_count_dict.get(C_nonmono) is None:
                        c_count_dict[C_nonmono] = 0
                    c_count_dict[C_nonmono] +=1

                for neighbor_info in problem.get_neighbors(current_state_fwd):
                    # Handle cases where get_neighbors might return just state or (state, move_info)
                    if isinstance(neighbor_info, tuple) and len(neighbor_info) >= 1:
                        neighbor_state = neighbor_info[0]
                        move_info = neighbor_info[1] if len(neighbor_info) > 1 else None
                    else:
                        neighbor_state = neighbor_info
                        move_info = None
                    #if neighbor_state in closed_fwd: continue

                    cost = problem.get_cost(current_state_fwd, neighbor_state, move_info) 
                    tentative_g_score = current_g_fwd + cost

                    # Check whether current heuristic is consistent: if h(n) > cost(n, n') + h(n')
                    if self.priority_key == 'f' and h_consistent:
                        h_score = problem.heuristic(neighbor_state)
                        if current_h > cost + h_score + 1e-6:
                            status += f" Inconsistent heuristic detected (fwd)."
                            h_consistent = False

                    prior_g = g_score_fwd.get(neighbor_state, float('inf'))
                    if tentative_g_score < prior_g:
                        came_from_fwd[neighbor_state] = current_state_fwd 
                        g_score_fwd[neighbor_state] = tentative_g_score
                        h_score = problem.heuristic(neighbor_state) 
                        prior_f = prior_g + h_score   # NOTE: heuristic must always return same value for the same state, otherwise must store past heuristics in dict
                        frontier_fwd.push(  neighbor_state, 
                                            frontier_fwd.calc_priority(g=tentative_g_score, h=h_score),
                                            frontier_fwd.calc_tiebreak1(g=tentative_g_score, h=h_score),
                                            prior_f, prior_g)  
            
            # --- Backward Step ---
            if not frontier_bwd.isEmpty() and current_priority == frontier_bwd.peek(priority_only=True):
                current_state_bwd = frontier_bwd.pop(item_only=True)   # item, priority, tiebreaker1, tiebreaker2
                current_g_bwd = g_score_bwd.get(current_state_bwd)
                if self.priority_key == 'g': 
                    current_h = 0
                else: 
                    current_h = problem.heuristic(current_state_bwd, backward=True)
                    if self.priority_key == 'f' and h_admissable:
                        if cstar and current_g_bwd + current_h > cstar + 1e-6:
                            status += f" Inadmissable heuristic detected (Bwd)."
                            h_admissable = False
                # left the check for stale entries, but PriorityQueue now removes duplicates internally s.t. only non-stale entries are popped..
                if current_g_bwd + 1e-6 < current_priority - current_h:
                    stale_count += 1
                    continue

                #if current_state_bwd in closed_bwd: continue   # we don't need a closed set in this implementation
                #closed_bwd.add(current_state_bwd) 

                if current_state_bwd in g_score_fwd: 
                    current_path_cost = g_score_fwd[current_state_bwd] + current_g_bwd
                    found_goal_count += 1
                    if current_path_cost < U: 
                        U = current_path_cost
                        meeting_node = current_state_bwd
                        U_update_count += 1
                        if self.priority_key == 'h':  # BFS is not optimal so may as well end as soon as a path found
                            status += f"Terminating BFS as path found (bwd). U:{U}."
                            break
                        #break  #NOTE: if break here tend to get optimal or nearly optimal paths with far fewer node expansions than A*
                    #else: #finds non-optimal paths
                    #    print(f"2. Terminating as current path cost {current_path_cost} >= U {U}.")
                    #    break    

                nodes_expanded += 1
                if cstar and C_nonmono < cstar:
                    nodes_expanded_below_cstar += 1
                if self.priority_key != 'h':
                    if c_count_dict.get(C_nonmono) is None:
                        c_count_dict[C_nonmono] = 0
                    c_count_dict[C_nonmono] +=1

                for neighbor_info in problem.get_neighbors(current_state_bwd):
                    # Handle cases where get_neighbors might return just state or (state, move_info)
                    if isinstance(neighbor_info, tuple) and len(neighbor_info) >= 1:
                        neighbor_state = neighbor_info[0]
                        move_info = neighbor_info[1] if len(neighbor_info) > 1 else None
                    else:
                        neighbor_state = neighbor_info
                        move_info = None
                    #if neighbor_state in closed_bwd: continue
                    
                    cost = problem.get_cost(current_state_bwd, neighbor_state, move_info) 
                    tentative_g_score = current_g_bwd + cost 

                    # Check whether current heuristic is consistent: if h(n) > cost(n, n') + h(n')
                    if self.priority_key == 'f' and h_consistent:
                        h_score = problem.heuristic(neighbor_state, backward=True)
                        if current_h > cost + h_score + 1e-6:
                            status += f" Inconsistent heuristic detected (bwd)."
                            h_consistent = False

                    prior_g = g_score_bwd.get(neighbor_state, float('inf'))
                    if tentative_g_score < prior_g:
                        came_from_bwd[neighbor_state] = current_state_bwd 
                        g_score_bwd[neighbor_state] = tentative_g_score
                        h_score = problem.heuristic(neighbor_state, backward=True)
                        prior_f = prior_g + h_score   # NOTE: heuristic must always return same value for the same state otherwise must store past heuristics
                        frontier_bwd.push(  neighbor_state, 
                                            frontier_bwd.calc_priority(g=tentative_g_score, h=h_score), 
                                            frontier_bwd.calc_tiebreak1(g=tentative_g_score, h=h_score),
                                            prior_f, prior_g)  

            
        end_time = time.time()
        max_ram = round(start_ram - min(min_ram, util.get_available_ram()), 2)

        image_file = 'no file'
        if not status:
            status = "Completed."
        if priority_diminished > 0:
            status += f" Priority diminished count:{priority_diminished}."
        if stale_count > 0:
            status += f" Stale count:{stale_count}."
        if found_goal_count > 0:
            status += f" Found goal {found_goal_count} times."
        if U_update_count > 0:
            status += f" Updated U {U_update_count} times."
        nodes_expanded_below_cstar_auto = -1
        if len(c_count_dict) > 0:
            #status += f" c_count_dict len:{len(c_count_dict)}"
            nodes_expanded_below_cstar_auto = sum(c_count_dict[p] for p in c_count_dict if p < U)

        print(status)

        if meeting_node:
            path = reconstruct_bidirectional_path(came_from_fwd, came_from_bwd, start_node, goal_node, meeting_node)
            if not path:
                status += " Path too long to reconstruct."
            if self.visualise and hasattr(problem, 'visualise'):
                image_file = problem.visualise(path=path, path_type=self._str_repr, 
                                            meeting_node=meeting_node, visited_fwd=set(g_score_fwd.keys()), visited_bwd=set(g_score_bwd.keys()), 
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
            return {"path": path, "cost": final_reported_cost, "nodes_expanded": nodes_expanded,  
                    "nodes_expanded_below_cstar": nodes_expanded_below_cstar, "nodes_expanded_below_cstar_auto": nodes_expanded_below_cstar_auto,
                    "time": end_time - start_time, "optimal": optimality_guaranteed, "visual": image_file,
                    "max_heap_len": max_heap_size_combined, 
                    "g_score_len": len(g_score_fwd)+len(g_score_bwd),
                    "max_ram_taken": max_ram,
                    "status": status,
                    "prob_str": problem.prob_str, "heur": problem.h_str, "degr": problem.degradation, "admiss": problem.admissible, "costtype": problem.cost_type, "CS_pre": problem.cstar,
                    "nodes_sec": nodes_expanded / (end_time - start_time) if end_time > start_time else 0,}


        status += " No path found."
        return {"path": None, "cost": -1, "nodes_expanded": nodes_expanded, 
                "nodes_expanded_below_cstar": nodes_expanded_below_cstar, "nodes_expanded_below_cstar_auto": nodes_expanded_below_cstar_auto,
                "time": end_time - start_time, "optimal": optimality_guaranteed, "visual": image_file,
                "max_heap_len": max_heap_size_combined, 
                "g_score_len": len(g_score_fwd)+len(g_score_bwd),
                "max_ram_taken": max_ram,
                "status": status,
                "prob_str": problem.prob_str, "heur": problem.h_str, "degr": problem.degradation, "admiss": problem.admissible, "costtype": problem.cost_type, "CS_pre": problem.cstar,
                "nodes_sec": nodes_expanded / (end_time - start_time) if end_time > start_time else 0,}



    def __str__(self): # enable str(object) to return algo name
        return self._str_repr


class bd_lb_search:
    """ Performs MEP-based Bidirectional search 
    Handles variable costs.
    if visualise is True and problem supports it, will output visualisation to a subdir off the problem input dir.
    """
    def __init__(self, tb_dir='NBS', tb_select='F', tb_order='NONE', version='A', min_edge_cost=1.0, switch_after_U_set=False,
                 visualise=True, visualise_dirname = '', min_ram=2.0, timeout=30.0, data_struct='P', algo_name='', rust=True,
                 bpmx1=False):
        """
        visualise: If True, will output a visualisation of the search to a subdir off the output dir.
        tb_dir: Strategy for determining direction(s) to search in. 
        tb_select: Strategy for determining node(s) to select for expansion.
        tb_order: Strategy for determining order of expansion eg 'R', 'FIFO', 'LIFO', or 'NONE' = heap/bucket ordering.
        min_ram: Minimum RAM in GB to keep available during search. If RAM goes below this, the search will (sometimes) stop but in practice Python may sometimes grab all mem and the os will kill the process before this condition fires.
        timeout: Timeout in minutes for the search. If the search takes longer than this, it will stop.

        tb_dir - strategy for direction choice: 

        'NBS': Always expand in both directions, 
        'F'/'B': Forward only, backward only
        'A': expand in alternating direction to past time, 
        'P': Pohl: expand direction based on smallest cardinality of open lists, 
        'R': expand in a random direction, 
        'G': expand direction based on lowest expandable g in open lists, 
        'S': expand direction based on which READY_d has smallest expandable |glevel| of any glevel, 
        'S0': smallest |glevel| of the lowest glevel in Fwd and Bwd
        'SM': smallest |glevel| of any glevel in MWVC in Fwd and Bwd
        'SM0': DVCBS: smallest |glevel| of lowest glevel in MWVC in Fwd and Bwd,
        'SB': expand direction based on which READY_d has smallest expandable |g-f bucket| 
        'SBM0': expand direction based on which READY_d has smallest expandable |g-f bucket| in lowest glevel in MWVC
        'EC': Expand direction based on which READY_d has glevel with largest edge count ie is connected with most glevels in other direction
        'LN': Vidal-like: Expand direction based on which READY_d has glevel connected with largest node count in other direction 
        'LN0': Vidal-like: Expand direction based on which READY_d has lowest glevel connected with largest node count in other direction
        'LM': Expand direction based on which READY_d has glevel in MWVC connected with largest node count in other direction 
        'LM0': Expand direction based on which READY_d has lowest glevel in MWVC connected with largest node count in other direction
        'GBF': expand direction that has a glevel satisfying g_D + max_g_expanded_OppositeD + EPS <= GLB

        
        tb_select - strategy for selecting node(s) to expand in selected direction(s) from Ready_d:

        'F':   select single node in first bucket i.e. a node in bucket with lowest g and lowest f 
        'FHF': select single node in highest f in lowest g
        'FHG': select single node in lowest f in highest g
        'B': entire first bucket i.e. bucket with lowest g and lowest f
        'R': random node from lowest g
        'RA': random node from all expandable nodes
        'ALL': all expandable buckets
        'GBF': expand all glevels satisfying g_D + max_g_expanded_OppositeD + EPS <= GLB
        'SG': smallest expandable glevel
        'SM': smallest expandable glevel in MWVC
        'SLG': smallest bucket in lowest g (eg use with tb_dir SBM0 that will select direction with smallest bucket in lowest g that is in a MWVC)
        'LG': lowest glevel - frequently used in conjunction with tb_dir ending in 0
        'HG': highest glevel
        'S': DVCBS: smallest glevel of the minimum expandable glevel that is in any MVC of expandable_f X expandable_b
        'SB': smallest bucket of any expandable bucket - with tiebreak towards highest g
        'SBL': smallest bucket of any expandable bucket - with tiebreak towards lowest g
        'EC': Expand glevel with largest edge count ie. is connected with most glevels in other direction
        'LN': Vidal-like: Expand glevel with largest node count over connected glevels in other direction
        'LM': Vidal-like: Expand glevel in MWVC with largest node count over connected glevels in other direction


        tb_order - determines the order for expanding selected nodes in selected direction if more than one node:

        'R': random
        'FIFO' / 'LIFO' 
        'NONE' - no explicit ordering applied
        
        """
        self.timeout = timeout
        self.min_ram = min_ram
        self.visualise = visualise
        self.visualise_dirname = visualise_dirname
        self.min_edge_cost = min_edge_cost  # Minimum edge cost to consider in the search
        self.switch_after_U_set = switch_after_U_set

        self.version = version.upper()  # Per Shperberg 2019 Pseudocode 'A' for the "Return ALL paths" version (although we only return first path) or 'F' for the "Return first path version"
        if self.version not in ['A', 'F']:
            raise ValueError(f"ERROR: Invalid version: '{self.version}'. Must be 'A' or 'F'.")

        self.tb_dir = tb_dir.upper()
        if self.tb_dir not in ['NBS', 'F', 'B', 'A', 'P', 'R', 'G', 'S', 'S0', 'SM', 'SM0', 'SB', 'SBM0', 'EC', 'LN', 'LN0', 'LM', 'LM0', 'GBF']:
            raise ValueError(f"ERROR: Invalid tb_dir: '{self.tb_dir}'. Must be 'NBS', 'F', 'B', 'A', 'P', 'R', 'G', 'S', 'S0', 'SM', 'SM0', 'SB', 'SBM0', 'EC', 'LN', 'LN0', 'LM', 'GBF'.")

        self.tb_select = tb_select.upper()
        if self.tb_select not in ['F', 'FHF', 'FHG', 'B', 'R', 'RA', 'ALL', 'GBF', 'SLG', 'LG', 'HG', 'SG', 'SM', 'SB', 'SBL', 'EC', 'LN', 'LM']:
            raise ValueError(f"ERROR: Invalid tb_dir: '{self.tb_select}'. Must be 'F', 'FHF', 'FHG', 'B', 'R', 'RA', 'ALL', 'GBF', 'SLG', 'LG', 'HG', 'SG', 'SM', 'SB', 'SBL', 'EC' or 'LN'.")

        self.tb_order = tb_order.upper()
        if self.tb_order not in ['R', 'FIFO', 'LIFO', 'NONE']:
            raise ValueError(f"ERROR: Invalid tb_order: '{self.tb_order}'. Must be 'R', 'FIFO', 'LIFO', or 'NONE'.")

        self.do_calc_expandable = False
        self.do_mwvc = False   # Flag for tiebreakers requiring Min Weighted Vertex Cover (MWVC) to be calculated. This incurs overhead so not doing it if unnecessary.
        if self.tb_dir in ['S', 'S0', 'SM', 'SM0', 'SB', 'SBM0', 'EC', 'LN', 'LN0', 'LM', 'LM0'] or self.tb_select not in ['F', 'FHF', 'B', 'R', 'LG']:
            self.do_calc_expandable = True   # Flag for tiebreakers requiring frontier.calc_expandable() to be run. This incurs overhead so not doing it if unnecessary.
        if self.tb_dir in ['LM', 'LM0', 'SM', 'SM0', 'SBM0'] or self.tb_select in ['LM', 'SM']:
            self.do_mwvc = True

        self.data_struct = data_struct.upper()  # 'P' for PriorityQueue, 'B' for Buckets
        if self.data_struct not in ['P', 'B']:
            raise ValueError(f"ERROR: Invalid data_struct: '{self.data_struct}'. Must be 'P' or 'B'.")
        
        if self.data_struct == 'P':
            if self.tb_select != 'F':
                raise ValueError(f"ERROR: Invalid data_struct '{self.data_struct}'. Must be 'B' for any tb_select other than 'F'.")
            if self.tb_dir not in ['NBS', 'F', 'B', 'A', 'P', 'R', 'G']:
                raise ValueError(f"ERROR: Invalid data_struct '{self.data_struct}'. Must be 'B' for any tb_select other than 'NBS', 'F', 'B', 'A', 'P', 'R', or 'G'.")

        if self.tb_order == 'LIFO':
            self.increment_tb1 = -1
        elif self.tb_order == 'FIFO':
            self.increment_tb1 = 1
        else:
            self.increment_tb1 = 0
        self.rand_upper_bound = 100000000000
        self.ordering = 0

        self.rust = rust
        self.bpmx1 = bpmx1

        self.algo = algo_name
        self._str_repr = f"BiDirLBPairs-{self.algo}-dir{self.tb_dir}-sel{self.tb_select}-ord{self.tb_order}-ver{self.version}-ds{self.data_struct}-eps{self.min_edge_cost}-uf{self.switch_after_U_set}-bpmx1{self.bpmx1}-rust{self.rust}"


    def search(self, problem):
        """ Run the search on a problem instance and return dict of results."""
        optimality_guaranteed = problem.optimality_guaranteed

        self.ordering = 0
        if self.rust:
            import rust_utils
            nodes_fwd = rust_utils.RustDict()   # dictionary key state to store named tuple of (g, h, parent)
            nodes_bwd = rust_utils.RustDict()   # dictionary key state to store named tuple of (g, h, parent)
            Node = rust_utils.NodeData
        else:
            nodes_fwd = {}
            nodes_bwd = {}
            Node = data_structures.NodeData

        start_time = time.time()
        start_node = problem.initial_state()
        goal_node = problem.goal_state()
        if problem.is_goal(start_node): return {"path": [start_node], "cost": 0, "nodes_expanded": 0, "nodes_expanded_below_cstar": 0,
                                                "time": 0, "optimal": optimality_guaranteed, 'visual': 'no file', "max_heap_size": 0}

        frontiers = data_structures.LBPairs(version=self.version, min_edge_cost=self.min_edge_cost, 
                                            data_struct=self.data_struct, 
                                            tb_dir=self.tb_dir, tb_select=self.tb_select, tb_order=self.tb_order)
        h_initial = problem.heuristic(start_node)
        frontiers.push('F', [0, self.calc_ordering(), start_node], h_initial) # Push with Direction, (g, fifolifoval, state) and priority (f)
        nodes_fwd[start_node] = Node(0, h_initial, None)  # dict stores named tuple (g, h, parent) for each state

        h_goal = problem.heuristic(goal_node, backward=True)
        frontiers.push('B', [0, self.calc_ordering(), goal_node], h_goal) # Push with Direction, (g, fifolifoval, state) and priority (f)
        nodes_bwd[goal_node] = Node(0, h_goal, None)  # dict stores named tuple (g, h, parent) for each state

        nodes_expanded = 0
        GLB_forcemono = 0                                  # Force monotonicity of GLB for c_count_dict to count nodes expanded below cstar correctly
        GLB = 0   #min(h_initial, h_goal) #-1.0            # Current lowest cost on either frontier ie  min(fminF, fminB, gminF+gminb+eps)
        U = float('inf')    # Current lowest cost of path found in either direction
        if hasattr(problem, "cstar"):
            cstar = problem.cstar
        else:
            cstar = None
        nodes_expanded_below_cstar = 0
        c_count_dict = {}   # count dict where copy of Glb/c forced monotonic to count nodes expanded below cstar correctly
        meeting_node = None
        max_heap_size_combined = 0
        i = 0
        checkmem = 1000
        status = ""
        stale_count = 0
        found_goal_count = 0
        U_update_count = 0
        h_consistent = True  # optionally check the consistency of the heuristic if running A* (not exhaustive)
        h_admissable = True  # optionally check the admissability of the heuristic if running A* and cstar is supplied (not exhaustive)
        priority_diminished = 0
        start_ram = util.get_available_ram()
        min_ram = start_ram
        num_expansions_fwd = 0  # Num of expansions in fwd direction this iteration
        num_expansions_bwd = 0  # Num of expansions in bwd direction this iteration - if fwd+bwd = 0 then terminate with error as inf loop
        num_iter_no_expansions = 0  # Number of iterations with no expansions in either direction - error if >= 2 meaning a potential fwd only iter + bwd only iter failed
        force_low_tiebreak = False  # only improves expansion count slightly (where expanding > 1 node in a direction) so disabling for now
        switch_after_U_set = self.switch_after_U_set # change to tb_select='F' after U < inf: disabling due to this taking excessive time compared to not using it in some domains eg pancake

        while not frontiers.forward.isEmpty() and not frontiers.backward.isEmpty():
            curr_heap_size = frontiers.get_max_heap_size()
            if curr_heap_size > max_heap_size_combined:
                max_heap_size_combined = curr_heap_size
            if (time.time()-start_time)/60.0 > self.timeout:
                status += f"Timeout after {(time.time()-start_time)/60:.4f} mins."
                break
            if i % checkmem == 0:
                min_ram = min(min_ram, util.get_available_ram())
                if min_ram < self.min_ram:
                    status += f"Out of RAM ({min_ram:.4f}GB remaining)."
                    break
            i += 1

            found, new_GLB = frontiers.prepare_expandable(GLB)
            if not found:  # If no expandable nodes, we are done - never happens in current test domains
                status += f"Completed. No expandable nodes found. Old GLB:{GLB} New GLB:{new_GLB} U:{U}."
                break

            if new_GLB + 1e-6 < GLB:
                priority_diminished += 1
            GLB = new_GLB    #<- For inconsistent heuristics concerned there could a corner case where diminished_priority state led to better soln but we terminate prematurely so not forcing GLB to be monotonic (even though none of my tests with consistent/inconsistent heuristics actually cause non-monoticity...)
            GLB_forcemono = max(GLB_forcemono, GLB)  # Force monotonicity of "shadow GLB" for c_count_dict to guarantee count of nodes expanded below cstar is correct 

            if GLB >= U: # If the estimated lowest cost path on frontier (lowest lb(u,v)) is greater or equal cost than the best path found, stop
                status += f"Completed. Termination condition GLB ({GLB}) >= U ({U}) met."
                break

            if self.do_calc_expandable:
                frontiers.calc_expandable(self.do_mwvc)
            fwd, bwd = frontiers.calc_direction()
            num_expansions_fwd = 0  # Num of expansions in fwd direction this iteration
            num_expansions_bwd = 0  # Num of expansions in bwd direction this iteration - if fwd+bwd = 0 then terminate with error as inf loop


            # --- Forward Step ---
            if not frontiers.forward.isEmpty() and fwd:
                #g, f, ordering, current_state_fwd = frontiers.pop('F', item_only=False)  # g, f, ordering, state
                expand_list = frontiers.select_and_order('F')
                num_expansions_fwd = len(expand_list)
                # NOTE: DVCBS HOG2 code has optimization s.t if path found between any u in fwd expand_list and v in bwd expand list then terminate with optimal found. For algos expanding many nodes in both directions in same iter this will reduce the expansion count by len(fwd expand_list)+len(bwd expand list up to point path found) but at the cost of iterating through both expand_lists to do the check.
                for (ordering, g, f, current_state) in expand_list:
                    current_node = nodes_fwd[current_state] 
                    current_g = round(current_node.g, 2)
                    current_h = current_node.h
                    if h_admissable:
                        if cstar and f > cstar + 1e-6:
                            status += f" Possible inadmissable heuristic detected (Fwd) GLB:{GLB} f:{f} h:{current_h} g:{g} cstar:{cstar} state:{current_state}."
                            h_admissable = False
                    
                    if current_g + 1e-6 < g:    # Our Ready and Wait implementations mark existing entries stale before adding duplicates so this condition will only trigger rarely if the g on the frontier doesn't match prior_g
                        stale_count += 1
                        continue

                    if f >= U:      # optimisation from both NBS and DVCBS HOG2 code
                        continue

                    nodes_expanded += 1
                    if cstar and GLB_forcemono < cstar:
                        nodes_expanded_below_cstar += 1
                    if c_count_dict.get(GLB_forcemono) is None:
                        c_count_dict[GLB_forcemono] = 0
                    c_count_dict[GLB_forcemono] +=1

                    if g > frontiers.forward_max_g_expanded:
                        frontiers.forward_max_g_expanded = g

                    neighbors_list = []
                    best_h = 0
                    for neighbor_info in problem.get_neighbors(current_state):
                        if isinstance(neighbor_info, tuple) and len(neighbor_info) >= 1:  # Handle cases where get_neighbors might return just state or (state, move_info)
                            neighbor_state = neighbor_info[0]
                            move_info = neighbor_info[1] if len(neighbor_info) > 1 else None
                        else:
                            neighbor_state = neighbor_info
                            move_info = None

                        cost = problem.get_cost(current_state, neighbor_state, move_info)
                        tentative_g_score = round(current_g + cost, 2)
                        if neighbor_state in nodes_fwd:
                            h_score = round(nodes_fwd[neighbor_state].h, 3)
                        else:
                            h_score = round(problem.heuristic(neighbor_state), 3)
                        if self.bpmx1:
                            best_h = max(best_h, round(h_score - cost, 3))
                        neighbors_list.append( {'state': neighbor_state, 'g': tentative_g_score , 'h': h_score, 'cost': cost} )

                    if self.bpmx1:
                        if best_h > current_h:  # a child h > parent h - cost  so increase parent h
                            current_h = best_h
                            current_node = Node(current_g, best_h, current_node.parent)
                            nodes_fwd[current_state] = current_node
                        else:
                            best_h = current_h  # parent h > child h - cost so use parent h to potentially increase child h

                    for neighbor in neighbors_list:
                        neighbor_state = neighbor['state']
                        tentative_g_score = neighbor['g']

                        if tentative_g_score >= U:   # Optimisation in NBS and DVCBS HOG2 code
                            continue

                        neighbor_node = nodes_fwd.get(neighbor_state)
                        if neighbor_node is None:
                            prior_g = float('inf')
                        else:
                            prior_g = round(neighbor_node.g, 2)

                        if tentative_g_score < prior_g:
                            h_score = neighbor['h']  #round(problem.heuristic(neighbor_state), 3) 
                            if self.bpmx1:
                                h_score = max(h_score, round(best_h - neighbor['cost'], 3))   # if parent h - cost > child h then increase child h
                            neighbor_node = Node(tentative_g_score, h_score, current_state)
                            if h_consistent: # Check whether current heuristic is consistent: if h(n) > cost(n, n') + h(n')
                                if current_h > neighbor['cost'] + h_score + 1e-6:
                                    status += f" Inconsistent heuristic detected. parent h {current_h} > edgecost {neighbor['cost']} + child h {h_score} + 1e-6."
                                    h_consistent = False
                            nodes_fwd[neighbor_state] = neighbor_node  # Add/Update the node in the Nodes dict
                            if prior_g != float('inf') or tentative_g_score+h_score < U:       # Optimisation in NBS and DVCBS HOG2 code
                                #force_low_tiebreak = False          # test optimisation not in NBS or DVCBS HOG2 code - make a small positive difference on algos that expand levels
                                #if neighbor_state in g_score_bwd: 
                                #    current_path_cost = g_score_bwd[neighbor_state] + tentative_g_score
                                #    if current_path_cost < U:
                                #        force_low_tiebreak = True
                                prior_f = prior_g + h_score
                                frontiers.push('F', [tentative_g_score, self.calc_ordering(force_low_tiebreak), neighbor_state], 
                                                tentative_g_score+h_score, 
                                                prior_f, prior_g)  

                        if neighbor_state in nodes_bwd: 
                            current_path_cost = round(nodes_bwd[neighbor_state].g, 2) + tentative_g_score
                            found_goal_count += 1
                            if current_path_cost < U:
                                U = current_path_cost
                                meeting_node = neighbor_state
                                U_update_count += 1
                                if switch_after_U_set:
                                    self.tb_select = 'F'  # If path found, reset tb_select to F so that next iteration will recalc CLB after expanding each node
                                    frontiers.tb_select = 'F'  # Reset tb_select to F so that next iteration will recalc CLB after expanding each node
                                    if self.tb_order in ['FIFO', 'LIFO']:
                                        self.tb_order = 'NONE'  # For bucket implementation, tb_select='F', tb_order FIFO or LIFO is slow so reset to NONE
                                        frontiers.tb_order = 'NONE'
                    # end of for state in expand_list loop - after each expansion check whether U has diminished to current GLB - DVCBS HOG2 code optimisation
                    if GLB >= U: # If the estimated lowest cost path on frontier is greater cost than the best path found, stop
                        status += f"Completed in FWD Exp. Termination condition GLB ({GLB}) >= U ({U}) met."
                        break  # break for state in expand_list
            if GLB >= U: # If the estimated lowest cost path on frontier is greater cost than the best path found, stop
                break # break main loop


            # --- Backward Step ---
            if not frontiers.backward.isEmpty() and bwd:
                #g, f, ordering, current_state_bwd = frontiers.pop('B', item_only=False)
                expand_list = frontiers.select_and_order('B')
                num_expansions_bwd = len(expand_list)
                for (ordering, g, f, current_state) in expand_list:
                    current_node = nodes_bwd[current_state] 
                    current_g = round(current_node.g, 2)
                    current_h = current_node.h
                    if h_admissable:
                        if cstar and f > cstar + 1e-6:
                            status += f" Possible inadmissable heuristic detected (Bwd) GLB:{GLB} f:{f} h:{current_h} g:{g} cstar:{cstar} state:{current_state}."
                            h_admissable = False
                    
                    if current_g + 1e-6 < g:    # Our Ready and Wait implementations mark existing entries stale before adding duplicates so this condition will only trigger if there is an inconsistent heuristic and then only rarely
                        stale_count += 1
                        continue

                    if f >= U:      # optimisation from both NBS and DVCBS HOG2 code
                        continue

                    nodes_expanded += 1
                    if cstar and GLB_forcemono < cstar:
                        nodes_expanded_below_cstar += 1
                    if c_count_dict.get(GLB_forcemono) is None:
                        c_count_dict[GLB_forcemono] = 0
                    c_count_dict[GLB_forcemono] +=1

                    if g > frontiers.backward_max_g_expanded:
                        frontiers.backward_max_g_expanded = g

                    neighbors_list = []
                    best_h = 0
                    for neighbor_info in problem.get_neighbors(current_state):
                        if isinstance(neighbor_info, tuple) and len(neighbor_info) >= 1:  # Handle cases where get_neighbors might return just state or (state, move_info)
                            neighbor_state = neighbor_info[0]
                            move_info = neighbor_info[1] if len(neighbor_info) > 1 else None
                        else:
                            neighbor_state = neighbor_info
                            move_info = None

                        cost = problem.get_cost(current_state, neighbor_state, move_info)
                        tentative_g_score = round(current_g + cost, 2)
                        if neighbor_state in nodes_bwd:
                            h_score = round(nodes_bwd[neighbor_state].h, 3)
                        else:
                            h_score = round(problem.heuristic(neighbor_state, backward=True), 3)
                        if self.bpmx1:
                            best_h = max(best_h, round(h_score - cost, 3))
                        neighbors_list.append( {'state': neighbor_state, 'g': tentative_g_score , 'h': h_score, 'cost': cost} )

                    if self.bpmx1:
                        if best_h > current_h:  # a child h > parent h - cost  so increase parent h
                            current_h = best_h
                            current_node = Node(current_g, best_h, current_node.parent)
                            nodes_bwd[current_state] = current_node
                        else:
                            best_h = current_h  # parent h > child h - cost so use parent h to potentially increase child h

                    for neighbor in neighbors_list:
                        neighbor_state = neighbor['state']
                        tentative_g_score = neighbor['g']
                        if tentative_g_score >= U:   # Optimisation in NBS and DVCBS HOG2 code
                            continue
                        
                        neighbor_node = nodes_bwd.get(neighbor_state)
                        if neighbor_node is None:
                            prior_g = float('inf')
                        else:
                            prior_g = round(neighbor_node.g, 2)

                        if tentative_g_score < prior_g:
                            h_score = neighbor['h']  #round(problem.heuristic(neighbor_state), 3) 
                            if self.bpmx1:
                                h_score = max(h_score, round(best_h - neighbor['cost'], 3))   # if parent h - cost > child h then increase child h
                            neighbor_node = Node(tentative_g_score, h_score, current_state)
                            if h_consistent: # Check whether current heuristic is consistent: if h(n) > cost(n, n') + h(n')
                                if current_h > neighbor['cost'] + h_score + 1e-6:
                                    status += f" Inconsistent heuristic detected. parent h {current_h} > edgecost {neighbor['cost']} + child h {h_score} + 1e-6."
                                    h_consistent = False
                            nodes_bwd[neighbor_state] = neighbor_node  # Add/Update the node in the Nodes dict
                            if prior_g != float('inf') or tentative_g_score+h_score < U:   # Optimisation in NBS and DVCBS HOG2 code
                                #force_low_tiebreak = False          # test optimisation not in NBS or DVCBS HOG2 code
                                #if neighbor_state in g_score_fwd: 
                                #    current_path_cost = g_score_fwd[neighbor_state] + tentative_g_score
                                #    if current_path_cost < U:
                                #        force_low_tiebreak = True
                                prior_f = prior_g + h_score   # NOTE: heuristic must always return same value for the same state otherwise must store past heuristics
                                frontiers.push('B', [tentative_g_score, self.calc_ordering(force_low_tiebreak), neighbor_state], 
                                                tentative_g_score+h_score,
                                                prior_f, prior_g)

                        if neighbor_state in nodes_fwd: 
                            current_path_cost = round(nodes_fwd[neighbor_state].g, 2) + tentative_g_score
                            found_goal_count += 1
                            if current_path_cost < U:
                                U = current_path_cost
                                meeting_node = neighbor_state
                                U_update_count += 1
                                if switch_after_U_set:
                                    self.tb_select = 'F'  # If path found, reset tb_select to F so that next iteration will recalc CLB after expanding each node
                                    frontiers.tb_select = 'F'  # Reset tb_select to F so that next iteration will recalc CLB after expanding each node
                                    if self.tb_order in ['FIFO', 'LIFO']:
                                        self.tb_order = 'NONE'  # For bucket implementation, tb_select='F', tb_order FIFO or LIFO is slow so reset to NONE
                                        frontiers.tb_order = 'NONE'

                    # end of for state in expand_list loop - after each expansion check whether U has diminished to current GLB - DVCBS HOG2 code optimisation
                    if GLB >= U: # If the estimated lowest cost path on frontier is greater cost than the best path found, stop
                        status += f"Completed in BWD Exp. Termination condition GLB ({GLB}) >= U ({U}) met."
                        break  # break for state in expand_list
            if GLB >= U: # If the estimated lowest cost path on frontier is greater cost than the best path found, stop
                break # break main loop
            if num_expansions_fwd + num_expansions_bwd == 0:  # If no nodes expanded in either direction 
                num_iter_no_expansions += 1
                if num_iter_no_expansions >= 2:  # If no nodes expanded in either direction for 2 iterations, terminate with error (2 iterations to allow for an attempt at each direction with eg alter)
                    status += f"ERROR: Terminating since no nodes expanded in either direction over 2 iterations! GLB:{GLB} U:{U}."
                    break
            else:
                num_iter_no_expansions = 0


        end_time = time.time()
        max_ram = round(start_ram - min(min_ram, util.get_available_ram()), 2)
        max_bucket_size, distinct_f, distinct_g = frontiers.get_max_bucket_stats()

        image_file = 'no file'
        if not status:
            status = "Completed."
        if priority_diminished > 0:
            status += f" Priority diminished count:{priority_diminished}."
        if stale_count > 0:
            status += f" Stale count:{stale_count}."
        if found_goal_count > 0:
            status += f" Paths met {found_goal_count} times."
        if U_update_count > 0:
            status += f" Updated U {U_update_count} times."
        nodes_expanded_below_cstar_auto = -1
        if len(c_count_dict) > 0:
            nodes_expanded_below_cstar_auto = sum(c_count_dict[glb] for glb in c_count_dict if glb < U)

        print(status)

        if meeting_node:
            if str(problem).startswith('GRID-'):
                convert_func = util.decode_numbers
                convert_func_back = util.encode_numbers
            else:
                convert_func = tuple
                convert_func_back = tuple
            path = reconstruct_bidirectional_path(nodes_fwd, nodes_bwd, start_node, goal_node, meeting_node, convert_func=convert_func)
            if not path:
                status += " Path too long to reconstruct."
            if self.visualise and hasattr(problem, 'visualise'):
                image_file = problem.visualise(path=path, path_type=self._str_repr, 
                                            meeting_node=meeting_node, visited_fwd=set(nodes_fwd.keys()), visited_bwd=set(nodes_bwd.keys()), 
                                            visualise_dirname=self.visualise_dirname)
                if not image_file: image_file = 'no file'
            final_cost = -1
            recalculated_cost = 0
            cost_mismatch = False
            if path:  # check path reconstruction - NOT NEEDED post debugging!!
                try:
                    for i in range(len(path) - 1):
                        recalculated_cost += problem.get_cost(convert_func_back(path[i]), convert_func_back(path[i+1])) # Use fallback cost
                    final_cost = recalculated_cost
                    if abs(final_cost - U) > 1e-6: 
                        cost_mismatch = True
                except Exception as e:
                    print(f"Error recalculating bidirectional path cost: {e}")
                    final_cost = U 
            if cost_mismatch: 
                print(f"Warning: Bidirectional cost mismatch! PathRecalc={final_cost}, SearchCost={U}")
            final_reported_cost = U # Report cost found by search
            return {"path": path, "cost": final_reported_cost, "nodes_expanded": nodes_expanded,  
                    "nodes_expanded_below_cstar": nodes_expanded_below_cstar, "nodes_expanded_below_cstar_auto": nodes_expanded_below_cstar_auto,
                    "time": end_time - start_time, "optimal": optimality_guaranteed, "visual": image_file,
                    "max_heap_len": max_heap_size_combined, 
                    "g_score_len": len(nodes_fwd)+len(nodes_bwd),
                    "max_ram_taken": max_ram,
                    "status": status, 
                    "prob_str": problem.prob_str, "heur": problem.h_str, "degr": problem.degradation, "admiss": problem.admissible, "costtype": problem.cost_type, "CS_pre": problem.cstar,
                    "nodes_sec": nodes_expanded / (end_time - start_time) if end_time > start_time else 0,
                    "max_bucket_size": max_bucket_size, "distinct_f": distinct_f, "distinct_g": distinct_g,}


        status += " No path found."
        return {"path": None, "cost": -1, "nodes_expanded": nodes_expanded, 
                "nodes_expanded_below_cstar": nodes_expanded_below_cstar, "nodes_expanded_below_cstar_auto": nodes_expanded_below_cstar_auto,
                "time": end_time - start_time, "optimal": optimality_guaranteed, "visual": image_file,
                "max_heap_len": max_heap_size_combined, 
                "g_score_len": len(nodes_fwd)+len(nodes_bwd),
                "max_ram_taken": max_ram,
                "status": status,
                "prob_str": problem.prob_str, "heur": problem.h_str, "degr": problem.degradation, "admiss": problem.admissible, "costtype": problem.cost_type, "CS_pre": problem.cstar,
                "nodes_sec": nodes_expanded / (end_time - start_time) if end_time > start_time else 0,
                "max_bucket_size": max_bucket_size, "distinct_f": distinct_f, "distinct_g": distinct_g,}


    def calc_ordering(self, force_low_tiebreak=False):
        """ Calculate fifo/lifo ordering """
        if force_low_tiebreak:  # Force node to be expanded first of nodes in expand_list eg if we know its a meeting_node
            self.ordering = -1
        if self.tb_order == 'NONE':
            return 0
        elif self.tb_order in ['FIFO', 'LIFO']:
            self.ordering += self.increment_tb1
        elif self.tb_order == 'R':
            self.ordering = random.randint(0, self.rand_upper_bound)
        return self.ordering

    def __str__(self): # enable str(object) to return algo name
        return self._str_repr



def reconstruct_bidirectional_path(nodes_fwd, nodes_bwd, start_state, goal_state, meeting_node, convert_func=tuple):
    """Reconstructs path for bidirectional search."""
    path1 = []
    curr = meeting_node
    limit = 10000000 
    count = 0
    while curr is not None: 
        path1.append(convert_func(curr))
        curr = nodes_fwd[curr].parent
        #curr = came_from_fwd.get(curr)
        count += 1
        if count > limit: print("Error: Path fwd reconstruction exceeded limit."); return None
    path1.reverse() 
    
    path2 = []

    curr = nodes_bwd[meeting_node].parent 
    #curr = came_from_bwd.get(meeting_node) 
    count = 0
    while curr is not None: 
        path2.append(convert_func(curr))
        curr = nodes_bwd[curr].parent
#         curr = came_from_bwd.get(curr)
        count += 1
        if count > limit: print("Error: Path bwd reconstruction exceeded limit."); return None
         
    return path1 + path2


