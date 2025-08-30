"""
Unidirectional Searches

Dijkstra / Uniform Cost Search  (g only)
Greedy Best First Search        (h only)
A* Search                    (f = g + h)    
"""
import time
#import rust_utils
from sortedcontainers import SortedDict
import util
import data_structures


algo_name_map = {'g': "UniformCost", 'h': "GreedyBestFirst", 'f': "Astar"}


class generic_search:
    """
    Performs a generic unidirectional heuristic search.
    Priority can be based on 'g', 'h', or 'f' = g+h. Handles variable costs.
    if visualise is True and problem supports it, will output visualisation to a subdir off the problem input dir.
    """
    def __init__(self, priority_key='f', tiebreaker1='-g', tiebreaker2 = 'NONE', 
                 visualise=True, visualise_dirname='', min_ram=2.0, timeout=30.0, min_edge_cost=0.0, 
                 rust=False, bpmx1=False):
        """
        priority_key: 'g', 'h', or 'f' = g+h. Determines the priority of the nodes in the search.
        visualise: If True, will output a visualisation of the search to a subdir off the output dir.
        tiebreaker1/2: 1st and 2nd level Tiebreaker for the priority queue. Can be eg 'g', 'FIFO', 'LIFO', or 'NONE' for no tiebreaker = heap ordering.
        min_ram: Minimum RAM in GB to keep available during search. If RAM goes below this, the search will (sometimes) stop but in practice Python may sometimes grab all mem and the os will kill the process before this condition fires.
        timeout: Timeout in minutes for the search. If the search takes longer than this, it will stop.
        """
        if priority_key not in algo_name_map: raise ValueError(f"priority_key must be in {algo_name_map}")
        self.timeout = timeout
        self.min_ram = min_ram
        self.priority_key = priority_key
        self.visualise = visualise
        self.visualise_dirname = visualise_dirname
        self.tiebreaker1 = tiebreaker1  # see calc_tiebreak_val for options
        self.tiebreaker2 = tiebreaker2
        self.min_edge_cost = min_edge_cost  # used for making h = max(h, eps): On std tests using eps=1 decreases expansions below C* but expansions <= C* can actually increase so use with caution
        self.rust = rust
        self.bpmx1 = bpmx1  # Felner et al 2011. Make inconsistent heuristics "more consistent" by propagating h values between parent and children. Here we do the simplest version, BPMX(1), which only propagates one step. Only does anything if priority_key is 'f' or 'h'.
        self._str_repr = f"{algo_name_map[self.priority_key]}-p{self.priority_key}-tb1{self.tiebreaker1}-tb2{self.tiebreaker2}-eps{self.min_edge_cost}-bpmx1{self.bpmx1}-rust{self.rust}"


    def search(self, problem):
        """ Run the search on a problem instance and return dict of results."""
        optimality_guaranteed = (self.priority_key == 'g') or (self.priority_key=='f' and problem.optimality_guaranteed)

        if self.rust:
            import rust_utils
            nodes_fwd = rust_utils.RustDict()   # dictionary key state to store named tuple of (g, h, parent)
            Node = rust_utils.NodeData
        else:
            nodes_fwd = {}
            Node = data_structures.NodeData

        start_time = time.time() 
        start_node = problem.initial_state()
        h_initial = problem.heuristic(start_node) if self.priority_key in ['h', 'f'] else 0
        g_initial = 0

        frontier = data_structures.PriorityQueue(priority_key=self.priority_key, 
                                                 tiebreaker1=self.tiebreaker1, tiebreaker2=self.tiebreaker2) # Priority queue
        frontier.push(start_node, 
                      frontier.calc_priority(g=g_initial, h=h_initial), 0) # Push with priority and tiebreaker1

        nodes_fwd[start_node] = Node(g_initial, h_initial, None)  # dict stores named tuple (g, h, parent) for each state

        nodes_expanded = 0
        C_nonmono = -1
        C = -1.0         # Current lowest cost on frontier
        U = float('inf') # Current lowest cost found for start to goal
        if hasattr(problem, "cstar"):
            cstar = problem.cstar
        else:
            cstar = None
        nodes_expanded_below_cstar = 0
        nodes_expanded_below_cstar_auto = 0
        c_count_dict = {}
        i = 0
        checkmem = 1000
        status = ""
        stale_count = 0
        found_goal_count = 0
        U_update_count = 0
        found_path = False
        h_consistent = True  # optionally check the consistency of the heuristic if running A* (not exhaustive)
        h_admissable = True  # optionally check the admissability of the heuristic if running A* and cstar is supplied (not exhaustive)
        priority_diminished = 0
        start_ram = util.get_available_ram()
        min_ram = start_ram

        while not frontier.isEmpty():
            if (time.time()-start_time)/60.0 > self.timeout:
                status += f"Timeout after {(time.time()-start_time)/60:.4f} mins."
                break
            if i % checkmem == 0:
                min_ram = min(min_ram, util.get_available_ram()) 
                if min_ram < self.min_ram:
                    status += f"Out of RAM ({min_ram:.4f}GB remaining)."
                    break
            i += 1

            current_priority = frontier.peek(priority_only=True) # Peek at the lowest priority element. 

            if current_priority + 1e-6 < C:  # This can happen with inconsistent heuristic
                priority_diminished += 1

            C = current_priority    # C = max(C, current_priority) <- this 'max' works empirically but concerned it *could* fail for inconsistent heuristic where priority diminishes
            C_nonmono = max(C_nonmono, current_priority)  # C_nonmono is used for c_count_dict to count nodes expanded below cstar correctly in the case of inconsistent heuristics

            if C >= U: # If the estimated lowest cost path on frontier is greater cost than the best path found, stop
                found_path = True
                status += f"Completed. Termination condition C ({C}) >= U ({U}) met."
                break

            current_state = frontier.pop(item_only=True) # Pop the state with the lowest priority
            current_node = nodes_fwd[current_state]
            current_g = round(current_node.g, 2)
            if self.priority_key == 'g': 
                current_h = 0
            else: 
                current_h = round(current_node.h, 3)
                if not problem.is_goal(current_state):
                    current_h = max(current_h, self.min_edge_cost)  
                if self.priority_key == 'f' and h_admissable:
                    if cstar and current_g + current_h > cstar + 1e-6:
                        status += f" Inadmissable heuristic detected."
                        h_admissable = False

            g_from_frontier = round(current_priority - current_h, 2)

            # left the check for stale entries, but PriorityQueue now removes duplicates internally..
            if current_g < g_from_frontier:
                stale_count += 1

            nodes_expanded += 1
            if cstar and C_nonmono < cstar:
                nodes_expanded_below_cstar += 1
            if self.priority_key != 'h':
                if c_count_dict.get(C_nonmono) is None:
                    c_count_dict[C_nonmono] = 0
                c_count_dict[C_nonmono] +=1

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
                if self.priority_key == 'g':
                    h_score = 0
                elif neighbor_state in nodes_fwd:
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

                neighbor_node = nodes_fwd.get(neighbor_state)
                if neighbor_node is None:
                    prior_g = float('inf')
                else:
                    prior_g = round(neighbor_node.g, 2)

                at_goal = False
                if problem.is_goal(neighbor_state):  # Works when here
                    at_goal = True 
                    found_goal_count += 1
                    if tentative_g_score < U:
                        U = tentative_g_score
                        found_path = True
                        U_update_count += 1
                        if self.priority_key == 'h':  # BFS is not optimal so may as well end as soon as a path found
                            neighbor_node = Node(tentative_g_score, 0, current_state)
                            nodes_fwd[neighbor_state] = neighbor_node  # Update the node in the dict  
                            status += f"Terminating BFS as path found. U:{U}."
                            break

                if tentative_g_score < prior_g:  #Per Wikipedia citing Russell&Norvig: if a node is reached by one path, removed from openSet, and subsequently reached by a cheaper path, it will be added to openSet again. This is essential to guarantee that the path returned is optimal if the heuristic function is admissible but not consistent. If the heuristic is consistent, when a node is removed from openSet the path to it is guaranteed to be optimal so the test ‘tentative_gScore < gScore[neighbor]’ will always fail if the node is reached again.
                    if self.priority_key == 'g':
                        h_score = 0
                    else:
                        h_score = neighbor['h']  #round(problem.heuristic(neighbor_state), 3) 
                        if not at_goal:  # by default min_edge_cost = 0 but can try this with min_edge_cost > 0 for "eps enhanced A*"
                            h_score = max(h_score, self.min_edge_cost)  # Don't use eps > 0 for h fns that always return in [0,1] or this will push all h to 1.. if eg degradation pushes h to < eps or h naturually < eps then make h eps
                        if self.bpmx1:
                            h_score = max(h_score, round(best_h - neighbor['cost'], 3))   # if parent h - cost > child h then increase child h
                    neighbor_node = Node(tentative_g_score, h_score, current_state)
                    if self.priority_key == 'f' and h_consistent: # Check whether current heuristic is consistent: if h(n) > cost(n, n') + h(n')
                        if current_h > neighbor['cost'] + h_score + 1e-6:
                            status += f" Inconsistent heuristic detected. parent h {current_h} > edgecost {neighbor['cost']} + child h {h_score} + 1e-6."
                            h_consistent = False
                    nodes_fwd[neighbor_state] = neighbor_node  # Add/Update the node in the Nodes dict
                    frontier.push(neighbor_state, 
                                    frontier.calc_priority(g=tentative_g_score, h=h_score), 
                                    frontier.calc_tiebreak1(g=tentative_g_score, h=h_score),
                                    prior_g=prior_g  ) # Push with priority and tiebreaker1 calculated priority

            if found_path:
                if self.priority_key == 'h':
                    break  # If BFS, break after first path found


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
            nodes_expanded_below_cstar_auto = sum(c_count_dict[p] for p in c_count_dict if p < U)
        
        print(status)

        if found_path:
            if str(problem).startswith('GRID-'):
                convert_func = util.decode_numbers
            else:
                convert_func = tuple
            path = reconstruct_path(nodes_fwd, start_node, problem.goal_state(), convert_func=convert_func)
            if not path:
                status += " Path too long to reconstruct."
            if self.visualise and hasattr(problem, 'visualise'):
                image_file = problem.visualise(path=path, path_type=self._str_repr, 
                                               visited_fwd=set(nodes_fwd.keys()), visualise_dirname=self.visualise_dirname)
                if not image_file: 
                    image_file = 'no file'

            return {"path": path, "cost": U, "nodes_expanded": nodes_expanded, 
                    "nodes_expanded_below_cstar": nodes_expanded_below_cstar, "nodes_expanded_below_cstar_auto": nodes_expanded_below_cstar_auto,
                    "time": end_time - start_time, "optimal": optimality_guaranteed, "visual": image_file, 
                    "max_heap_len": frontier.max_heap_size, 
                    "g_score_len": len(nodes_fwd),
                    "max_ram_taken": max_ram,
                    "status": status,
                    "prob_str": problem.prob_str, "heur": problem.h_str, "degr": problem.degradation, "admiss": problem.admissible, "costtype": problem.cost_type, "CS_pre": problem.cstar,
                    "nodes_sec": nodes_expanded / (end_time - start_time) if end_time > start_time else 0,}

        status += " No path found."
        return {"path": None, "cost": -1, "nodes_expanded": nodes_expanded, 
                "nodes_expanded_below_cstar": nodes_expanded_below_cstar,  "nodes_expanded_below_cstar_auto": nodes_expanded_below_cstar_auto,
                "time": end_time - start_time, "optimal": optimality_guaranteed, "visual": image_file, 
                "max_heap_len": frontier.max_heap_size, 
                "g_score_len": len(nodes_fwd),
                "max_ram_taken": max_ram,
                "status": status,
                "prob_str": problem.prob_str, "heur": problem.h_str, "degr": problem.degradation, "admiss": problem.admissible, "costtype": problem.cost_type, "CS_pre": problem.cstar,
                "nodes_sec": nodes_expanded / (end_time - start_time) if end_time > start_time else 0,}



    def __str__(self): # enable str(object) to return algo name
        return self._str_repr


def reconstruct_path(nodes_fwd, start_state, goal_state, convert_func=tuple):
    """Reconstructs the path from start to goal. Path is list of states"""
    path = []
    current = goal_state
    start_node = start_state 
    if current == start_node: 
        return [convert_func(start_node)]
    limit = 10000000 # Generic large limit
    count = 0
    while current != start_node:
        path.append(convert_func(current))
        parent = nodes_fwd[current].parent
        current = parent
        count += 1
        if count > limit: 
            print(f"Error: Path reconstruction exceeded limit ({limit}).")
            return None
            
    path.append(convert_func(start_node))
    return path[::-1] 


