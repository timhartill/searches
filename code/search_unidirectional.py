"""
Unidirectional Searches

Dijkstra / Uniform Cost Search  (g only)
Greedy Best First Search        (h only)
A* Search                    (f = g + h)    
"""
import time
import util
import data_structures

algo_name_map = {'g': "UniformCost", 'h': "GreedyBestFirst", 'f': "Astar"}


# --- Generic Unidirectional Search Function ---
class generic_search:
    """
    Performs a generic best-first search using a closed set.
    Priority can be based on 'g', 'h', or 'f' = g+h. Handles variable costs.
    if visualise is True and problem supports it, will output visualisation to a subdir off the problem input dir.
    """
    def __init__(self, priority_key='f', tiebreaker1='-g', tiebreaker2 = 'NONE', 
                 visualise=True, visualise_dirname='', min_ram=2.0, timeout=30.0):
        """
        :param problem: The search problem to solve
        :param priority_key: 'g', 'h', or 'f' = g+h. Determines the priority of the nodes in the search.
        :param visualise: If True, will output a visualisation of the search to a subdir off the problem input dir.
        :param tiebreaker2: 2nd level Tiebreaker for the priority queue. Can be 'FIFO', 'LIFO', or 'NONE' for no 2nd level.
        """
        if priority_key not in algo_name_map: raise ValueError(f"priority_key must be in {algo_name_map}")
        self.timeout = timeout
        self.min_ram = min_ram
        self.priority_key = priority_key
        self.visualise = visualise
        self.visualise_dirname = visualise_dirname
        self.tiebreaker1 = tiebreaker1  # see calc_tiebreak_val for options
        self.tiebreaker2 = tiebreaker2
        self._str_repr = f"{algo_name_map[priority_key]}-p{priority_key}-tb1{tiebreaker1}-tb2{tiebreaker2}"


    def search(self, problem):
        optimality_guaranteed = (self.priority_key == 'g') or (self.priority_key=='f' and problem.optimality_guaranteed) or (self.priority_key=='FIFO' and problem.optimality_guaranteed and not problem.use_variable_costs)
        
        start_time = time.time() 
        start_node = problem.initial_state()
        h_initial = problem.heuristic(start_node) if self.priority_key in ['h', 'f'] else 0
        g_initial = 0
        #if self.priority_key == 'g': initial_priority = initial_g
        #elif self.priority_key == 'h': initial_priority = h_initial
        #else: initial_priority = initial_g + h_initial # 'f'

        frontier = data_structures.PriorityQueue(priority_key=self.priority_key, 
                                                 tiebreaker1=self.tiebreaker1, tiebreaker2=self.tiebreaker2) # Priority queue
        frontier.push(start_node, 
                      frontier.calc_priority(g=g_initial, h=h_initial), 0) # Push with priority and tiebreaker1
        came_from = {start_node: None}    # Dictionary of node:parent for path reconstruction
        g_score = {start_node: g_initial}
        closed_set = set()
        nodes_expanded = 0
        C = -1.0         # Current lowest cost on frontier
        U = float('inf') # Current lowest cost found
        found_path = None
        if hasattr(problem, "cstar"):
            cstar = problem.cstar
        else:
            cstar = None
        nodes_expanded_below_cstar = 0
        i = 0
        status = ""
        checkmem = 100000

        while not frontier.isEmpty():

            C = frontier.peek(priority_only=True) # Peek at the lowest priority element

            if C >= U: # If the estimated lowest cost path on frontier is greater cost than the best path found, stop
                status += f"Completed. Termination condition U ({U}) >= C ({C}) met."
                # this check is for consistency with our BDHS algorithms - won't be triggered since breaking below when find goal 
                break

            current_state = frontier.pop(item_only=True) # Pop the state with the lowest priority
            
            # Optimization: If current_state's g_score is worse than recorded, skip
            # This can allegedly happen with duplicate states in the queue with different priorities
            if current_state in g_score and (g_score[current_state] + 1e-6) < (C - (problem.heuristic(current_state) if self.priority_key != 'g' else 0)):
                continue 

            if current_state in closed_set: continue
            nodes_expanded += 1
            
            closed_set.add(current_state) # Add after popping and checking

            current_g_score = g_score.get(current_state)

            if cstar and current_g_score < cstar:
                nodes_expanded_below_cstar += 1

            if problem.is_goal(current_state):
                end_time = time.time()
                U = g_score.get(current_state)   
                found_path = current_state
                status += f"Completed. Goal found U: ({U})  C: ({C})."
                break 

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
                    h_score = problem.heuristic(neighbor_state) # for flexibility in calculations; redundant for eg uniform cost...
                    frontier.push(neighbor_state, 
                                  frontier.calc_priority(g=tentative_g_score, h=h_score), 
                                  frontier.calc_tiebreak1(g=tentative_g_score, h=h_score) ) # Push with priority and tiebreaker1 calculated priority
            if (time.time()-start_time)/60.0 > self.timeout:
                status += f"Timeout after {(time.time()-start_time)/60:.4f} mins."
                break
            if i % checkmem == 0 and util.get_available_ram() < self.min_ram:
                status += f"Out of RAM ({util.get_available_ram():.4f}GB remaining)."
                break
            i += 1

        end_time = time.time()
        image_file = 'no file'
        if not status:
            status = "Completed."
        print(status)
        if found_path:
            path = reconstruct_path(came_from, start_node, found_path)
            if not path:
                status += " Path too long to reconstruct."
            if self.visualise and hasattr(problem, 'visualise'):
                image_file = problem.visualise(path=path, path_type=self._str_repr, 
                                               visited_fwd=closed_set, visualise_dirname=self.visualise_dirname)
                if not image_file: 
                    image_file = 'no file'

            return {"path": path, "cost": U, "nodes_expanded": nodes_expanded, "nodes_expanded_below_cstar": nodes_expanded_below_cstar,
                    "time": end_time - start_time, "optimal": optimality_guaranteed, "visual": image_file, 
                    "max_heap_len": frontier.max_heap_size, 
                    "closed_set_len": len(closed_set), "closed_set_gb": util.get_size(closed_set),
                    "g_score_len": len(g_score), "g_score_gb": util.get_size(g_score),
                    "came_from_len": len(came_from), "came_from_gb": util.get_size(came_from),
                    "status": status}

        status += " No path found."

        return {"path": None, "cost": -1, "nodes_expanded": nodes_expanded, "nodes_expanded_below_cstar": nodes_expanded_below_cstar,
                "time": end_time - start_time, "optimal": optimality_guaranteed, "visual": image_file, 
                "max_heap_len": frontier.max_heap_size, 
                "closed_set_len": len(closed_set), "closed_set_gb": util.get_size(closed_set), 
                "g_score_len": len(g_score), "g_score_gb": util.get_size(g_score),
                "came_from_len": len(came_from), "came_from_gb": util.get_size(came_from),
                "status": status }


    def __str__(self): # enable str(object) to return algo name
        return self._str_repr


def reconstruct_path(came_from, start_state, goal_state):
    """Reconstructs the path from start to goal. Path is list of states"""
    path = []
    current = goal_state
    start_node = start_state 
    if current == start_node: return [start_node]
    
    limit = 100000 # Generic large limit

    count = 0
    while current != start_node:
        path.append(tuple(current))
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
            
    path.append(tuple(start_node))
    return path[::-1] 


