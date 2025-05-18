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

        frontier = data_structures.PriorityQueue(priority_key=self.priority_key, 
                                                 tiebreaker1=self.tiebreaker1, tiebreaker2=self.tiebreaker2) # Priority queue
        frontier.push(start_node, 
                      frontier.calc_priority(g=g_initial, h=h_initial), 0) # Push with priority and tiebreaker1
        #state_info = data_structures.StateInfo()
        came_from = {start_node: None}    # Dictionary of node:parent for path reconstruction
        g_score = {start_node: g_initial}
        #state_info.add(start_node, parent=None, g=g_initial)
        closed_set = set()
        nodes_expanded = 0
        C = -1.0         # Current lowest cost on frontier
        U = float('inf') # Current lowest cost found to goal
        found_path = None
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
        h_admissable = True  # optionally check the admissability of the heuristic if running A* (not exhaustive)
        priority_diminished = 0

        while not frontier.isEmpty():
            if (time.time()-start_time)/60.0 > self.timeout:
                status += f"Timeout after {(time.time()-start_time)/60:.4f} mins."
                break
            if i % checkmem == 0 and util.get_available_ram() < self.min_ram:
                status += f"Out of RAM ({util.get_available_ram():.4f}GB remaining)."
                break
            i += 1

            current_priority = frontier.peek(priority_only=True) # Peek at the lowest priority element. 

            C = max(C, current_priority)  
            if current_priority + 1e-6 < C:  # This can happen with inconsistent heuristic which causes a state to be re-visited with a lower priority
                #print(f" Current priority {current_priority} is less than C {C}.")
                priority_diminished += 1

            if C >= U: # If the estimated lowest cost path on frontier is greater cost than the best path found, stop
                found_path = True
                status += f"Completed. Termination condition C ({C}) >= U ({U}) met."
                break


            current_state = frontier.pop(item_only=True) # Pop the state with the lowest priority
            current_g_score = g_score[current_state]
            if self.priority_key == 'g': 
                current_h = 0
            else: 
                current_h = problem.heuristic(current_state)
                if self.priority_key == 'f' and h_admissable:
                    if cstar and current_g_score + current_h > cstar + 1e-6:
                        status += f" Inadmissable heuristic detected."
                        h_admissable = False
            # Check for stale entries (duplicates in the heap with higher priority (f/g)_score
            # that were added before a better path was found). If the extracted
            # node's priority (- h if any) is higher than its current best known g_score,
            # it means we found a better path already, so we discard this stale entry.
            # The alternative would have been to delete from the priority queue in expansion below which is problematic with a heap
            if current_g_score + 1e-6 < current_priority - current_h:
                stale_count += 1
                continue  # skip stale dup state - rather than deleting from heapq before push in expansion

            #if current_state in closed_set: continue
            #closed_set.add(current_state) 

            if problem.is_goal(current_state):  # Update "lowest known soln cost" U when hit the goal
                found_goal_count += 1
                if current_g_score < U:
                    U = current_g_score
                    found_path = True
                    U_update_count += 1
                    if self.priority_key == 'h':  # BFS is not optimal so may as well end as soon as a path found
                        status += f"Terminating BFS as path found. U:{U}."
                        break
                    #continue
                    #break # if don't break here but continue then C >= U condition will fire above.


            nodes_expanded += 1
            if cstar and current_g_score < cstar:
                nodes_expanded_below_cstar += 1
            if self.priority_key != 'h':
                if c_count_dict.get(current_g_score) is None:
                    c_count_dict[current_g_score] = 0
                c_count_dict[current_g_score] +=1


            for neighbor_info in problem.get_neighbors(current_state):
                # Handle cases where get_neighbors might return just state or (state, move_info)
                if isinstance(neighbor_info, tuple) and len(neighbor_info) >= 1:
                    neighbor_state = neighbor_info[0]
                    move_info = neighbor_info[1] if len(neighbor_info) > 1 else None
                else:
                    neighbor_state = neighbor_info
                    move_info = None

                #if neighbor_state in closed_set: continue

                cost = problem.get_cost(current_state, neighbor_state, move_info)
                tentative_g_score = current_g_score + cost

                # Check whether current heuristic is consistent: if h(n) > cost(n, n') + h(n')
                if self.priority_key == 'f' and h_consistent:
                    h_score = problem.heuristic(neighbor_state)
                    if current_h > cost + h_score + 1e-6:
                        status += f" Inconsistent heuristic detected."
                        h_consistent = False


                #neighbor_g_score = state_info.get_g(neighbor_state)
                if tentative_g_score < g_score.get(neighbor_state, float('inf')):  #Per Wikipedia citing Russell&Norvig: if a node is reached by one path, removed from openSet, and subsequently reached by a cheaper path, it will be added to openSet again. This is essential to guarantee that the path returned is optimal if the heuristic function is admissible but not consistent. If the heuristic is consistent, when a node is removed from openSet the path to it is guaranteed to be optimal so the test ‘tentative_gScore < gScore[neighbor]’ will always fail if the node is reached again.
                    #state_info.add(neighbor_state, parent=current_state, g=tentative_g_score)
                    came_from[neighbor_state] = current_state 
                    g_score[neighbor_state] = tentative_g_score
                    h_score = problem.heuristic(neighbor_state) # for flexibility in calculations; redundant for eg uniform cost...
                    frontier.push(neighbor_state, 
                                  frontier.calc_priority(g=tentative_g_score, h=h_score), 
                                  frontier.calc_tiebreak1(g=tentative_g_score, h=h_score) ) # Push with priority and tiebreaker1 calculated priority

        end_time = time.time()
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
        nodes_expanded_below_cstar_auto = 0
        if len(c_count_dict) > 0:
            #status += f" c_count_dict len:{len(c_count_dict)}"
            nodes_expanded_below_cstar_auto = sum(c_count_dict[g] for g in c_count_dict if g < U)
        


        print(status)
        if found_path:
            #print("#### C count Dict ####")
            #if len(c_count_dict) < 100:
            #    print(c_count_dict)
            #path = reconstruct_path(state_info, start_node, found_path)
            path = reconstruct_path(came_from, start_node, problem.goal_state())
            if not path:
                status += " Path too long to reconstruct."
            if self.visualise and hasattr(problem, 'visualise'):
                image_file = problem.visualise(path=path, path_type=self._str_repr, 
                                               visited_fwd=closed_set, visualise_dirname=self.visualise_dirname)
                if not image_file: 
                    image_file = 'no file'

            return {"path": path, "cost": U, "nodes_expanded": nodes_expanded, 
                    "nodes_expanded_below_cstar": nodes_expanded_below_cstar, "nodes_expanded_below_cstar_auto": nodes_expanded_below_cstar_auto,
                    "time": end_time - start_time, "optimal": optimality_guaranteed, "visual": image_file, 
                    "max_heap_len": frontier.max_heap_size, 
                    "closed_set_len": len(closed_set),
#                    "state_dict_len": len(state_info.state_dict),
                    "g_score_len": len(g_score),
                    "came_from_len": len(came_from),
                    "status": status}

        status += " No path found."

        return {"path": None, "cost": -1, "nodes_expanded": nodes_expanded, 
                "nodes_expanded_below_cstar": nodes_expanded_below_cstar,  "nodes_expanded_below_cstar_auto": nodes_expanded_below_cstar_auto,
                "time": end_time - start_time, "optimal": optimality_guaranteed, "visual": image_file, 
                "max_heap_len": frontier.max_heap_size, 
                "closed_set_len": len(closed_set), 
#                "state_dict_len": len(state_info.state_dict),
                "g_score_len": len(g_score),
                "came_from_len": len(came_from),
                "status": status}


    def __str__(self): # enable str(object) to return algo name
        return self._str_repr


#def reconstruct_path(state_info, start_state, goal_state):
def reconstruct_path(came_from, start_state, goal_state):
    """Reconstructs the path from start to goal. Path is list of states"""
    path = []
    current = goal_state
    start_node = start_state 
    if current == start_node: return [tuple(start_node)]
    
    limit = 100000 # Generic large limit

    count = 0
    while current != start_node:
        path.append(tuple(current))
#        parent = state_info.get_parent(current)
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


