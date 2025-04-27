"""
Unidirectional Searches

Dijkstra / Uniform Cost Search  (g only)
Greedy Best First Search        (h only)
A* Search                    (f = g + h)    

"""
import time
import util


# --- Generic Unidirectional Search Function ---
def generic_search(problem, priority_key='f', visualise=True):
    """
    Performs a generic best-first search using a closed set.
    Priority can be based on 'g', 'h', or 'f' = g+h. Handles variable costs.
    if visualise is True and problem supports it, will output visualisation to a subdir off the problem input dir.
    """
    if priority_key not in ['g', 'h', 'f']: raise ValueError("priority_key must be 'g', 'h', or 'f'")
    algo_name_map = {'g': "Uniform Cost", 'h': "Greedy Best-First", 'f': "Astar"}
    algorithm_name = algo_name_map[priority_key] #+ " (Generic)"
    optimality_guaranteed = (priority_key == 'g') or (priority_key=='f' and problem.optimality_guaranteed)

    image_file = "no file"

    start_time = time.time() 
    start_node = problem.initial_state()
    h_initial = problem.heuristic(start_node) if priority_key in ['h', 'f'] else 0
    initial_g = 0
    if priority_key == 'g': initial_priority = initial_g
    elif priority_key == 'h': initial_priority = h_initial
    else: initial_priority = initial_g + h_initial # 'f'

    frontier = util.PriorityQueue(tiebreaker2='FIFO') # Priority queue
    frontier.push(start_node, initial_priority, 0) # Push with priority and tiebreaker1
    came_from = {start_node: None}    # Dictionary of node:parent for path reconstruction
    g_score = {start_node: initial_g}
    closed_set = set()
    nodes_expanded = 0
    C = -1.0 # Current lowest cost on frontier
    U = float('inf') # Current lowest cost found
    found_path = None
    nodes_expanded_below_cstar = 0

    while not frontier.isEmpty():

        C = frontier.peek(priority_only=True) # Peek at the lowest priority element

        if C >= U: # If the estimated lowest cost path on frontier is greater cost than the best path found, stop
            print(f"Termination condition U ({U}) >= C ({C}) met.")
            # this check is for consistency with our BDHS algorithms - won't be triggered since breaking below when find goal 
            break

        current_state = frontier.pop(item_only=True) # Pop the state with the lowest priority
        
        # Optimization: If current_state's g_score is worse than recorded, skip
        # This can allegedly happen with duplicate states in the queue with different priorities
        if current_state in g_score and (g_score[current_state] + 1e-6) < (C - (problem.heuristic(current_state) if priority_key != 'g' else 0)):
            continue 

        if current_state in closed_set: continue
        nodes_expanded += 1
        closed_set.add(current_state) # Add after popping and checking

        if problem.is_goal(current_state):
            end_time = time.time()
            U = g_score.get(current_state)   
            found_path = current_state
            break 

        current_g_score = g_score.get(current_state)
        if current_g_score is None:
            print(f"Error: g_score for current_state {current_state} not found in g_score map. Continuing...") 
            continue # Should have g_score if reached here

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
                frontier.push(neighbor_state, priority, -tentative_g_score) # Push with priority and tiebreaker1 = -g ie higher g popped first
                

    end_time = time.time()

    if found_path:
        path = reconstruct_path(came_from, start_node, found_path)
        if visualise and hasattr(problem, 'visualise'):
            image_file = problem.visualise(path=path, path_type=algorithm_name, visited_fwd=closed_set)
            if not image_file: image_file = 'no file'
        return {"path": path, "cost": U, "nodes_expanded": nodes_expanded, 
                "time": end_time - start_time, "optimal": optimality_guaranteed, "visual": image_file, 
                "max_heap_size": frontier.max_heap_size}

    return {"path": None, "cost": -1, "nodes_expanded": nodes_expanded, "time": end_time - start_time, 
            "optimal": optimality_guaranteed, "visual": image_file, "max_heap_size": frontier.max_heap_size }


def reconstruct_path(came_from, start_state, goal_state):
    """Reconstructs the path from start to goal. Path is list of states"""
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


