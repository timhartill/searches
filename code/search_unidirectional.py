"""
Unidirectional Searches

Dijkstra / Uniform Cost Search  (g only)
Greedy Best First Search        (h only)
A* Search                    (f = g + h)    

"""
import time
import heapq


# --- Generic Unidirectional Search Function ---
def generic_search(problem, priority_key='f', cstar=None, visualise=True):
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
    nodes_expanded_below_cstar = 0

    while frontier:
        current_priority, current_state = heapq.heappop(frontier)
        
        # Optimization: If current_state's g_score is worse than recorded, skip
        # This can happen with duplicate states in the queue with different priorities
        #if current_state in g_score and g_score[current_state] < current_priority - (problem.heuristic(current_state) if priority_key != 'g' else 0):
             # Check g_score derived from priority vs stored g_score if applicable
             # Let's rely on the closed set check primarily
             #pass 

        if current_state in closed_set: continue
        nodes_expanded += 1
        closed_set.add(current_state) # Add after popping and checking

        if problem.is_goal(current_state):
            end_time = time.time()
            path = reconstruct_path(came_from, start_node, current_state)
            final_g_score = g_score.get(current_state)
            if visualise and hasattr(problem, 'visualise'):
                image_file = problem.visualise(path=path, path_type=algorithm_name, visited_fwd=closed_set)
                if not image_file: image_file = 'no file'
            return {"path": path, "cost": final_g_score, "nodes_expanded": nodes_expanded, "time": end_time - start_time, "algorithm": algorithm_name, "optimal": optimality_guaranteed, "visual": image_file}

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
    return {"path": None, "cost": -1, "nodes_expanded": nodes_expanded, "time": end_time - start_time, "algorithm": algorithm_name, "optimal": False, "visual": image_file }


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


