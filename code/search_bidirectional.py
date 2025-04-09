"""
BiDirectional Searches

Bidirectional A*

"""
import heapq
import time



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
            if current_g_fwd >= best_path_cost: continue          
            closed_fwd.add(current_state_fwd)
            nodes_expanded += 1
            if current_state_fwd in g_score_bwd: 
                current_path_cost = current_g_fwd + g_score_bwd[current_state_fwd]
                if current_path_cost < best_path_cost: 
                    best_path_cost = current_path_cost
                    meeting_node = current_state_fwd
            
            for neighbor_info in problem.get_neighbors(current_state_fwd):
                neighbor_state, move_info = (neighbor_info if isinstance(neighbor_info, tuple) and len(neighbor_info)==2 else (neighbor_info, None))
                if neighbor_state in closed_fwd: continue
                cost = problem.get_cost(current_state_fwd, neighbor_state, move_info) 
                tentative_g_score = current_g_fwd + cost
                if tentative_g_score < g_score_fwd.get(neighbor_state, float('inf')):
                    came_from_fwd[neighbor_state] = current_state_fwd 
                    g_score_fwd[neighbor_state] = tentative_g_score
                    h_score = problem.heuristic(neighbor_state) 
                    f_score = tentative_g_score + h_score
                    if f_score < best_path_cost: 
                        heapq.heappush(frontier_fwd, (f_score, neighbor_state))
        
        # --- Backward Step ---
        if frontier_bwd:
            _, current_state_bwd = heapq.heappop(frontier_bwd)
            if current_state_bwd in closed_bwd: continue
            current_g_bwd = g_score_bwd.get(current_state_bwd, float('inf'))
            if current_g_bwd + problem.heuristic(current_state_bwd) >= best_path_cost: continue 
            closed_bwd.add(current_state_bwd)
            nodes_expanded += 1
            if current_state_bwd in g_score_fwd: 
                current_path_cost = g_score_fwd[current_state_bwd] + current_g_bwd
                if current_path_cost < best_path_cost: 
                    best_path_cost = current_path_cost
                    meeting_node = current_state_bwd

            for neighbor_info in problem.get_neighbors(current_state_bwd):
                neighbor_state, move_info = (neighbor_info if isinstance(neighbor_info, tuple) and len(neighbor_info)==2 else (neighbor_info, None))
                if neighbor_state in closed_bwd: continue
                cost = problem.get_cost(current_state_bwd, neighbor_state, move_info) 
                tentative_g_score = current_g_bwd + cost 
                if tentative_g_score < g_score_bwd.get(neighbor_state, float('inf')):
                    came_from_bwd[neighbor_state] = current_state_bwd 
                    g_score_bwd[neighbor_state] = tentative_g_score
                    h_score = problem.heuristic(neighbor_state)
                    f_score = tentative_g_score + h_score
                    if f_score < best_path_cost: 
                        heapq.heappush(frontier_bwd, (f_score, neighbor_state))
        
    end_time = time.time()
    if meeting_node:
        path = reconstruct_bidirectional_path(came_from_fwd, came_from_bwd, start_node, goal_node, meeting_node)
        final_cost = -1
        recalculated_cost = 0
        cost_mismatch = False
        if path:
             try:
                 for i in range(len(path) - 1):
                     recalculated_cost += problem.get_cost(path[i], path[i+1]) # Use fallback cost
                 final_cost = recalculated_cost
                 if abs(final_cost - best_path_cost) > 1e-6: cost_mismatch = True
             except Exception as e:
                 print(f"Error recalculating bidirectional path cost: {e}")
                 final_cost = best_path_cost 
        if cost_mismatch: 
            print(f"Warning: Bidirectional cost mismatch! PathRecalc={final_cost}, SearchCost={best_path_cost}")
        final_reported_cost = best_path_cost # Report cost found by search
        return {"path": path, "cost": final_reported_cost if path else -1, "nodes_expanded": nodes_expanded, "time": end_time - start_time, "algorithm": "Bidirectional A*"}
    else:
        return {"path": None, "cost": -1, "nodes_expanded": nodes_expanded, "time": end_time - start_time, "algorithm": "Bidirectional A*"}


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


