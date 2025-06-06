"""
Code for Monte Carlo Tree Search (MCTS)

Can utilise heuristic in selection or rollout


"""
import math
import random
import time
import collections

import util



class HeuristicMCTSNode:  
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


class heuristic_mcts_search:
    """
    Heuristic MCTS search function.
    This function performs a Monte Carlo Tree Search (MCTS) with optional heuristic guidance.
    It can use heuristics in the selection phase and/or in the simulation (rollout) phase.
    """
    def __init__(self, iterations=100000, max_depth=150, exploration_weight=1.41, 
                 heuristic_weight=0.0, heuristic_rollout=False, epsilon=1e-6):
        self.iterations = iterations
        self.max_depth = max_depth
        self.exploration_weight = exploration_weight
        self.heuristic_weight = heuristic_weight
        self.heuristic_rollout = heuristic_rollout
        self.epsilon = epsilon
        self._str_repr = f"MCTS-hw{heuristic_weight}-hr{heuristic_rollout}-i{iterations}-md{max_depth}-ew{exploration_weight}"


    def search(self, problem):

        if hasattr(problem, 'load_or_create_pdbs') and problem.h_str.startswith('pdb'):  # create pdb here so it doesn't get pre-loaded for every problem instance in advance
            problem.load_or_create_pdbs()

        start_time = time.time()
        start_node = problem.initial_state()
        # Use the HeuristicMCTSNode
        root = HeuristicMCTSNode(state=start_node, problem=problem) 

        if root.is_terminal(): 
            return {"path": [root.state], "cost": 0, "nodes_expanded": 1, "time": time.time()-start_time, "optimal": False, "visual": "no file",
                    "iterations": 0, "h_weight": self.heuristic_weight, "h_rollout": self.heuristic_rollout,
                    "status": "Already at goal."}

        for i in range(self.iterations):
            node = root
            path_to_leaf = [node]
            
            # 1. Selection (using potentially heuristic-guided best_child)
            while not node.is_terminal() and node.is_fully_expanded():
                selected_child = node.best_child(exploration_weight=self.exploration_weight, 
                                                heuristic_weight=self.heuristic_weight, 
                                                epsilon=self.epsilon) 
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

            while not problem.is_goal(current_state) and sim_depth < self.max_depth:
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

                if self.heuristic_rollout and valid_neighbors_info:
                    # Heuristic-biased rollout: Choose neighbor probabilistically based on h value
                    neighbor_states = [ni[0] for ni in valid_neighbors_info]
                    heuristics = [problem.heuristic(ns) for ns in neighbor_states]
                    
                    # Calculate weights (lower heuristic -> higher weight)
                    weights = [1.0 / (self.epsilon + h) for h in heuristics]
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
                cost_step = problem.get_cost(current_node.state, child.state) #NOTE: not using move_info
                new_cost = current_cost + cost_step
                
                if new_cost < visited_in_tree.get(child.state, float('inf')):
                    # Check cost before adding to prevent cycles/redundancy? Or trust closed set?
                    # Let's add if cheaper or not visited in this BFS path extraction phase
                    visited_in_tree[child.state] = new_cost
                    new_path_list = current_path_list + [child.state]
                    if new_cost < min_cost_in_tree: # Only explore if potentially better
                        queue.append((child, new_path_list, new_cost))

        # Prepare results dictionary
        #algo_name = f"Heuristic MCTS (SelW={heuristic_weight:.2f}, Rollout={heuristic_rollout})"
        if best_path_found_list:
            return {"path": best_path_found_list, "cost": min_cost_in_tree, "nodes_expanded": nodes_explored_in_tree, 
                    "time": end_time - start_time, "optimal": False, "visual": "no file", "iterations": self.iterations, 
                    "tree_root_visits": root.visits, "h_weight": self.heuristic_weight, "h_rollout": self.heuristic_rollout,
                     "status": "Path found." }
        else:
            # If goal not found, return best guess for next move from root (greedy exploitation)
            best_first_move_node = root.best_child(exploration_weight=0, heuristic_weight=0) # Pure exploitation
            return {"path": None, "cost": -1, "nodes_expanded": nodes_explored_in_tree, 
                    "time": end_time - start_time, "optimal": False, "visual": "no file", "iterations": self.iterations, 
                    "best_next_state_estimate": util.make_prob_str(initial_state=best_first_move_node.state) if best_first_move_node else None, 
                    "tree_root_visits": root.visits, "h_weight": self.heuristic_weight, "h_rollout": self.heuristic_rollout,
                    "status": "Path not found."}

    def __str__(self): # enable str(object) to return algo name
        return self._str_repr




