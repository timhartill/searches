"""
Code for puzzle type problems

Sliding Tile
Pancake
Towers of Hanoi

Supports
- Variable or unit edge costs
- Admissible or inadmissible heuristics
- Degradation of heuristic (ignore first k elements)



"""
import math
import random

import util



# --- SlidingTileProblem (Corrected Formatting & use_variable_costs flag) ---
class SlidingTileProblem:
    """
    Implements the Sliding Tile puzzle problem interface.
    Can use uniform cost (1) or variable cost (value of tile moved).
    """
    def __init__(self, initial_state, goal_state=None, 
                 use_variable_costs=False, make_heuristic_inadmissable=False,
                 degradation=0, heuristic="manhattan", cstar=None):
        self.initial_state_tuple = tuple(initial_state)
        self.n = int(math.sqrt(len(initial_state)))
        if self.n * self.n != len(initial_state):
            self.max_cols = self.n
            # Check if the state is in the form of n x n or n+1 x n
            max_rows, col_check = divmod(len(initial_state), self.max_cols)
            if col_check != 0:
                raise ValueError("Invalid state length for a sliding tile puzzle. Must be n x n or n+1 x n.")
            self.max_rows = max_rows
        else: # square puzzle
            self.max_rows = self.n
            self.max_cols = self.n
            
        if goal_state:
            if len(goal_state) != len(initial_state):
                raise ValueError("Goal state must be the same length as initial state.")
            self.goal_state_tuple = tuple(goal_state)
        else:
            sorted_list = list(range(1, self.max_rows * self.max_cols)) + [0]
            self.goal_state_tuple = tuple(sorted_list)
            
        self._goal_positions = {tile: i for i, tile in enumerate(self.goal_state_tuple)}
        self._start_positions = {tile: i for i, tile in enumerate(self.initial_state_tuple)}  # for bdhs
        self.use_variable_costs = use_variable_costs
        self.optimality_guaranteed = True
        self.make_heuristic_inadmissable = make_heuristic_inadmissable
        if make_heuristic_inadmissable:
            self.h_multiplier = len(initial_state) * (degradation+10)
            self.optimality_guaranteed = False
        else:
            self.h_multiplier = 1    
        self.degradation = degradation    
        self.cstar = cstar
        cost_type = "VarCost" if use_variable_costs else "UnitCost"
        self.h_str = heuristic  #"Manhattan" # Manhattan distance heuristic is the only one implemented
        self._str_repr = f"SlidingTile-{self.max_rows}x{self.max_cols}-{util.make_prob_str(initial_state=self.initial_state_tuple, goal_state=self.goal_state_tuple)}-{cost_type}-h{self.h_str}-d{degradation}-a{not make_heuristic_inadmissable}-cs{cstar}"

    def initial_state(self): 
        return self.initial_state_tuple
        
    def goal_state(self): 
        return self.goal_state_tuple
        
    def is_goal(self, state, backward=False): 
        if backward:
            return state == self.initial_state_tuple
        return state == self.goal_state_tuple

    def get_neighbors(self, state):
        """Returns list of tuples: (neighbor_state, moved_tile_value) from state
        where moved_tile_value is the value (label) of the tile that was moved to the blank space.
        """
        neighbors = []
        blank_index = state.index(0)    # index of blank in state list 
        row, col = divmod(blank_index, self.max_cols)
        moves = {'up': (row - 1, col), 'down': (row + 1, col), 'left': (row, col - 1), 'right': (row, col + 1)}
        for move_dir, (new_row, new_col) in moves.items():
            if 0 <= new_row < self.max_rows and 0 <= new_col < self.max_cols:  # if valid move
                new_blank_index = new_row * self.max_cols + new_col  # index of blank if move this direction
                new_state_list = list(state)
                moved_tile_value = new_state_list[new_blank_index] 
                # swap blank with the tile in the new position:
                new_state_list[blank_index], new_state_list[new_blank_index] = new_state_list[new_blank_index], new_state_list[blank_index]
                neighbors.append( (tuple(new_state_list), moved_tile_value) ) 
        return neighbors 

    def get_cost(self, state1, state2, move_info=None):
        """
        Returns cost of move. If use_variable_costs is True, cost is the
        value of the tile moved (min 1). Otherwise, cost is 1.
        """
        if not self.use_variable_costs:
            return 1 
            
        if move_info is not None:
            moved_tile_value = move_info
            return max(1, moved_tile_value) 
        else: # Used when reconstructing path cost post-search without move_info
            blank1_idx, blank2_idx, moved_tile = -1, -1, -1
            # Ensure state1 and state2 are tuples/sequences before len()
            if not hasattr(state1, '__len__') or not hasattr(state2, '__len__') or len(state1) != len(state2):
                print(f"Warning: Invalid states for cost calculation fallback: {state1}, {state2}")
                return 1 # Fallback cost
            for i in range(len(state1)):
                if state1[i] == 0: blank1_idx = i
                if state2[i] == 0: blank2_idx = i
            if blank1_idx != -1 and blank2_idx != -1 :
                 moved_tile = state2[blank1_idx] 
                 return max(1, moved_tile)
            else:
                 print(f"Warning: Could not determine moved tile between {state1} and {state2}")
                 return 1 

    def heuristic(self, state, backward=False):
        """
        Calculates the Manhattan distance heuristic (number of steps).
        NOTE: This heuristic counts steps (cost=1). If variable (positive) costs are used,
        its effectiveness will decrease but still admissable since var costs >= unit costs.
        """
        if backward: # For bidirectional search
            target_positions = self._start_positions
        else:
            target_positions = self._goal_positions
        distance = 0
        multiplier = 1
        ignored_tiles = set(range(self.degradation + 1))
        for i, tile in enumerate(state):
            if tile not in ignored_tiles:
                current_pos = divmod(i, self.max_cols)
                goal_idx = target_positions.get(tile)
                if goal_idx is None: continue 
                goal_pos = divmod(goal_idx, self.max_cols)
                if self.make_heuristic_inadmissable:
                    multiplier = i
                distance += multiplier * (abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1]))
        return distance * self.h_multiplier

    def __str__(self): 
        return self._str_repr

# --- PancakeProblem (Corrected Formatting & use_variable_costs flag) ---
class PancakeProblem:
    """
    Implements the Pancake Sorting problem interface.
    Can use uniform cost (1) or variable cost (k = pancakes flipped).
    """
    def __init__(self, initial_state, goal_state=None, 
                 use_variable_costs=False, make_heuristic_inadmissable=False,
                 degradation=0, heuristic = "symgap", cstar=None):
        self.initial_state_tuple=tuple(initial_state) 
        self.n=len(initial_state)
        if goal_state: self.goal_state_tuple=tuple(goal_state) 
        else: self.goal_state_tuple=tuple(sorted(initial_state))
        self.use_variable_costs = use_variable_costs
        self.optimality_guaranteed = (not use_variable_costs) and (not make_heuristic_inadmissable)
        self.make_heuristic_inadmissable = make_heuristic_inadmissable
        if make_heuristic_inadmissable:
            self.h_multiplier = len(initial_state) * (degradation+10)
            self.optimality_guaranteed = False
        else:
            self.h_multiplier = 1    
        self.degradation = degradation
        self.cstar = cstar
        cost_type = "VarCost" if use_variable_costs else "UnitCost"
        self.h_str = heuristic   #"SymGap" # Symmetric Gap heuristic is the only one implemented
        self._str_repr = f"Pancake-{self.n}-{util.make_prob_str(initial_state=self.initial_state_tuple, goal_state=self.goal_state_tuple)}-{cost_type}-h{self.h_str}-d{degradation}-a{not make_heuristic_inadmissable}-cs{cstar}"
        
    def initial_state(self): 
        return self.initial_state_tuple
        
    def goal_state(self): 
        return self.goal_state_tuple
        
    def is_goal(self, state, backward=False): 
        if backward:
            return state == self.initial_state_tuple
        return state == self.goal_state_tuple

    def get_neighbors(self, state):
        """Returns list of tuples: (neighbor_state, k_flipped)"""
        neighbors = []
        for k in range(2, self.n + 1):
            if k > 1: 
                flipped_part = state[:k][::-1]
                rest_part = state[k:]
                neighbors.append((flipped_part + rest_part, k))
        return neighbors 

    def get_cost(self, state1, state2, move_info=None):
        """
        Returns cost of move. If use_variable_costs is True, cost is k 
        (number of pancakes flipped). Otherwise, cost is 1.
        """
        if not self.use_variable_costs:
            return 1
            
        if move_info is not None:
            k_flipped = move_info
            return k_flipped 
        else: 
            if not hasattr(state1, '__len__') or not hasattr(state2, '__len__') or len(state1) != len(state2): return 1 
            n=len(state1)
            if state1 == state2: return 0 
            
            diff_k = n 
            for k in range(n):
                 if state1[k] != state2[k]:
                      diff_k = k + 1 
                      break

            found_k = 1 
            for k_try in range(max(2, diff_k), n + 1):
                 if state1[:k_try][::-1] == state2[:k_try] and state1[k_try:] == state2[k_try:]:
                      found_k = k_try; break
                 elif k_try == n and state1[:k_try][::-1] == state2[:k_try]: 
                      found_k = k_try; break

            # Verify diff_k if loop didn't find larger k
            if found_k == 1 and diff_k >= 2:
                if state1[:diff_k][::-1] == state2[:diff_k] and state1[diff_k:] == state2[diff_k:]:
                    found_k = diff_k
                else:
                    print(f"Warning: Could not determine k flipped between {state1} and {state2}")
            elif found_k == 1 and diff_k < 2: # Should only happen if states are same, handled above
                 print(f"Warning: Unexpected state comparison in pancake get_cost fallback.")

            return found_k

    def gap_heuristic(self, state_1, state_2):
        """
        Calculates a heuristic value based on the order of elements in two states,
        ignoring the first 'degradation' elements.

        Args:
            state_1: A list of integers representing the first state.
            state_2: A list of integers representing the second state (the goal state).
            degradation: An integer representing the number of initial elements to ignore.

        Returns:
            An integer representing the heuristic value.

        Raises:
            ValueError: If the input states have different lengths.
        """
        if len(state_1) != len(state_2):
            raise ValueError("Input states must have the same length.")

        heuristic_value = 0
        ignored_pancakes = set(range(1, self.degradation + 1))
        multiplier = 1

        for i in range(len(state_1) - 1):
            pancake_i = state_1[i]
            pancake_j = state_1[i + 1]

            if pancake_i in ignored_pancakes or pancake_j in ignored_pancakes:
                continue

            try:
                goal_position_i = state_2.index(pancake_i)  # find index
            except ValueError:
                continue  # Handle the case where pancake_i is not in state_2

            if (goal_position_i != 0 and state_2[goal_position_i - 1] == pancake_j) or \
                (goal_position_i != len(state_1) - 1 and state_2[goal_position_i + 1] == pancake_j):
                heuristic_value += 0
            else:
                if self.make_heuristic_inadmissable:
                    multiplier = i
                heuristic_value += multiplier

        return heuristic_value

    def heuristic(self, state, backward=False):
        """
        Calculates the Symetric Gap Heuristic (number of adjacent non-consecutive pairs both ways).
        NOTE: This counts number of "breaks". If variable costs (cost=k) are used,
        this heuristic likely becomes non-admissible as one flip (cost k) can fix
        at most 2 gaps.
        """
        #return sum(1 for i in range(self.n-1) if abs(state[i]-state[i+1]) > 1) * self.h_multiplier
        if backward: target_tuple = self.initial_state_tuple
        else: target_tuple = self.goal_state_tuple
        return max(self.gap_heuristic(state, target_tuple), 
                   self.gap_heuristic(target_tuple, state)) * self.h_multiplier

    def __str__(self): 
        return self._str_repr


# --- TowersOfHanoiProblem (Corrected Formatting) ---
class TowersOfHanoiProblem:
    """Implements the Towers of Hanoi problem interface. Cost is always 1.
    Heuristic is either standard 3 peg or "Infinite Peg Relaxation" heuristic.
    Infinite Peg Relaxation is admissible but weak, std 3 peg only admissable for 3 peg problems.
    Allows for degrading heuristic by ignoring disks.
    Allows for inadmissible heuristic. 
    State is a Tuple of current peg for each disk eg ('A', 'A', 'B', 'C', 'A', 'A', 'A') for 7 disks with idx 0 = smallest disk.
    For simplicity, initial peg is always 'A', goal state is list(target peg*num_disks) and number of pegs is ord(target peg)-ord(initial peg).
    """
    def __init__(self, initial_state= ['A','A','A','A','A','A','A','A','A','A','A','A'],
                 goal_state = ['D','D','D','D','D','D','D','D','D','D','D','D'], 
                 make_heuristic_inadmissable=False, degradation=0, 
                 heuristic="infinitepegrelaxation", cstar=None): 
        if len(goal_state) == 1:  # allow shorthand for goal state eg ['D']
            goal_state = [goal_state[0]] * len(initial_state)
        if len(initial_state) != len(goal_state):
            raise ValueError(f"Initial {initial_state} and goal {goal_state} states must have the same length.")
        self.num_disks = len(initial_state)
        if self.num_disks < 1: 
            raise ValueError("Number of disks must be at least 1.")
        self.initial_peg = 'A'  # for simplicity, always start from A
        self.target_peg = goal_state[0]  # target peg is the peg where all disks should end up
        initial_code_point = ord(self.initial_peg)
        terminal_code_point = ord(self.target_peg)
        if initial_code_point >= terminal_code_point:
            raise ValueError(f"Initial peg {self.initial_peg} and target pegs {self.target_peg} must be different with ord(target) > ord(initial).")
        pegs = []
        for code_point in range(initial_code_point, terminal_code_point + 1):
            pegs.append(chr(code_point))
        self.pegs=pegs
        if not (self.initial_peg in self.pegs and self.target_peg in self.pegs and self.initial_peg != self.target_peg):
            raise ValueError(f"Invalid initial or target peg. Must be one of {self.pegs} and different.")
        if any(peg not in self.pegs for peg in initial_state):
            raise ValueError(f"Invalid pegs in initial state. Must be one of {self.pegs}.")

        self.use_variable_costs = False # Cost is always 1
        self.optimality_guaranteed = (not self.use_variable_costs) and (not make_heuristic_inadmissable)
        self.make_heuristic_inadmissable = make_heuristic_inadmissable
        if make_heuristic_inadmissable:
            self.h_multiplier = self.num_disks * (degradation+10)**2
            self.optimality_guaranteed = False
        else:
            self.h_multiplier = 1
        heuristic = heuristic.lower()   
        if heuristic not in ["3pegstd", "infinitepegrelaxation"]:
            raise ValueError(f"Invalid heuristic: {heuristic}. Must be '3pegstd' or 'infinitepegrelaxation'.")
        if len(self.pegs) > 3 and heuristic == "3pegstd": # not optimal for bidirectional or A* for > 3 pegs
            self.optimality_guaranteed = False
        self.h_str = heuristic 
        self.degradation = degradation    
        self.initial_state_tuple = tuple(initial_state) #tuple([initial_peg]*num_disks)  # (A, A, A, ..., A)  Smallest disk is index 0
        self.goal_state_tuple=tuple(goal_state)      # (C, C, C, ..., C)
        self.cstar = cstar
        self._str_repr=f"TowersOfHanoi-{self.num_disks}-{util.make_prob_str(initial_state=self.initial_state_tuple, goal_state=self.goal_state_tuple)}-h{heuristic}-d{degradation}-a{self.optimality_guaranteed and not make_heuristic_inadmissable}-cs{cstar}"

    def initial_state(self): 
        return self.initial_state_tuple
        
    def goal_state(self): 
        return self.goal_state_tuple
        
    def is_goal(self, state, backward=False): 
        if backward:
            return state == self.initial_state_tuple
        return state == self.goal_state_tuple

    def _get_peg_tops(self, state):
        """Helper to find the smallest (topmost) disk index on each peg.
        eg _get_peg_tops(('A', 'A', 'B', 'C', 'A', 'A', 'A')) = {'A': 0, 'B': 2, 'C': 3}
        """
        peg_tops = {p: None for p in self.pegs}
        # Use infinity to correctly find the minimum disk index
        peg_top_disk_index = {p: float('inf') for p in self.pegs} 
        
        for disk_index, peg in enumerate(state):
             if disk_index < peg_top_disk_index[peg]:
                  peg_top_disk_index[peg] = disk_index
                  
        # Update peg_tops with the actual disk index found, or None
        for peg in self.pegs:
             if peg_top_disk_index[peg] != float('inf'):
                  peg_tops[peg] = peg_top_disk_index[peg] 
                  
        return peg_tops

    def get_neighbors(self, state): 
        """Returns list of tuples: (neighbor_state, cost=1)
        eg get_neighbours(('A', 'A', 'B', 'C', 'A', 'A', 'A')) =
                [(('B', 'A', 'B', 'C', 'A', 'A', 'A'), 1),
                 (('C', 'A', 'B', 'C', 'A', 'A', 'A'), 1),
                 (('A', 'A', 'C', 'C', 'A', 'A', 'A'), 1)]
        """
        nbs=[] 
        pts=self._get_peg_tops(state);
        pegs = self.pegs.copy() 
        for sp in pegs:
            dtm = pts[sp] # Disk To Move index (top disk on source)
            if dtm is not None: # If source peg is not empty
                for dp in pegs: # Destination Peg
                    if sp != dp: # Cannot move to same peg
                        tdod = pts[dp] # Top Disk On Destination index
                        # Check move validity: dest empty OR moving disk < disk on dest
                        if tdod is None or dtm < tdod:
                            nsl = list(state)
                            nsl[dtm] = dp # Move disk dtm to peg dp
                            nbs.append((tuple(nsl), 1)) # Append (new_state, cost)
        return nbs

    def heuristic(self, state, backward=False): 
        """Calculates the standard admissible heuristic for 3 peg Towers of Hanoi and "Infinite Peg Relaxation" heuristic
        which is admissable but relatively weak.
        Allows for degrading heuristic by ignoring disks
        Allows for inadmissable heuristic but A* still always finds optimal path.. 
        """
        h=0 
        if backward: ctp=self.initial_peg # bdhs current target peg for disk k
        else: ctp=self.target_peg 
        multiplier = 1
        ignored_disks = set(range(self.degradation))

        if self.h_str == "3pegstd":
            for k in range(self.num_disks-1,-1,-1): # Iterate largest disk (N-1) down to smallest (0)
                if k in ignored_disks: 
                    continue # Ignore disks in degradation
                if state[k] != ctp: 
                    # If disk k is not where it should be relative to disk k+1 (or final target)
                    # it and all smaller disks must move. Lower bound cost is 2^k.
                    if self.make_heuristic_inadmissable:
                        multiplier = random.choice(range(1,self.num_disks)) #TJH: It seems with anything based on k A* still finds optimal path 
                    h += 2**k * multiplier
                    # The new target for disk k-1 becomes the 'third' peg 
                    # (neither where k is, nor where k should have been)
                    ctp = next(p for p in self.pegs if p!=state[k] and p!=ctp)
                # else: disk k is on the correct peg (ctp), so target for k-1 remains ctp.
        elif self.h_str == "infinitepegrelaxation":
            # See Additive Pattern databases, Felner et al 2004
            # much weaker than pattern database heuristics but admissable for > 3 pegs and works in bidirectional
            for peg in self.pegs:
                if peg != ctp:
                    # 1 Sum for non-goal pegs = 2 * # disks on peg - 1
                    num_disks_on_peg = 0
                    for k in range(self.num_disks):
                        if k in ignored_disks: 
                            continue
                        if state[k] == peg:
                            if self.make_heuristic_inadmissable:
                                multiplier = random.choice(range(1,self.num_disks)) #k**k
                            num_disks_on_peg += multiplier
                    h += (2*num_disks_on_peg) - 1
                else:  # goal peg    
                    # 2 Sum for goal peg = 2 * each disk that must move to allow other disks to move onto goal. Count downward from largest until break
                    goal_disks = []
                    for k in range(self.num_disks-1,-1,-1):
                        if state[k] == peg:
                            goal_disks.append(k)
                    start_count = False
                    num_disks_on_peg = 0
                    for i,k in enumerate(goal_disks):
                        if i > 0 and goal_disks[i-1] - goal_disks[i] > 1:
                            start_count = True
                        if start_count:
                            if k in ignored_disks: 
                                continue
                            if self.make_heuristic_inadmissable:
                                multiplier = random.choice(range(1,self.num_disks)) #k**k
                            num_disks_on_peg += multiplier
                    h += 2*num_disks_on_peg

        return h * self.h_multiplier

    def get_cost(self, state1, state2, move_info=None): 
        """Cost is always 1 for Towers of Hanoi."""
        return 1
    
#    def visualise(self, cell_size: int = 10, path: list = None, meeting_node: tuple = None,
#                  path_type: str = '', output_file_ext: str = 'png',
#                  display: bool = False, return_image: bool = False):
#        """ Placeholder - see GridProblem for implemented example"""
#        return None

    def __str__(self): 
        return self._str_repr



