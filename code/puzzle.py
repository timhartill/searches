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



# --- SlidingTileProblem (Corrected Formatting & use_variable_costs flag) ---
class SlidingTileProblem:
    """
    Implements the Sliding Tile puzzle problem interface.
    Can use uniform cost (1) or variable cost (value of tile moved).
    """
    def __init__(self, initial_state, goal_state=None, 
                 use_variable_costs=False, make_heuristic_inadmissable=False,
                 degradation=0):
        self.initial_state_tuple = tuple(initial_state)
        self.n = int(math.sqrt(len(initial_state)))
        if self.n * self.n != len(initial_state):
            raise ValueError("Invalid state length for a square puzzle.")
            
        if goal_state:
            self.goal_state_tuple = tuple(goal_state)
        else:
            sorted_list = list(range(1, self.n * self.n)) + [0]
            self.goal_state_tuple = tuple(sorted_list)
            
        self._goal_positions = {tile: i for i, tile in enumerate(self.goal_state_tuple)}
        self.use_variable_costs = use_variable_costs
        self.optimality_guaranteed = True
        self.make_heuristic_inadmissable = make_heuristic_inadmissable
        if make_heuristic_inadmissable:
            self.h_multiplier = len(initial_state) * (degradation+10)
            self.optimality_guaranteed = False
        else:
            self.h_multiplier = 1    
        self.degradation = degradation    
        cost_type = "VarCost" if use_variable_costs else "UnitCost"
        self._str_repr = f"SlidingTile-{self.n}x{self.n}-{cost_type}-d{degradation}-a{not make_heuristic_inadmissable}"

    def initial_state(self): 
        return self.initial_state_tuple
        
    def goal_state(self): 
        return self.goal_state_tuple
        
    def is_goal(self, state): 
        return state == self.goal_state_tuple

    def get_neighbors(self, state):
        """Returns list of tuples: (neighbor_state, moved_tile_value) from state
        where moved_tile_value is the value (label) of the tile that was moved to the blank space.
        """
        neighbors = []
        blank_index = state.index(0)    # index of blank in state list 
        row, col = divmod(blank_index, self.n)
        moves = {'up': (row - 1, col), 'down': (row + 1, col), 'left': (row, col - 1), 'right': (row, col + 1)}
        for move_dir, (new_row, new_col) in moves.items():
            if 0 <= new_row < self.n and 0 <= new_col < self.n:  # if valid move
                new_blank_index = new_row * self.n + new_col  # index of blank if move this direction
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
        else: 
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

    def heuristic(self, state):
        """
        Calculates the Manhattan distance heuristic (number of steps).
        NOTE: This heuristic counts steps (cost=1). If variable (positive) costs are used,
        its effectiveness will decrease but still admissable since var costs >= unit costs.
        """
        distance = 0
        multiplier = 1
        ignored_tiles = set(range(self.degradation + 1))
        for i, tile in enumerate(state):
            if tile not in ignored_tiles:
                current_pos = divmod(i, self.n)
                goal_idx = self._goal_positions.get(tile)
                if goal_idx is None: continue 
                goal_pos = divmod(goal_idx, self.n)
                if self.make_heuristic_inadmissable:
                    multiplier = i
                distance += multiplier * (abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1]))
        return distance * self.h_multiplier


    def visualise(self, cell_size: int = 10, path: list = None, meeting_node: tuple = None,
                  path_type: str = '', output_file_ext: str = 'png',
                  display: bool = False, return_image: bool = False):
        """ Placeholder - see GridProblem for implemented example"""
        return None


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
                 degradation=0):
        self.initial_state_tuple=tuple(initial_state); self.n=len(initial_state)
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
        cost_type = "VarCost" if use_variable_costs else "UnitCost"
        self._str_repr = f"Pancake-{self.n}-{cost_type}-d{degradation}-a{not make_heuristic_inadmissable}"
        
    def initial_state(self): 
        return self.initial_state_tuple
        
    def goal_state(self): 
        return self.goal_state_tuple
        
    def is_goal(self, state): 
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

    def heuristic(self, state):
        """
        Calculates the Symetric Gap Heuristic (number of adjacent non-consecutive pairs both ways).
        NOTE: This counts number of "breaks". If variable costs (cost=k) are used,
        this heuristic likely becomes non-admissible as one flip (cost k) can fix
        at most 2 gaps.
        """
        #return sum(1 for i in range(self.n-1) if abs(state[i]-state[i+1]) > 1) * self.h_multiplier
        return max(self.gap_heuristic(state, self.goal_state_tuple), 
                   self.gap_heuristic(self.goal_state_tuple, state)) * self.h_multiplier


    def visualise(self, cell_size: int = 10, path: list = None, meeting_node: tuple = None,
                  path_type: str = '', output_file_ext: str = 'png',
                  display: bool = False, return_image: bool = False):
        """ Placeholder - see GridProblem for implemented example"""
        return None


    def __str__(self): 
        return self._str_repr

# --- TowersOfHanoiProblem (Corrected Formatting) ---
class TowersOfHanoiProblem:
    """Implements the Towers of Hanoi problem interface. Cost is always 1."""
    def __init__(self, num_disks, initial_peg='A', target_peg='C', 
                 make_heuristic_inadmissable=False, degradation=0): 
        self.use_variable_costs = False # Cost is always 1
        self.optimality_guaranteed = (not self.use_variable_costs) and (not make_heuristic_inadmissable)
        self.make_heuristic_inadmissable = make_heuristic_inadmissable
        if make_heuristic_inadmissable:
            self.h_multiplier = num_disks * (degradation+10)**2
            self.optimality_guaranteed = False
        else:
            self.h_multiplier = 1    
        self.degradation = degradation    
        assert num_disks >= 1, "Number of disks must be at least 1."
        self.num_disks = num_disks
        self.pegs=['A','B','C']
        assert initial_peg in self.pegs and target_peg in self.pegs and initial_peg != target_peg, \
            f"Invalid initial or target peg. Must be one of {self.pegs} and different."
        self.initial_peg = initial_peg
        self.target_peg = target_peg
        self.aux_peg=next(p for p in self.pegs if p!=initial_peg and p!=target_peg)
        self._initial_state=tuple([initial_peg]*num_disks)  # (A, A, A, ..., A)
        self._goal_state=tuple([target_peg]*num_disks)      # (C, C, C, ..., C)
        self._str_repr=f"TowersOfHanoi-{num_disks}-d{degradation}-a{not make_heuristic_inadmissable}"

    def initial_state(self): 
        return self._initial_state
        
    def goal_state(self): 
        return self._goal_state
        
    def is_goal(self, state): 
        return state == self._goal_state

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
        random.shuffle(pegs)    #  TJH to avoid search bias
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
        random.shuffle(nbs) # Shuffle neighbors to avoid bias in search. TJH Added along with peg order randomisation
        return nbs

    def heuristic(self, state): 
        """Calculates the standard admissible heuristic for Towers of Hanoi.
        Allows for degrading heuristic by ignoring disks
        Allows for inadmissable heuristic but A* still always finds optimal path.. 
        """
        h=0 
        ctp=self.target_peg # current target peg for disk k
        multiplier = 1
        ignored_disks = set(range(self.degradation))

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
        return h * self.h_multiplier

    def get_cost(self, state1, state2, move_info=None): 
        """Cost is always 1 for Towers of Hanoi."""
        return 1
    
    def visualise(self, cell_size: int = 10, path: list = None, meeting_node: tuple = None,
                  path_type: str = '', output_file_ext: str = 'png',
                  display: bool = False, return_image: bool = False):
        """ Placeholder - see GridProblem for implemented example"""
        return None

    def __str__(self): 
        return self._str_repr



