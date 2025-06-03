"""
Data Structures
"""
import heapq
import random
import util

REMOVED = '^'  # Used to mark an entry as removed in the Ready and Wait priority queues

class PriorityQueue:
    """ Priority Queue implementation optionally supporting 3 levels of priority: 
            heuristic value, tiebreaker1, tiebreaker2
            tb2 is currently purely internally calculated for fifo/lifo
            the tb1 value is always passed in by the caller but setting up the PriorityQueue with 
            A tiebreakerx of 'FIFO' OR 'LIFO' mode will set self.count_tbx to be incremented or decremented 
            so that the caller can access it and pass it in as the tiebreaker1 value
            A tiebreaker of "R" with set the tiebreaker1 to a random number between 0 and rand_upper_bound
            A priority_key of 'LIFO' makes a stack and 'FIFO' a queue
    ie list of tuples (priority, tiebreaker1, tiebreaker2, item)
    """
    def __init__(self, priority_key='f', tiebreaker1='FIFO', tiebreaker2='NONE', rand_upper_bound=100000000000):
        self.heap = []
        self.rand_upper_bound = rand_upper_bound

        self.priority_key = priority_key
        self.next_priority = 0
        if priority_key == 'FIFO':
            self.increment_priority = 1
        elif priority_key == 'LIFO':
            self.increment_priority = -1
        else:
            self.increment_priority = 0

        self.tiebreaker1 = tiebreaker1
        self.next_tb1 = 0
        if tiebreaker1 == 'FIFO':
            self.increment_tb1 = 1
        elif tiebreaker1 == 'LIFO':
            self.increment_tb1 = -1
        else:
            self.increment_tb1 = 0

        self.tiebreaker2 = tiebreaker2
        self.use_tb2 = False
        if tiebreaker2 != 'NONE':
            self.use_tb2 = True

        self.next_tb2 = 0
        if tiebreaker2 == 'FIFO':
            self.increment_tb2 = 1
        elif tiebreaker2 == 'LIFO':
            self.increment_tb2 = -1
        else:
            self.increment_tb2 = 0

        self.max_heap_size = 0
        return

    def push(self, item, priority, tiebreaker1=0, tiebreaker2=0):
        if self.use_tb2:
            entry = (priority, tiebreaker1, tiebreaker2, item)
        else:
            entry = (priority, tiebreaker1, item)

        heapq.heappush(self.heap, entry)
        if self.max_heap_size < len(self.heap):
            self.max_heap_size = len(self.heap)
        return

    def pop(self, item_only=True):
        if self.use_tb2:
            priority, tiebreaker1, tiebreaker2, item = heapq.heappop(self.heap)
        else:
            priority, tiebreaker1, item = heapq.heappop(self.heap)
            tiebreaker2 = None
        if item_only:
            return item
        return item, priority, tiebreaker1, tiebreaker2

    def isEmpty(self):
        return len(self.heap) == 0

    def peek(self, priority_only=True):
        """View the lowest priority element without popping it
        """
        if not self.isEmpty():
            if priority_only:
                return self.heap[0][0]  
            else:
                # Return the whole entry (priority, tiebreaker1, tiebreaker2, item)
                return self.heap[0]
        return None

    def calc_priority(self, g, h=0):
        """Calculates the priority value based on heap vars and/or the type and values of g and h
        Called from the search algo prior to calling the push method
        """
        if self.priority_key == 'g':
            return g
        elif self.priority_key == '-g':  # higher g popped first
            return -g
        elif self.priority_key == 'h':
            return h
        elif self.priority_key == 'f':
            return g + h
        elif self.priority_key in ['FIFO', 'LIFO']:
            self.next_priority += self.increment_priority
            return self.next_priority
        elif self.priority_key == 'R':
            self.next_priority = random.randint(0, self.rand_upper_bound)
            return self.next_priority
        elif self.priority_key == 'NONE':
            return 0
        else:
            raise ValueError(f"Invalid priority_key: {self.priority_key}")


    def calc_tiebreak1(self, g, h=0):
        """Calculates the tiebreaker1 value based on tiebreaker type and/or the type and values of g and h
        Called from the search algo prior to calling the push method
        """
        if self.tiebreaker1 == 'g':
            return g
        elif self.tiebreaker1 == '-g':  # higher g popped first
            return -g
        elif self.tiebreaker1 == 'h':
            return h
        elif self.tiebreaker1 == 'f':
            return g + h
        elif self.tiebreaker1 in ['FIFO', 'LIFO']:
            self.next_tb1 += self.increment_tb1
            return self.next_tb1
        elif self.tiebreaker1 == 'R':
            self.next_tb1 = random.randint(0, self.rand_upper_bound)
            return self.next_tb1
        elif self.tiebreaker1 == 'NONE':
            return 0
        else:
            raise ValueError(f"Invalid tiebreaker1: {self.tiebreaker1}")

    def calc_tiebreak2(self, g, h=0):
        """Calculates the tiebreaker2 value based on tiebreaker type and/or the type and values of g and h
        Called from the search algo prior to calling the push method (NOTE: tb2 is UNUSED by any algorithms currently!)
        """
        if self.tiebreaker2 == 'g':
            return g
        elif self.tiebreaker2 == '-g':  # higher g popped first
            return -g
        elif self.tiebreaker2 == 'h':
            return h
        elif self.tiebreaker2 == 'f':
            return g + h
        elif self.tiebreaker2 in ['FIFO', 'LIFO']:
            self.next_tb2 += self.increment_tb2
            return self.next_tb2
        elif self.tiebreaker2 == 'R':
            self.next_tb2 = random.randint(0, self.rand_upper_bound)
            return self.next_tb2
        elif self.tiebreaker2 == 'NONE':
            return 0
        else:
            raise ValueError(f"Invalid tiebreaker2: {self.tiebreaker2}")


class WaitingReadyPriorityQueue:
    """ Two priority queues: one for waiting states and one for ready states
    Used in LB Pairs family of Bidirectional search algorithms - one of these in each direction
    Wait priority is f and Ready priority is g, so expandable nodes are those in Ready which satisfy 
    g_forward + g_backward + epsilon <= GLB ("C" in A*/"naive BDHS") having already satisfied f_direction <= GLB to be moved from Wait to Ready
    Wait priority queue entries are tuples of (f, [g, fifo/lifo_value, state])
    Ready priority queue entries are tuples of (g, [f, fifo/lifo_value, state])
    """
    def __init__(self, version='A'):
        """ version is 'A' for All means move_to_read uses <= GLB, 'F' for First means move_to_ready uses < GLB
        """
        self.version = version
        if self.version not in ['A', 'F']:
            raise ValueError(f"Invalid version: {self.version}. Must be 'A' or 'F'.")
        self.wait_heap = []
        self.ready_heap = []
        self.wait_max_heap_size = 0
        self.ready_max_heap_size = 0
        self.wait_entry_finder = {}  # mapping of state to entry in wait_heap for deletion
        self.ready_entry_finder = {} # mapping of state to entry in ready_heap 
        return

    def remove_task(self, state):
        """ Mark an existing entry as REMOVED. entry format: (f/g, [f/g, fifo/lifo_value, state])"""
        if state in self.wait_entry_finder:
            entry = self.wait_entry_finder.pop(state)
            entry[-1][-1] = REMOVED
        if state in self.ready_entry_finder:
            entry = self.ready_entry_finder.pop(state)
            entry[-1][-1] = REMOVED

    def push(self, item, priority):
        """ Push item list of [g, fifo/lifovalue, state] onto Wait queue, removing any existing item with matching state first.
        Note: heapq will order by priority then by each element in the item list so equivalent to priority, fifo/lifovalue, state
        """
        self.remove_task(item[-1])  # 'Remove' the state if it already exists in the wait_heap or ready_heap
        entry = (priority, item)  # entry is (f, [g, fifo/lifo_value, state]) and allowable to update state to 'R' as it's in a list even though nested in a tuple!
        heapq.heappush(self.wait_heap, entry)
        self.wait_entry_finder[item[-1]] = entry
        if self.wait_max_heap_size < len(self.wait_heap):
            self.wait_max_heap_size = len(self.wait_heap)
        return

    def move_to_ready(self, GLB, always_move_equal=False):
        """ Move all states from Wait to Ready that satisfy the GLB condition
        Returns the number of states moved
        """
        count = 0
        while self.wait_heap and self.wait_heap[0][0] < GLB:
            f, (g, ordering, state) = heapq.heappop(self.wait_heap)
            if state != REMOVED:  # Only move if the state is not marked as REMOVED
                del self.wait_entry_finder[state]
                entry = (g, [f, ordering, state])
                heapq.heappush(self.ready_heap, entry)
                self.ready_entry_finder[state] = entry
                count += 1
        if self.version == 'A' or always_move_equal:
            while self.wait_heap and self.wait_heap[0][0] == GLB:
                # If we are in the "all" version and the next item is exactly GLB, we also move it to ready
                f, (g, ordering, state) = heapq.heappop(self.wait_heap)
                if state != REMOVED:
                    del self.wait_entry_finder[state]
                    entry = (g, [f, ordering, state])
                    heapq.heappush(self.ready_heap, entry)
                    self.ready_entry_finder[state] = entry
                    count += 1
        if self.ready_max_heap_size < len(self.ready_heap):
            self.ready_max_heap_size = len(self.ready_heap)
        return count
    
    def move_one_to_ready(self, GLB):
        """ Move one state from Wait to Ready that satisfies the GLB condition
        Returns 1 if a state was moved, 0 otherwise
        """
        while self.wait_heap and self.wait_heap[0][0] <= GLB:
            f, (g, ordering, state) = heapq.heappop(self.wait_heap)
            if state != REMOVED:
                del self.wait_entry_finder[state]
                entry = (g, [f, ordering, state])
                heapq.heappush(self.ready_heap, entry)
                self.ready_entry_finder[state] = entry
                if self.ready_max_heap_size < len(self.ready_heap):
                    self.ready_max_heap_size = len(self.ready_heap)
                return 1
        return 0

    def pop(self, item_only=True):
        """ Pop the lowest priority element from Ready. Entry Format: (g, [f, ordering, state]) 
        """
        state = REMOVED
        while self.ready_heap:
            g, (f, ordering, state) = heapq.heappop(self.ready_heap)   # Pop until we find a valid state that is not marked as REMOVED
            if state != REMOVED:
                del self.ready_entry_finder[state]
                break
        if state != REMOVED:
            if item_only:
                return state
            else:
                return g, f, ordering, state
        return None


    def isEmpty(self):
        """ Check if both Wait and Ready heaps are empty excluding items marked for removal
        """
        return len(self.wait_entry_finder) == 0 and len(self.ready_entry_finder) == 0

    def peek_wait(self, priority_only=True):
        """View the lowest priority element on Wait (fmin) without popping it 
        after popping any entries marked as REMOVED
        """
        while self.wait_heap and self.wait_heap[0][-1][-1] == REMOVED:
            heapq.heappop(self.wait_heap)

        if self.wait_heap:
            if priority_only:
                return self.wait_heap[0][0]
            else:
                return self.wait_heap[0]   # Return the whole entry
        return 0

    def peek_ready(self, priority_only=True):
        """View the lowest priority element on Ready (gmin) without popping it
        after popping any entries marked as REMOVED
        """
        while self.ready_heap and self.ready_heap[0][-1][-1] == REMOVED:
            heapq.heappop(self.ready_heap)

        if self.ready_heap:
            if priority_only:
                return self.ready_heap[0][0]
            else:
                return self.ready_heap[0]     # Return the whole entry
        return None


class LBPairs:
    """ Two WaitingReadyPriorityQueue structures, one for forward, one for backward
    Used in LB Pairs family of Bidirectional search algorithms 
    Wait priority is f and Ready priority is g, so expandable nodes are those in Ready which satisfy 
    g_forward + g_backward + epsilon <= GLB ("C" in A*/"naive BDHS") having already satisfied f_direction <= GLB to be moved from Wait to Ready
    Wait priority queue entries are tuples of (f, [g, fifo/lifo_value, state])
    Ready priority queue entries are tuples of (g, [f, fifo/lifo_value, state])

    NOTE: GLB is called C_LB in Chen 2017, LB in Shperberg 2019 and C in A* and naive BDHS
    """
    def __init__(self, version='A', min_edge_cost=1.0):
        """ version is 'A' for All means move_to_read uses <= GLB, 'F' for First means move_to_ready uses < GLB
        eps is the minimum edge cost. If unknown can set to 0.0
        """
        if version not in ['A', 'F']:
            raise ValueError(f"Invalid version: {version}. Must be 'A' or 'F'.")
        self.min_edge_cost = min_edge_cost
        if self.min_edge_cost < 0.0:
            raise ValueError(f"Invalid min_edge_cost: {self.min_edge_cost}. Must be >= 0.")
        self.version = version
        self.forward = WaitingReadyPriorityQueue(version)
        self.backward = WaitingReadyPriorityQueue(version)
        return

    def push(self, direction, item, priority):
        """ Push item list of [g, fifo/lifovalue, state] onto Wait queue with priority f
        """
        if direction == 'F':
            self.forward.push(item, priority)
        elif direction == 'B':
            self.backward.push(item, priority)
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'F' or 'B'.")
        return

    def move_to_ready(self, GLB, always_move_equal=False):
        """ Move all states from Wait to Ready that satisfy the < or <= GLB condition in each direction
        Returns the number of states moved in each direction (countF, CountB)
        """
        count_f = self.forward.move_to_ready(GLB, always_move_equal)
        count_b = self.backward.move_to_ready(GLB, always_move_equal)
        return count_f, count_b

    def move_one_to_ready(self, GLB):
        """ Move one state from Wait to Ready that satisfies the <= GLB condition in each direction
        Returns the number of states moved in each direction (countF, CountB)
        """
        count_f = self.forward.move_one_to_ready(GLB)
        count_b = self.backward.move_one_to_ready(GLB)
        return count_f, count_b


    def pop(self, direction, item_only=True):
        """ Pop the lowest priority element from Ready in the specified direction
        """
        if direction == 'F':
            return self.forward.pop(item_only)
        elif direction == 'B':
            return self.backward.pop(item_only)
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'F' or 'B'.")

    def get_new_LB(self):
        """ Get the new CLB value (the final CLB in prepare_expandable is the new GLB)
            NOTE: GLB is called min_LB in Chen 2017, LB in Shperberg 2019 and C in A* and naive BDHS        
        """
        if self.forward.ready_heap:
            gmin_f = self.forward.peek_ready(priority_only=True)
        else:
            gmin_f = float('inf')
        if self.backward.ready_heap:
            gmin_b = self.backward.peek_ready(priority_only=True)
        else:
            gmin_b = float('inf')
        if self.forward.wait_heap:
            fmin_f = self.forward.peek_wait(priority_only=True)
        else:
            fmin_f = float('inf') 
        if self.backward.wait_heap:
            fmin_b = self.backward.peek_wait(priority_only=True)
        else:
            fmin_b = float('inf')
        return min(fmin_f, fmin_b, gmin_f + gmin_b + self.min_edge_cost)

    def prepare_expandable(self, GLB):
        """ Prepare the expandable nodes for the next iteration
            GLB is min(lb(u,v)). lb(u,v) = max(fmin_f, fmin_b, gmin_f + gmin_b + min_edge_cost)

            Returns found=True if there are expandable nodes in each ready queue along with the next GLB value
        """
        CLB = 0
        found = False

        while True:
            count_f, count_b = self.move_to_ready(CLB)
            #print(f"After initial move to ready Moved:{count_f} {count_b}")
            #print(f"Fwd Ready:{self.forward.ready_heap} Fwd Wait:{self.forward.wait_heap}")
            #print(f"Bwd Ready:{self.backward.ready_heap} Bwd Wait:{self.backward.wait_heap}")
            if self.forward.isEmpty() and self.backward.isEmpty():
                break
            if self.forward.ready_heap and self.backward.ready_heap:
                gmin = self.min_edge_cost
                gmin += self.forward.peek_ready(priority_only=True)
                gmin += self.backward.peek_ready(priority_only=True)
                if gmin <= CLB: # This is the condition for expandable nodes
                    found = True
                    #print(f"Expandable nodes found with GLB:{CLB} g+g:{gmin}")
                    break
            #count_f, count_b = self.move_to_ready(CLB, always_move_equal=True)
            if self.version == 'F':
                count_f, count_b = self.move_one_to_ready(CLB)
            else:
                count_f, count_b = 0, 0
            #print(f"After next move to ready Moved:{count_f} {count_b}")
            #print(f"Fwd Ready:{self.forward.ready_heap} Fwd Wait:{self.forward.wait_heap}")
            #print(f"Bwd Ready:{self.backward.ready_heap} Bwd Wait:{self.backward.wait_heap}")
            if count_f == 0 or count_b == 0:
                CLB = self.get_new_LB()
                #print(f"NEW CLB: {CLB}")
        return found, CLB

    def get_max_heap_size(self):
        """ Get the total size over both forward and backward queues
        """
        return sum([self.forward.wait_max_heap_size, self.forward.ready_max_heap_size,
                   self.backward.wait_max_heap_size, self.backward.ready_max_heap_size])

    



class StateInfo():
    """ Dict with state key to store g values and parent info for path reconstruction
    This was supposed to save a few GB of RAM on big problems over having two dicts with key=state as the state isn't duplicated
    but in reality took more RAM. So not used now! 
    """
    def __init__(self):
        self.state_dict = {}
        return
    
    def add(self, state, parent=None, g=0):
        """ Always adding or updating both parent and g at once """
        self.state_dict[state] = {'parent': parent, 'g': g}
        return

    def get_g(self, state, noval=float('inf')):
        state_info = self.state_dict.get(state, None)
        if state_info:
            return state_info.get('g', noval)
        return noval

    def get_parent(self, state, noval=None):
        state_info = self.state_dict.get(state, None)
        if state_info:
            return state_info.get('parent', noval)
        return noval
