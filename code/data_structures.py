"""
Data Structures
"""
import heapq
import random
import util

from sortedcontainers import SortedDict
from sortedcontainers import SortedKeyList

REMOVED = '^'.encode('utf-8')  # Used to mark an entry as removed in the Ready and Wait structures

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

    def push(self, item, priority, tiebreaker1=0, tiebreaker2=0, prior_f=float('inf'), prior_g=float('inf')):
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
        self.wait = []
        self.ready = []
        self.wait_max_size = 0
        self.ready_max_size = 0
        self.max_bucket_size = 1     # for compatibility with WaitingReadyBuckets
        self.max_distinct_f = 0      # for compatibility with WaitingReadyBuckets
        self.max_distinct_g = 0      # for compatibility with WaitingReadyBuckets
        self.wait_entry_finder = {}  # mapping of state to entry in wait for deletion
        self.ready_entry_finder = {} # mapping of state to entry in ready 
        return

    def remove_task(self, state):
        """ Mark an existing entry as REMOVED. entry format: (f/g, [f/g, fifo/lifo_value, state])"""
        if state in self.wait_entry_finder:
            entry = self.wait_entry_finder.pop(state)
            entry[-1][-1] = REMOVED
        if state in self.ready_entry_finder:
            entry = self.ready_entry_finder.pop(state)
            entry[-1][-1] = REMOVED

    def push(self, item, priority, prior_f=float('inf'), prior_g=float('inf')):
        """ Push item list of [g, fifo/lifovalue, state] onto Wait queue, 
            removing any existing item with matching state first.
            Note: heapq will order by priority then by each element in the item list so order is: 
                  priority, fifo/lifovalue, state
        """
        if prior_f != float('inf'):
            self.remove_task(item[-1])  # 'Remove' the state if it already exists in the wait or ready
        entry = (priority, item)  # entry is (f, [g, fifo/lifo_value, state]) and allowable to update state to 'R' as it's in a list even though nested in a tuple!
        heapq.heappush(self.wait, entry)
        self.wait_entry_finder[item[-1]] = entry
        if self.wait_max_size < len(self.wait):
            self.wait_max_size = len(self.wait)
        return


    def move_to_ready(self, GLB, always_move_equal=False):
        """ Move all states from Wait to Ready that satisfy the GLB condition
            Returns the number of states moved
        """
        count = 0
        while self.wait and self.wait[0][0] < GLB:
            f, (g, ordering, state) = heapq.heappop(self.wait)
            if state != REMOVED:  # Only move if the state is not marked as REMOVED
                del self.wait_entry_finder[state]
                entry = (g, [f, ordering, state])
                heapq.heappush(self.ready, entry)
                self.ready_entry_finder[state] = entry
                count += 1
        if self.version == 'A' or always_move_equal:
            while self.wait and self.wait[0][0] == GLB:
                # If we are in the "all" version and the next item is exactly GLB, we also move it to ready
                f, (g, ordering, state) = heapq.heappop(self.wait)
                if state != REMOVED:
                    del self.wait_entry_finder[state]
                    entry = (g, [f, ordering, state])
                    heapq.heappush(self.ready, entry)
                    self.ready_entry_finder[state] = entry
                    count += 1
        if self.ready_max_size < len(self.ready):
            self.ready_max_size = len(self.ready)
        return count
    
    def move_one_to_ready(self, GLB):
        """ Move one state from Wait to Ready that satisfies the GLB condition
        Returns 1 if a state was moved, 0 otherwise
        """
        while self.wait and self.wait[0][0] <= GLB:
            f, (g, ordering, state) = heapq.heappop(self.wait)
            if state != REMOVED:
                del self.wait_entry_finder[state]
                entry = (g, [f, ordering, state])
                heapq.heappush(self.ready, entry)
                self.ready_entry_finder[state] = entry
                if self.ready_max_size < len(self.ready):
                    self.ready_max_size = len(self.ready)
                return 1
        return 0

    def pop(self, item_only=True):
        """ Pop the lowest priority element from Ready. Entry Format: (g, [f, ordering, state]) 
        """
        state = REMOVED
        while self.ready:
            g, (f, ordering, state) = heapq.heappop(self.ready)   # Pop until we find a valid state that is not marked as REMOVED
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
        while self.wait and self.wait[0][-1][-1] == REMOVED:
            heapq.heappop(self.wait)

        if self.wait:
            if priority_only:
                return self.wait[0][0]
            else:
                return self.wait[0]   # Return the whole entry
        return float('inf')

    def peek_ready(self, priority_only=True):
        """View the lowest priority element on Ready (gmin) without popping it
        after popping any entries marked as REMOVED
        """
        while self.ready and self.ready[0][-1][-1] == REMOVED:
            heapq.heappop(self.ready)

        if self.ready:
            if priority_only:
                return self.ready[0][0]
            else:
                return self.ready[0]     # Return the whole entry
        return float('inf')



class WaitingReadyBuckets:
    """ Two SortedDicts: one for buckets of waiting states and one for buckets of ready states
    Used in LB Pairs family of Bidirectional search algorithms - one of these in each direction
    Wait priority is f and Ready priority is g, so expandable nodes are those in Ready which satisfy 
    g_forward + g_backward + epsilon <= GLB ("C" in A*/"naive BDHS") having already satisfied f_direction <= GLB to be moved from Wait to Ready
    Wait format: wait[f][g]: SortedKeyList( [ [fifo/lifo_value, state], ... ] ) with SKL key=state
    Ready formet: ready[g][f]: SortedKeyList( [ [fifo/lifo_value, state], ... ] ) with SKL key=state
    """
    def __init__(self, version='A'):
        """ version is 'A' for All means move_to_read uses <= GLB, 'F' for First means move_to_ready uses < GLB
        """
        self.version = version
        if self.version not in ['A', 'F']:
            raise ValueError(f"Invalid version: {self.version}. Must be 'A' or 'F'.")
        self.wait = SortedDict()  # SortedDict[key=f] and value=SortedDict[key=g] with value: SKList of entries [fifo/lifo_value, state]
        self.ready = SortedDict()  # SortedDict[key=g] and value=SortedDict[key=f] with value: SKList of entries [fifo/lifo_value, state]
        self.wait_max_size = 0
        self.ready_max_size = 0
        self.wait_curr_size = 0
        self.ready_curr_size = 0
        self.max_bucket_size = 0  # Maximum size of any bucket in wait or ready
        self.max_distinct_f = 0  # Max number of distinct f values in wait at a time - approx as some buckets may be empty
        self.max_distinct_g = 0  # Max number of distinct g values in ready at a time - approx as some buckets may be empty
        return

    def remove_task(self, state, f, g):
        """ Delete an existing entry. entry format: ([fifo/lifo_value, state], f, g)
        Note f,g must be the prior values of the entry to be removed not the current values..
        """
        if f in self.wait and g in self.wait[f]:
            curr_len = len(self.wait[f][g])
            if curr_len:
                idx = self.wait[f][g].bisect_key_left(state)
                if idx < curr_len and self.wait[f][g][idx][-1] == state:
                    entry = self.wait[f][g].pop(idx)
                    self.wait_curr_size -= 1

        if g in self.ready and f in self.ready[g]:
            curr_len = len(self.ready[g][f])
            if curr_len:
                idx = self.ready[g][f].bisect_key_left(state)
                if idx < curr_len and self.ready[g][f][idx][-1] == state:
                    entry = self.ready[g][f].pop(idx)
                    self.ready_curr_size -= 1
        return

    def push(self, item, priority, prior_f=float('inf'), prior_g=float('inf')):
        """ Push item list of [g, fifo/lifovalue, state] onto Wait with priority f, 
            removing any existing item with matching state first.
        """
        g = item[0]
        if prior_f != float('inf'):
            self.remove_task(item[-1], prior_f, prior_g)  # Remove the state from the previous bucket
        entry = [item[1], item[-1]]  # entry is [fifo/lifo_value, state]
        if priority not in self.wait:
            self.wait[priority] = SortedDict()
        if g not in self.wait[priority]:
            self.wait[priority][g] = SortedKeyList(key=lambda e: e[-1])  # SortedKeyList to keep entries sorted by state
        self.wait[priority][g].add(entry)  # Insert the entry in the list of entries in this [f][g] bucket
        self.wait_curr_size += 1
        if self.wait_max_size < self.wait_curr_size:
            self.wait_max_size = self.wait_curr_size
        return

    def move_to_ready(self, GLB, always_move_equal=False):
        """ Move all states from Wait to Ready that satisfy the GLB condition
            Returns the number of states moved
        """
        if len(self.wait) > self.max_distinct_f:
            self.max_distinct_f = len(self.wait)
        count = 0
        while self.wait:
            f = self.wait.peekitem(index=0)[0]  # Get the lowest f value
            if f >= GLB:  # take f < GLB
                break
            g_buckets = self.wait.pop(f)  # Get the SortedDict of g buckets for this f
            for g, entries in g_buckets.items():
                bucket_len = len(entries)
                if bucket_len > 0:
                    if g not in self.ready:
                        self.ready[g] = SortedDict()
                    if f not in self.ready[g] or len(self.ready[g][f]) == 0:
                        self.ready[g][f] = entries
                    else:
                        self.ready[g][f].update(entries)
                    count += bucket_len
                    self.ready_curr_size += bucket_len
                    self.wait_curr_size -= bucket_len
                    if len(self.ready[g][f]) > self.max_bucket_size:
                        self.max_bucket_size = len(self.ready[g][f])

        if self.version == 'A' or always_move_equal:
            while self.wait:
                f = self.wait.peekitem(index=0)[0]  # Get the lowest f value
                if f > GLB:  # take f = GLB
                    break
                g_buckets = self.wait.pop(f)  # Get the SortedDict of g buckets for this f
                for g, entries in g_buckets.items():
                    bucket_len = len(entries)
                    if bucket_len > 0:
                        if g not in self.ready:
                            self.ready[g] = SortedDict()
                        if f not in self.ready[g] or len(self.ready[g][f]) == 0:
                            self.ready[g][f] = entries
                        else:
                            self.ready[g][f].update(entries)
                        count += bucket_len
                        self.ready_curr_size += bucket_len
                        self.wait_curr_size -= bucket_len
                        if len(self.ready[g][f]) > self.max_bucket_size:
                            self.max_bucket_size = len(self.ready[g][f])
        if self.ready_max_size < self.ready_curr_size:
            self.ready_max_size = self.ready_curr_size
        if len(self.ready) > self.max_distinct_g:
            self.max_distinct_g = len(self.ready)
        return count
    
    def move_one_to_ready(self, GLB):
        """ Move one f-g bucket from Wait to Ready that satisfies the GLB condition
            Returns num entries moved if a bucket was moved, 0 otherwise
        """
        if len(self.wait) > self.max_distinct_f:
            self.max_distinct_f = len(self.wait)
        while self.wait:
            f = self.wait.peekitem(index=0)[0]  # Get the lowest f value
            if f > GLB:  # take f <= GLB
                return 0
            if not self.wait[f]:
                self.wait.pop(f)
                continue  # remove if no g buckets
            while self.wait[f]:
                lowest_g = self.wait[f].peekitem(index=0)[0]  # Get the lowest g value in this f bucket
                entries = self.wait[f].pop(lowest_g)  # Get the SortedKeyList of entries
                bucket_len = len(entries)
                if bucket_len > 0:
                    if lowest_g not in self.ready:
                        self.ready[lowest_g] = SortedDict()
                    if f not in self.ready[lowest_g] or len(self.ready[lowest_g][f]) == 0:
                        self.ready[lowest_g][f] = entries
                    else:
                        self.ready[lowest_g][f].update(entries)
                    self.ready_curr_size += bucket_len
                    self.wait_curr_size -= bucket_len
                    if len(self.ready[lowest_g][f]) > self.max_bucket_size:
                        self.max_bucket_size = len(self.ready[lowest_g][f])
                    if self.ready_max_size < self.ready_curr_size:
                        self.ready_max_size = self.ready_curr_size
                    if len(self.ready) > self.max_distinct_g:
                        self.max_distinct_g = len(self.ready)
                    if not self.wait[f]:
                        self.wait.pop(f)
                    return bucket_len  # Return the number of entries moved
                
        return 0  # No entries moved

    def pop_g_level(self, item_only=True):
        """ Pop SortedDict of f buckets in the lowest g from Ready excluding empty buckets. 
            Entry Format: SortedDict[f]: SortedKeyList( [ [ordering, state] ])
        """
        out_dict = SortedDict()  # Create a new SortedDict[f] to hold the popped SortedKeyLists of entries
        while self.ready:
            g = self.ready.peekitem(index=0)[0]  # Get the lowest g value
            f_buckets = self.ready.pop(g)  # Get the SortedDict of f buckets for this g
            for f, entries in f_buckets.items():
                bucket_len = len(entries)
                if bucket_len == 0:    # skip empty f buckets
                    continue
                out_dict[f] = entries  # Add the f bucket (a SortedKeyList) to the output dict
                self.ready_curr_size -= bucket_len
            if out_dict:  # If we have any entries to return
                return out_dict if item_only else (g, out_dict)
        return None  # No entries left in Ready

    def pop(self, item_only=True):
        """ Pop the lowest priority element from Ready[lowest g][lowest f]
        """
        while self.ready:
            g = self.ready.peekitem(index=0)[0]
            if not self.ready[g]:  # self.ready[g] = empty SortedDict, remove and continue
                self.ready.pop(g)
                continue  
            f = self.ready[g].peekitem(index=0)[0]  # Get the lowest f value
            if not self.ready[g][f]:  # self.ready[g][f] = [] Skl is empty, remove and continue
                self.ready[g].pop(f)  
                continue
            ordering, state  = self.ready[g][f].pop(0)  # Pop the first item in the SortedKeyList
            self.ready_curr_size -= 1
            if not self.ready[g][f]:  # self.ready[g][f] = [] Skl is now empty, remove
                self.ready[g].pop(f)  
            if not self.ready[g]:  # self.ready[g] = empty SortedDict now, remove
                self.ready.pop(g)
            if item_only:
                return state
            else:
                return g, f, ordering, state
        return None

    def isEmpty(self):
        """ Check if both Wait and Ready heaps are empty excluding items marked for removal
        """
        return self.wait_curr_size == 0 and self.ready_curr_size == 0

    def peek_wait(self, priority_only=True):
        """View the lowest priority element on Wait (fmin) without popping it 
        after removing empty buckets up till the lowest f and lowest g within f that has entries
        """
        while self.wait:
            f = self.wait.peekitem(index=0)[0]  # Get the lowest f value
            if not self.wait[f]:  # remove if no g buckets
                self.wait.pop(f)
                continue  
            while self.wait[f]:
                g = self.wait[f].peekitem(index=0)[0]
                if len(self.wait[f][g]) == 0:  # remove if no entries in g bucket
                    self.wait[f].pop(g)
                    continue  
                if priority_only:
                    return f
                else:
                    return (f, g, self.wait[f][g])
        return float('inf')


    def peek_ready(self, priority_only=True):
        """View the lowest priority element on Ready (gmin) without popping it
            after removing empty buckets up till the lowest g and lowest f within g that has entries
        """
        while self.ready:
            g = self.ready.peekitem(index=0)[0]  # Get the lowest g value
            if not self.ready[g]:
                self.ready.pop(g)
                continue  # remove if no f buckets
            while self.ready[g]:
                f = self.ready[g].peekitem(index=0)[0]
                if len(self.ready[g][f]) == 0:
                    self.ready[g].pop(f)
                    continue  # remove if no entries in f bucket
                if priority_only:
                    return g
                else:
                    return (f, g, self.ready[g][f])
        return float('inf')



class LBPairs:
    """ Two WaitingReadyPriorityQueue structures, one for forward, one for backward
    Used in LB Pairs family of Bidirectional search algorithms 
    Wait priority is f and Ready priority is g, so expandable nodes are those in Ready which satisfy 
    g_forward + g_backward + epsilon <= GLB ("C" in A*/"naive BDHS") having already satisfied f_direction <= GLB to be moved from Wait to Ready
    Wait priority queue entries are tuples of (f, [g, fifo/lifo_value, state])
    Ready priority queue entries are tuples of (g, [f, fifo/lifo_value, state])

    NOTE: GLB is called min_LB in Chen 2017, LB in Shperberg 2019 and C in A* and naive BDHS
    """
    def __init__(self, version='A', min_edge_cost=1.0, data_struct='P'):
        """ version is 'A' for All means move_to_read uses <= GLB, 'F' for First means move_to_ready uses < GLB
        eps is the minimum edge cost. If unknown can set to 0.0
        data_struct is 'P' for PriorityQueue or 'B' for WaitingReadyBuckets
        """
        if version not in ['A', 'F']:
            raise ValueError(f"Invalid version: {version}. Must be 'A' or 'F'.")
        self.min_edge_cost = min_edge_cost
        if self.min_edge_cost < 0.0:
            raise ValueError(f"Invalid min_edge_cost: {self.min_edge_cost}. Must be >= 0.")
        self.version = version
        self.data_struct = data_struct
        if self.data_struct not in ['P', 'B']:
            raise ValueError(f"Invalid data_struct: {self.data_struct}. Must be 'P' for PQ or 'B' for Buckets.")
        if self.data_struct == 'B':
            self.forward = WaitingReadyBuckets(version)
            self.backward = WaitingReadyBuckets(version)
        else:  # self.data_struct == 'P'    
            self.forward = WaitingReadyPriorityQueue(version)
            self.backward = WaitingReadyPriorityQueue(version)
        return

    def push(self, direction, item, priority, prior_f=float('inf'), prior_g=float('inf')):
        """ Push item list of [g, fifo/lifovalue, state] onto Wait queue with priority f
        """
        if direction == 'F':
            self.forward.push(item, priority, prior_f, prior_g)
        elif direction == 'B':
            self.backward.push(item, priority, prior_f, prior_g)
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
        if self.forward.ready:
            gmin_f = self.forward.peek_ready(priority_only=True)
        else:
            gmin_f = float('inf')
        if self.backward.ready:
            gmin_b = self.backward.peek_ready(priority_only=True)
        else:
            gmin_b = float('inf')
        if self.forward.wait:
            fmin_f = self.forward.peek_wait(priority_only=True)
        else:
            fmin_f = float('inf') 
        if self.backward.wait:
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
            #print(f"Fwd Ready:{self.forward.ready} Fwd Wait:{self.forward.wait}")
            #print(f"Bwd Ready:{self.backward.ready} Bwd Wait:{self.backward.wait}")
            if self.forward.isEmpty() and self.backward.isEmpty():
                break
            if self.forward.ready and self.backward.ready:
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
            #print(f"Fwd Ready:{self.forward.ready} Fwd Wait:{self.forward.wait}")
            #print(f"Bwd Ready:{self.backward.ready} Bwd Wait:{self.backward.wait}")
            if count_f == 0 or count_b == 0:
                CLB = self.get_new_LB()
                #print(f"NEW CLB: {CLB}")
        return found, CLB

    def get_max_heap_size(self):
        """ Get the total size over both forward and backward queues
        """
        return sum([self.forward.wait_max_size, self.forward.ready_max_size,
                   self.backward.wait_max_size, self.backward.ready_max_size])
    
    def get_max_bucket_stats(self):
        """ Get the maximum size of any bucket in either forward or backward queues + max distinct f and g values
        """
        return (max(self.forward.max_bucket_size, self.backward.max_bucket_size), 
                max(self.forward.max_distinct_f, self.backward.max_distinct_f), 
                max(self.forward.max_distinct_g, self.backward.max_distinct_g) )
        

    

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



"""
fwd = WaitingReadyBuckets('F')

print(f"WAIT max:{fwd.wait_max_size} curr:{fwd.wait_curr_size} READY max:{fwd.ready_max_size} curr:{fwd.ready_curr_size}")
print(f"WAIT f keys:{list(fwd.wait.keys())}")
print(f"WAIT:{fwd.wait}")
print(f"READY g keys:{list(fwd.ready.keys())}")
print(f"READY:{fwd.ready}") 

fwd.push([0, 0, 'hh'], 100, float('inf'), float('inf'))
fwd.push([10, 0, 'jj'], 90, float('inf'), float('inf'))
fwd.push([9, 0, 'jj'], 88, 90, 10)
fwd.push([9, 0, 'kk'], 70, float('inf'), float('inf'))  #same g as jj but lower f
fwd.push([9, 0, 'll'], 70, float('inf'), float('inf'))  #same g as jj, kk same f as kk
fwd.push([8, 0, 'll'], 69, 70, 9) # update ll with new f,g. bucket splits correctly
fwd.push([8, 0, 'mm'], 69, float('inf'), float('inf')) # 2 items in same f,g bucket
fwd.wait[68] = SortedDict()  # Add a new f bucket with no entries
fwd.wait[67] = SortedDict()
fwd.wait[67][667] = SortedKeyList(key=lambda e: e[-1])  # Add a new g bucket with no entries
fwd.peek_wait(priority_only=False) # correct: (69, 8, SortedKeyList([[0, 'll'], [0, 'mm']]

fwd.move_to_ready(70)  # 2 Move all entries with f < 70 to ready
fwd.move_to_ready(70, always_move_equal=True) # 1
fwd.peek_ready(priority_only=True) # 8
fwd.peek_ready(priority_only=False) # ll mm
fwd.ready[7] = SortedDict()  # Add a new f bucket with no entries
fwd.ready[6] = SortedDict()
fwd.ready[7][777] = SortedKeyList(key=lambda e: e[-1])  # Add a new g bucket with no entries
fwd.peek_ready(priority_only=False) # 8: ll mm
fwd.move_one_to_ready(88) # 66: jj
fwd.move_one_to_ready(88) # correctly removes empty bucket, moves nothing
fwd.move_one_to_ready(90) # correctly removes empty bucket, moves nothing

fwd.move_to_ready(88, always_move_equal=True) # 0 correct and removes empty
fwd.move_to_ready(90, always_move_equal=True) # 0 correct and removes empty

fwd.pop(item_only=False) # correct: (8, 69, 0, 'll')
fwd.pop(item_only=False) # correct (8, 69, 0, 'mm')
fwd.pop(item_only=False) # correct, (9, 70, 0, 'kk') and removed empty [8][69] skl
fwd.pop(item_only=False) # correct, (9, 88, 0, 'jj') and removed empty [9][70] skl
fwd.pop(item_only=False) # correct, returns None and removed empty [9][88] skl


fwd.move_to_ready(90, always_move_equal=True)
fwd.move_one_to_ready(90)
fwd.move_one_to_ready(100)  # correctly moves 1. wait now empty, ready has 1
fwd.isEmpty() # False
fwd.pop(item_only=False)    $ correct: (0, 100, 0, 'hh')
fwd.isEmpty() # True
fwd.move_to_ready(100, always_move_equal=True)  # removed final empty bucket in wait
fwd.pop(item_only=False)  # removed final empty bucket in ready, returns None

frontier = LBPairs('F', 1.0, 'B')
frontier.data_struct
frontier.push('F', [0, 0, 'hh'], 100, float('inf'), float('inf'))
frontier.push('B', [0, 0, 'hg'], 100, float('inf'), float('inf'))

print(f"##### FORWARD #####")
print(f"WAIT max:{frontier.forward.wait_max_size} curr:{frontier.forward.wait_curr_size} READY max:{frontier.forward.ready_max_size} curr:{frontier.forward.ready_curr_size}")
print(f"WAIT f keys:{list(frontier.forward.wait.keys())}")
print(f"WAIT:{frontier.forward.wait}")
print(f"READY g keys:{list(frontier.forward.ready.keys())}")
print(f"READY:{frontier.forward.ready}") 
print(f"##### BACKWARD #####")
print(f"WAIT max:{frontier.backward.wait_max_size} curr:{frontier.backward.wait_curr_size} READY max:{frontier.backward.ready_max_size} curr:{frontier.backward.ready_curr_size}")
print(f"WAIT f keys:{list(frontier.backward.wait.keys())}")
print(f"WAIT:{frontier.backward.wait}")
print(f"READY g keys:{list(frontier.backward.ready.keys())}")
print(f"READY:{frontier.backward.ready}") 

frontier.prepare_expandable(0) # (True, 100)

frontier.pop('F', item_only=False) # (0, 100, 0, 'hh')
frontier.pop('B', item_only=False) # (0, 100, 0, 'hg')

frontier.push('F', [1, 0, 'f1'], 99, float('inf'), float('inf'))
frontier.push('B', [1, 0, 'b1'], 99, float('inf'), float('inf'))

frontier.prepare_expandable(0) # (True, 99)

frontier.pop('F', item_only=False) # (1, 99, 0, 'f1')
frontier.pop('B', item_only=False) # (1, 99, 0, 'b1')

frontier.push('F', [2, 0, 'f1'], 98, float('inf'), float('inf'))
frontier.push('B', [2, 0, 'b1'], 98, float('inf'), float('inf'))

frontier.prepare_expandable(0) #(True, 98)

frontier.pop('F', item_only=False) # (2, 98, 0, 'f1')
frontier.pop('B', item_only=False) # (2, 98, 0, 'b1')

"""