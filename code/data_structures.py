"""
Data Structures
"""
import heapq
import random
from collections import namedtuple

from numpy import fmin


import util

from sortedcontainers import SortedDict
from sortedcontainers import SortedKeyList
from sortedcontainers import SortedSet

REMOVED = '^'.encode('utf-8')  # Used to mark an entry as removed in the Ready and Wait structures

NodeData = namedtuple('NodeData', ['g', 'h', 'parent'])

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

        self.entry_finder = {} # mapping of state to entry in heap
        return

    def remove_task(self, state):
        """ Mark an existing entry as REMOVED. entry format: (f/g, tb1, <tb2,> [state])"""
        if state in self.entry_finder:
            entry = self.entry_finder.pop(state)
            entry[-1][-1] = REMOVED


    def push(self, item, priority, tiebreaker1=0, tiebreaker2=0, prior_g=float('inf')):
        """ item = state """
        if prior_g != float('inf'):
            self.remove_task(item)  # 'Remove' the state if it already exists
        if self.use_tb2:
            entry = (priority, tiebreaker1, tiebreaker2, [item]) # push as list so can set to removed...
        else:
            entry = (priority, tiebreaker1, [item])
        heapq.heappush(self.heap, entry)
        self.entry_finder[item] = entry
        if self.max_heap_size < len(self.heap):
            self.max_heap_size = len(self.heap)
        return

    def pop(self, item_only=True):
        state = REMOVED
        while self.heap:
            if self.use_tb2:
                priority, tiebreaker1, tiebreaker2, item = heapq.heappop(self.heap)
            else:
                priority, tiebreaker1, item = heapq.heappop(self.heap)
                tiebreaker2 = None
            state = item[-1]
            if state != REMOVED:
                del self.entry_finder[state]
                break
        if state != REMOVED:
            if item_only:
                return state
            else:
                return state, priority, tiebreaker1, tiebreaker2
        return None

    def isEmpty(self):
        return len(self.entry_finder) == 0  #len(self.heap) == 0

    def peek(self, priority_only=True):
        """View the lowest priority element without popping it
        heap entries: (priority, tiebreaker1, tiebreaker2, [item])
        """
        while self.heap and self.heap[0][-1][-1] == REMOVED:
            heapq.heappop(self.heap)

        if self.heap:
            if priority_only:
                return self.heap[0][0]  
            else:
                
                if self.use_tb2:
                    priority, tiebreaker1, tiebreaker2, item = self.heap[0]  
                else:
                    priority, tiebreaker1, item = self.heap[0]
                    tiebreaker2 = None
                state = item[-1]
                return priority, tiebreaker1, tiebreaker2, state  # Return the whole entry (priority, tiebreaker1, tiebreaker2, item[-1])
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
    """ Priority queue implementation used for Two priority queues: one for waiting states and one for ready states
    Used in LB Pairs family of Bidirectional search algorithms - two of these in each direction
    Wait priority is f and Ready priority is g, so expandable nodes are those in Ready which satisfy 
    g_forward + g_backward + epsilon <= GLB ("C" in A*/"naive BDHS") having already satisfied f_direction <= GLB to be moved from Wait to Ready
    Wait priority queue entries are tuples of (f, [g, ordering, state]) where ordering is the tiebreak value
    Ready priority queue entries are tuples of (g, [ordering, f, state])
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
        """ Mark an existing entry as REMOVED. entry format: 
        Wait: (f, [g, ordering, state])
        ready: (g, [ordering, f, state])
        """
        if state in self.wait_entry_finder:
            entry = self.wait_entry_finder.pop(state)
            entry[-1][-1] = REMOVED
        if state in self.ready_entry_finder:
            entry = self.ready_entry_finder.pop(state)
            entry[-1][-1] = REMOVED

    def push(self, item, priority, prior_f=float('inf'), prior_g=float('inf')):
        """ Push item list of [g, ordering, state] onto Wait queue, 
            removing any existing item with matching state first.
        """
        if prior_g != float('inf'):
            self.remove_task(item[-1])  # 'Remove' the state if it already exists in wait or ready
        entry = (priority, item)  # entry is (f, [g, ordering, state]) and allowable to update state to REMOVED as it's in a list even though nested in a tuple
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
                entry = (g, [ordering, f, state])       # ordering before f in ready
                heapq.heappush(self.ready, entry)
                self.ready_entry_finder[state] = entry
                count += 1
        if self.version == 'A' or always_move_equal:
            while self.wait and self.wait[0][0] == GLB:
                # If we are in the "all" version and the next item is exactly GLB, we also move it to ready
                f, (g, ordering, state) = heapq.heappop(self.wait)
                if state != REMOVED:
                    del self.wait_entry_finder[state]
                    entry = (g, [ordering, f, state])   # ordering before f in ready
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
                entry = (g, [ordering, f, state])   # ordering before f in ready
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
            g, (ordering, f, state) = heapq.heappop(self.ready)   # Pop until we find a valid state that is not marked as REMOVED
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
    
    def curr_size(self):
        return len(self.wait_entry_finder) + len(self.ready_entry_finder)

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

    def select_and_order(self, direction, lb):
        """ Select and return nodes to expand.
            For heap-based queue only tb_select = "F" is supported which is the first node in the lowest ordering in the lowest g level
            however tiebreaking can occur over all f values in a glevel unlike the bucket implementation.
            Note: a tiebreak of NONE means all ordering values are 0, hence effective ordering is g, f
        """
        # expand_nodes entry format (ordering, g, f, state)
        expand_nodes = SortedKeyList(key=lambda e: e[0])  # SortedKeyList to keep entries sorted by ordering
        g, f, ordering, current_state = self.pop(item_only=False)
        expand_nodes.add( (ordering, g, f, current_state) )
        return expand_nodes


class WaitingReadyBuckets:
    """ Two SortedDicts: one for buckets of waiting states and one for buckets of ready states
    Used in LB Pairs family of Bidirectional search algorithms - one of these in each direction
    Wait priority is f and Ready priority is g, so expandable nodes are those in Ready which satisfy 
    g_forward + g_backward + epsilon <= GLB ("C" in A*/"naive BDHS") having already satisfied f_direction <= GLB to be moved from Wait to Ready
    Wait format: wait[f][g]: SortedKeyList( [ [ordering, state], ... ] ) with SKL key=state and ordering = tiebreak value (within bucket inlike the PQ implementation where ordering works over whole glevel)
    Ready formet: ready[g][f]: SortedKeyList( [ [ordering, state], ... ] ) with SKL key=state
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
        self.max_f_in_ready_g = 0  # Max number of f buckets in any ready[g]
        self.expand_nodes = SortedKeyList(key=lambda e: e[0])  # SortedKeyList to keep entries (ordering, g, f, state) sorted by fifo/lifo/rand/0 value

        return


    def remove_task(self, state, f, g):
        """ Delete an existing entry. entry format: ([ordering, state], f, g)
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
        """ Push item list of [g, ordering, state] onto Wait with priority f, 
            removing any existing item with matching state first.
        """
        g = item[0]
        if prior_g != float('inf'):
            self.remove_task(item[-1], prior_f, prior_g)  # Remove the state from the previous bucket
        entry = [item[1], item[-1]]  # entry is [ordering, state]
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

    def find_idx(self, g, f, state):
        """ Returns the index of state in Ready[g][f] """
        if g in self.ready and f in self.ready[g]:
            curr_len = len(self.ready[g][f])
            if curr_len:
                idx = self.ready[g][f].bisect_key_left(state)
                if idx < curr_len and self.ready[g][f][idx][-1] == state:
                    return idx
        return None
    
    def find_lowest_ordered_idx(self, g, f):
        """ Return the index of the state in Ready[g][f] that has the lowest ordering in that g-f bucket
        """
        bucket = list(self.ready[g][f])  # fast copy of g-f bucket [ [ordering, state], ... ] 
        heapq.heapify(bucket)            # faster than full sorting and works since we only need to know the lowest, not the full order
        if len(bucket) > 0 and len(bucket[0]) > 0:
            state = bucket[0][-1]
            return self.find_idx(g, f, state)
        else:
            raise ValueError(f"WaitingReadyBuckets.find_lowest_ordered_idx g:{g} f:{f} bucket:'{bucket}' self.ready[g][f]:{self.ready[g][f]}")

    def pop_g_level(self, g):
        """ Pop all f buckets in the selected g from Ready ignoring empty buckets. 
            Adds to self.expand_nodes and return True
                self.expand_nodes:  entry format (ordering, g, f, state) sorted by ordering
        """
        if self.ready:
            f_buckets = self.ready.pop(g)  # Get the SortedDict of f buckets for this g 
            for f, bucket in f_buckets.items():   # implicitly removes ready[g] = empty SortedDict()
                bucket_len = len(bucket)
                if bucket_len == 0:    # skip empty f buckets - implicitly removes ready[g][f] = empty SKList()
                    continue
                self.ready_curr_size -= bucket_len
                for (ordering, state) in bucket:
                    self.expand_nodes.add( (ordering, g, f, state) )
            return True
        return None  # No entries left in Ready

    def pop_node_or_bucket(self, g, f, bucket=False, idx = -1):
        """ Pop an element or bucket from Ready[g][f] and [idx] if bucket = False
            Adds to self.expand_nodes and returns True
                self.expand_nodes:  entry format (ordering, g, f, state) sorted by ordering
        """
        if self.ready:
            if not self.ready[g]:  # self.ready[g] = empty SortedDict, remove and continue
                self.ready.pop(g)
                return None
            elif not self.ready[g][f]:  # self.ready[g][f] = [] Skl is empty, remove and continue
                self.ready[g].pop(f)
                return None
            if not bucket:
                ordering, state  = self.ready[g][f].pop(idx)  # Pop the idx-th item in the SortedKeyList
                self.ready_curr_size -= 1
                if not self.ready[g][f]:  # self.ready[g][f] = [] Skl is now empty, remove
                    self.ready[g].pop(f)
                if not self.ready[g]:  # self.ready[g] = empty SortedDict now, remove
                    self.ready.pop(g)
                self.expand_nodes.add( (ordering, g, f, state) )
            else: # pop whole bucket
                bucket = self.ready[g].pop(f)    # sklist [ [ordering, state], ...  ]
                self.ready_curr_size -= len(bucket)
                for (ordering, state) in bucket:
                    self.expand_nodes.add( (ordering, g, f, state) )
                if not self.ready[g]:  # self.ready[g] = empty SortedDict now, remove
                    self.ready.pop(g)
            return True 
        return None



    def pop(self, item_only=True, bucket=False):
        """ Pop the lowest priority element or bucket from Ready[lowest g][lowest f]
            If bucket, will add to self.expand_nodes and return True
                self.expand_nodes:  entry format (ordering, g, f, state) sorted by ordering
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
            if not bucket:
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
            else: # pop whole bucket
                bucket = self.ready[g].pop(f)    # sklist [ [ordering, state], ...  ]
                self.ready_curr_size -= len(bucket)
                for (ordering, state) in bucket:
                    self.expand_nodes.add( (ordering, g, f, state) )
                if not self.ready[g]:  # self.ready[g] = empty SortedDict now, remove
                    self.ready.pop(g)
                return True 
        return None

    def isEmpty(self):
        """ Check if both Wait and Ready heaps are empty excluding items marked for removal
        """
        return self.wait_curr_size == 0 and self.ready_curr_size == 0

    def curr_size(self):
        return self.wait_curr_size + self.ready_curr_size

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
        """View the lowest priority element on Ready (gmin) (or at idx) without popping it
            after removing empty buckets up till the lowest g and lowest f within g that has entries
            ie will either return the lowest g that has at least one non-empty f bucket or float('inf') if wait is empty
        """
        while self.ready:
            g = self.ready.peekitem(index=0)[0]  # Get the lowest g value
            if not self.ready[g]:
                self.ready.pop(g)  # remove if no f buckets
                continue  
            while self.ready[g]:
                f = self.ready[g].peekitem(index=0)[0]
                if len(self.ready[g][f]) == 0:
                    self.ready[g].pop(f)  # remove if no entries in f bucket
                    continue
                if priority_only:
                    return g
                else:
                    return (f, g, self.ready[g][f])
        return float('inf')

    def get_bucket_stats(self, g):
        """ Calculate stats over f buckets for ready[g]. Removes empty f buckets as it goes 
            if invalid g or no non-empty f-buckets in g, stats['f_count'] = 0
            stats:
                f_count: total number of f buckets in this g level
                f_smallest: f value of smallest f bucket in this g level
                f_smallest_count: size (len) of smallest f bucket in this g level
                g_total count: sum of sizes of f buckets in this g level = num states in this g level
        """
        stats = {'f_count': 0, 'f_smallest': -1, 'f_smallest_count': 0, 'g_total_count': 0}
        if g in self.ready:
            del_keys = []
            f_smallest_count = float('inf')
            for f, bucket in self.ready[g].items():
                f_len = len(bucket)
                if f_len == 0:
                    del_keys.append(f)
                    continue
                stats['f_count'] += 1
                stats['g_total_count'] += f_len
                if f_len < f_smallest_count:
                    stats['f_smallest'] = f
                    stats['f_smallest_count'] = f_len
            for f in del_keys:      # pop empty f buckets
                self.ready[g].pop(f)
            if not self.ready[g]:   # empty g SortedDict(), pop it
                self.ready.pop(g)
        return stats

    def select_and_order(self, direction, lb):
        """ Select and return nodes to expand in direction this class was created for.
            lb is the instance of the lb_pairs class calling this method which enable access to all the calculated values from lb.calc_expandable():

        lb.forward_expandable_g: key:g (sorted) val dict: 
            {'f_count':0 # f buckets in this glevel, 
             'f_smallest':0 lowest cardinality f bucket in this glevel, 
             'f_smallest_count': 0 # f buckets in this glevel, 
             'g_total_count': 0 # nodes in this glevel over all f buckets ie |glevel| aka the weight of this glevel, 
             'under_glb':0 # edges under GLB, 
             'eq_glb':0 # edges at GLB, 
             'edge_count':0  # edges, 
             'connected_total_count':0 total expandable nodes in opp direction with edge to this glevel, 
             'connected_smallest_count': float('inf') lowest cardinality f bucket in opp direction with edge to this glevel, 
             'connected_smallest_count_gf':(gD, fD) g and f of opp direction lowest cardinality f bucket
             }        
        lb.backward_expandable_g: as above
        lb.expandable_edges = set( (gF, gB), .. )   # set of edges 
        lb.forward_smallest_expandable_bucket: 
          SortedSet( [(-1, 0, 0)] ) sorted set of (g, f, count) of smallest expandable buckets fwd (> 1 if smallest equal)
        lb.backward_smallest_expandable_bucket:  as prior
        lb.forward_smallest_expandable_glevel: 
          SortedSet( [(0, 0)] ) sorted set of (g, count) of smallest expandable glevels (> 1 if multiple smallest equal)
        lb.backward_smallest_expandable_glevel: as prior
        lb.forward_most_interesting_glevel: {'most_edges': -1, 'most_nodes': -1, 'mwvc_most_nodes': -1, 'mwvc_smallest_count': -1, , 'lowest': -1 } g of fwd glevel with most edges and most nodes in bwd plus g for most connected nodes and |glevel| in MWVC plus lowest expandable g
        lb.backward_most_interesting_glevel: as prior

        
        lb.tb_select - strategy for selecting node(s) to expand in selected direction(s) from Ready_d:

        'F':   select single node in first bucket i.e. a node in bucket with lowest g and lowest f 
        'FHF': select single node in highest f in lowest g
        'FHG': select single node in lowest f in highest g
        'B': entire first bucket i.e. bucket with lowest g and lowest f
        'R': random node from lowest g
        'RA': random node from all expandable nodes
        'ALL': all expandable buckets
        'GBF': expand all glevels satisfying g_D + max_g_expanded_OppositeD + EPS <= GLB
        'SG': smallest expandable glevel
        'SM': smallest expandable glevel in MWVC
        'SLG': smallest bucket in lowest g (eg use with tb_dir SBM0 that will select direction with smallest bucket in lowest g that is in a MWVC)
        'LG': lowest glevel - frequently used in conjunction with tb_dir ending in 0
        'HG': highest glevel
        'S': DVCBS: smallest glevel of the minimum expandable glevel that is in any MVC of expandable_f X expandable_b 
        'SB': smallest bucket of any expandable bucket - with tiebreak towards highest g
        'SBL': smallest bucket of any expandable bucket - with tiebreak towards lowest g
        'EC': Expand glevel with largest edge count ie. is connected with most glevels in other direction
        'LN': Vidal-like: Expand glevel with largest node count over connected glevels in other direction
        'LM': Vidal-like: Expand glevel in MWVC with largest node count over connected glevels in other direction

        lb.tb_order - determines the order for expanding selected nodes in selected direction if more than one node. 
            Since expand_nodes is a SortedKeyList with key = ordering, simply adding to this list sets the expansion order.

        'R': random
        'FIFO' / 'LIFO' - this is different from using FIFO/LIFO to "select" nodes above
        'NONE' - no explicit ordering applied

        returns self.expand_nodes:  entry format (ordering, g, f, state) sorted by ordering
        """
        self.expand_nodes = SortedKeyList(key=lambda e: e[0])  # SortedKeyList to keep entries sorted by fifo/lifo/rand/0 value
        if lb.tb_select == 'F':   # select single node in lowest f in lowest g
            if lb.tb_order == 'NONE':
                g, f, ordering, current_state = self.pop(item_only=False)
                self.expand_nodes.add( (ordering, g, f, current_state) )
            else: # select node based on ordering value
                g = self.peek_ready(priority_only=True)  # pops any empty buckets until get to a non-empty [g][f] since calc_expandable not run for this tb_select
                f = self.ready[g].peekitem(index=0)[0]
                if lb.tb_order == 'R':
                    idx = random.randint(0, len(self.ready[g][f])-1)
                else:
                    idx = self.find_lowest_ordered_idx(g, f)
                self.pop_node_or_bucket(g, f, bucket=False, idx=idx)  # puts entries into expand_nodes
        elif lb.tb_select == 'FHF':    # select single node in highest f in lowest g
            g = self.peek_ready(priority_only=True)  # pops any empty buckets until get to a non-empty [g][f]
            f = self.ready[g].peekitem(index=-1)[0]
            while len(self.ready[g][f]) == 0:        # can be empty buckets in higher f than 1st f so pop until we have a non-empty highest f
                self.ready[g].pop(f)
                f = self.ready[g].peekitem(index=-1)[0]
            if lb.tb_order == 'R':
                idx = random.randint(0, len(self.ready[g][f])-1)
            else:
                idx = self.find_lowest_ordered_idx(g, f)
            self.pop_node_or_bucket(g, f, bucket=False, idx=idx)  # puts entries into expand_nodes
        elif lb.tb_select == 'FHG':    # select single node in lowest f in highest g
            if direction == 'F':
                expandable_g = lb.forward_expandable_g
            else:
                expandable_g = lb.backward_expandable_g
            g = list(expandable_g.keys())[-1]
            f = self.ready[g].peekitem(index=0)[0]
            if lb.tb_order == 'R':
                idx = random.randint(0, len(self.ready[g][f])-1)
            else:
                idx = self.find_lowest_ordered_idx(g, f)
            self.pop_node_or_bucket(g, f, bucket=False, idx=idx)  # puts entries into expand_nodes            
        elif lb.tb_select == 'B':
            self.pop(item_only=False, bucket=True)  # puts entries into expand_nodes
        elif lb.tb_select == 'R':
            g = self.peek_ready(priority_only=True)  # pops any empty buckets until get to a non-empty [g][f] since calc_expandable not run for this tb_select
            f = random.choice(list(self.ready[g].keys()))
            if len(self.ready[g][f]) == 0:
                f = self.ready[g].peekitem(index=0)[0]  # fallback: 1st f will always be non empty 
            idx = random.randint(0, len(self.ready[g][f])-1)
            self.pop_node_or_bucket(g, f, bucket=False, idx=idx)  # puts entries into expand_nodes
        elif lb.tb_select == 'RA':
            if direction == 'F':
                expandable_g = lb.forward_expandable_g
            else:
                expandable_g = lb.backward_expandable_g
            g = random.choice(list(expandable_g.keys()))
            f = random.choice(list(self.ready[g].keys()))
            idx = random.randint(0, len(self.ready[g][f])-1)
            self.pop_node_or_bucket(g, f, bucket=False, idx=idx)  # puts entries into expand_nodes
        elif lb.tb_select == 'ALL':
            if direction == 'F':
                expandable_g = lb.forward_expandable_g
            else:
                expandable_g = lb.backward_expandable_g
            for g in expandable_g:
                self.pop_g_level(g)
        elif lb.tb_select == 'GBF': # expand all glevels satisfying g_D + max_g_expanded_OppositeD + EPS <= GLB
            if direction == 'F':
                expandable_g = lb.forward_expandable_g
                max_g_expanded_OppDir = lb.backward_max_g_expanded
            else:
                expandable_g = lb.backward_expandable_g
                max_g_expanded_OppDir = lb.forward_max_g_expanded
            for g in expandable_g:
                if g + max_g_expanded_OppDir + lb.min_edge_cost <= lb.GLB:
                    self.pop_g_level(g)
                else:
                    break
        elif lb.tb_select == 'SLG':  # smallest f bucket in lowest expandable g
            if direction == 'F':
                expandable_g = lb.forward_expandable_g
                g = lb.forward_most_interesting_glevel['lowest']   #list(expandable_g.keys())[0]
            else:
                expandable_g = lb.backward_expandable_g
                g = lb.backward_most_interesting_glevel['lowest']   #list(expandable_g.keys())[0]
            f = expandable_g[g]['f_smallest']
            self.pop_node_or_bucket(g, f, bucket=True)  # puts entries into expand_nodes
        elif lb.tb_select == 'LG':  # lowest expandable g: frequently used in conjunction with tb_dir ending in 0 and others
            #if direction == 'F':
            #    expandable_g = lb.forward_most_interesting_glevel
            #else:
            #    expandable_g = lb.backward_most_interesting_glevel
            #g = expandable_g['lowest']   #list(expandable_g.keys())[0]
            g = self.peek_ready(priority_only=True)  # pops any empty buckets until get to a non-empty [g][f] since calc_expandable not run for this tb_select
            self.pop_g_level(g)
        elif lb.tb_select == 'LGSB':  # smallest gf bucket in lowest expandable g: eg used in conjunction with tb_dir SBM0
            if direction == 'F':
                expandable_g = lb.forward_most_interesting_glevel
            else:
                expandable_g = lb.backward_most_interesting_glevel
            g = expandable_g['lowest']   #list(expandable_g.keys())[0]
            self.pop_g_level(g)
        elif lb.tb_select == 'HG':  # highest expandable g
            if direction == 'F':
                expandable_g = lb.forward_expandable_g
            else:
                expandable_g = lb.backward_expandable_g
            g = list(expandable_g.keys())[-1]  
            self.pop_g_level(g)
        elif lb.tb_select == 'SG':  # smallest expandable g
            if direction == 'F':
                expandable_g = lb.forward_smallest_expandable_glevel
            else:
                expandable_g = lb.backward_smallest_expandable_glevel
            g, gcount = expandable_g[0]
            self.pop_g_level(g)
        elif lb.tb_select == 'SM':  # smallest expandable g in MWVC
            if direction == 'F':
                expandable_g = lb.forward_most_interesting_glevel
            else:
                expandable_g = lb.backward_most_interesting_glevel
            g = expandable_g['mwvc_smallest_count']
            if g == -1:  # no MWVC found, expand smallest glevel
                if direction == 'F':
                    expandable_g = lb.forward_smallest_expandable_glevel
                else:
                    expandable_g = lb.backward_smallest_expandable_glevel
                g, gcount = expandable_g[0]
            self.pop_g_level(g)
        elif lb.tb_select == 'SB':  # smallest f bucket in any expandable g with tiebreak towards highest g
            if direction == 'F':
                expandable_f = lb.forward_smallest_expandable_bucket
            else:
                expandable_f = lb.backward_smallest_expandable_bucket
            g, f, fcount = expandable_f[-1]
            self.pop_node_or_bucket(g, f, bucket=True)  # puts entries into expand_nodes
        elif lb.tb_select == 'SBL':  # smallest f bucket in any expandable g with tiebreak towards lowest g
            if direction == 'F':
                expandable_f = lb.forward_smallest_expandable_bucket
            else:
                expandable_f = lb.backward_smallest_expandable_bucket
            g, f, fcount = expandable_f[0]
            self.pop_node_or_bucket(g, f, bucket=True)  # puts entries into expand_nodes
        elif lb.tb_select == 'EC':  # expand glevel with highest edge count
            if direction == 'F':
                expandable_g = lb.forward_most_interesting_glevel
            else:
                expandable_g = lb.backward_most_interesting_glevel
            g = expandable_g['most_edges']
            self.pop_g_level(g)  # puts entries into expand_nodes
        elif lb.tb_select == 'LN':  # expand glevel with highest connected node count
            if direction == 'F':
                expandable_g = lb.forward_most_interesting_glevel
            else:
                expandable_g = lb.backward_most_interesting_glevel
            g = expandable_g['most_nodes']
            self.pop_g_level(g)  # puts entries into expand_nodes
        elif lb.tb_select == 'LM':  # expand glevel in MWVC with highest connected node count
            if direction == 'F':
                expandable_g = lb.forward_most_interesting_glevel
            else:
                expandable_g = lb.backward_most_interesting_glevel
            g = expandable_g['mwvc_most_nodes']
            if g == -1:  # no MWVC found, expand smallest glevel
                if direction == 'F':
                    expandable_g = lb.forward_smallest_expandable_glevel
                else:
                    expandable_g = lb.backward_smallest_expandable_glevel
                g, gcount = expandable_g[0]
            self.pop_g_level(g)  # puts entries into expand_nodes
        else:
            raise ValueError(f"WaitingReadyBuckets.select_and_order(): Invalid tb_select:{lb.tb_select} tb_order:{lb.tb_order}")
        return self.expand_nodes


class LBPairs:
    """ Two WaitingReadyPriorityQueue structures, one for forward, one for backward
    Used in LB Pairs family of Bidirectional search algorithms 
    Wait priority is f and Ready priority is g, so expandable nodes are those in Ready which satisfy 
    g_forward + g_backward + epsilon <= GLB ("C" in A*/"naive BDHS") having already satisfied f_direction <= GLB to be moved from Wait to Ready
    Wait priority queue entries are tuples of (f, [g, fifo/lifo_value, state])
    Ready priority queue entries are tuples of (g, [f, fifo/lifo_value, state])

    NOTE: GLB is called min_LB in Chen 2017, LB in Shperberg 2019 and C in A* and our naive BDHS
    """
    def __init__(self, version='A', min_edge_cost=1.0, data_struct='P', 
                 tb_dir='NBS', tb_select='F', tb_order='NONE'):
        """ version is 'A' for All means move_to_read uses <= GLB, 'F' for First means move_to_ready uses < GLB
       If unknown can set min_edge_cost (eps) to 0.0
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
        self.tb_dir = tb_dir
        self.tb_select = tb_select
        self.tb_order = tb_order
        self.last_direction = 'B'

        if self.data_struct == 'B':
            self.forward = WaitingReadyBuckets(version)
            self.backward = WaitingReadyBuckets(version)
        else:  # self.data_struct == 'P'
            self.forward = WaitingReadyPriorityQueue(version)
            self.backward = WaitingReadyPriorityQueue(version)
        self.GLB = 0
        self.forward_expandable_g = {}   # key:g val: (f_count, f_smallest, |f_smallest|, g_total_count, <GLB edge count, =GLB edge count, edge count, connected_total_count, connected_smallest_count, connected_smallest_count_gf (gD, fD))
        self.backward_expandable_g = {}  # key:g val: as prior
        self.expandable_edges = set()   # set of (gF, gB)
        #self.expandable_edges_reversed = set()
        self.forward_smallest_expandable_bucket = SortedSet( [(-1, 0, 0)] ) #  (f, g, count) 
        self.backward_smallest_expandable_bucket = SortedSet( [(-1, 0, 0)] ) #  (f, g, count) 
        self.forward_smallest_expandable_glevel = SortedSet( [(0, 0)] )  # (g, count)
        self.backward_smallest_expandable_glevel = SortedSet( [(0, 0)] )  # (g, count)
        self.forward_most_interesting_glevel = {'most_edges': -1, 'most_nodes': -1, 'mwvc_most_nodes': -1, 'mwvc_smallest_count': -1, 'lowest': -1 }   # fwd g of glevel with most edges to bwd and edges to most nodes in bwd and corresponding for subset in MWVC
        self.backward_most_interesting_glevel = {'most_edges': -1, 'most_nodes': -1, 'mwvc_most_nodes': -1, 'mwvc_smallest_count': -1, 'lowest': -1 }  # as prior
        self.forward_g_mwvc, self.backward_g_mwvc = [], []  # List of tuples of glevels that are in any MWVC

        self.forward_max_g_expanded = 0     # max g expanded in forward direction
        self.backward_max_g_expanded = 0    # max g expanded in backward direction
        self.forward_gmin = 0               # current gmin in forward direction
        self.backward_gmin = 0              # current gmin in backward direction
        self.forward_fmin = 0               # current fmin in forward direction
        self.backward_fmin = 0              # current fmin in backward direction
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
            lb(u,v) = max(fmin_f, fmin_b, gmin_f + gmin_b + min_edge_cost)
            GLB is min(lb(u,v)). 
            Returns found=True if there are expandable nodes in each ready queue along with the next GLB value
        """
        self.forward_fmin = self.forward.peek_wait(priority_only=True)
        if self.forward_fmin == float('inf'):
            self.forward_fmin = 0
        self.backward_fmin = self.backward.peek_wait(priority_only=True)
        if self.backward_fmin == float('inf'):
            self.backward_fmin = 0
        #CLB = 0  # CLB starts at 0 each time - always finds optimal soln but causes more examples of non-monotonic GLB
        CLB = GLB  # CLB starts at old GLB each time. NBS and DVCBS style - still get non-monotonic GLB, just less often than starting at 0 each time
        found = False
        #count_f, count_b = self.move_to_ready(CLB)  # NBS_f style - still get non-monotonic GLB if put move_to_ready() here but the "A" versions gets into infinite loop 
        while True:
            count_f, count_b = self.move_to_ready(CLB)     # DVCBS style - both "f" and "a" versions always find optimal soln and both sometimes get non-monotonic GLB
            if self.forward.isEmpty() and self.backward.isEmpty():
                break
            if self.forward.ready and self.backward.ready:
                self.forward_gmin = self.forward.peek_ready(priority_only=True)
                self.backward_gmin = self.backward.peek_ready(priority_only=True)
                if self.forward_gmin + self.backward_gmin + self.min_edge_cost <= CLB: # This is the condition for expandable nodes
                    found = True
                    break
            if self.version == 'F':
                count_f, count_b = self.move_one_to_ready(CLB)
            else:
                count_f, count_b = 0, 0
            #if count_f == 0 or count_b == 0:   # Per Chen pseudocode - clb non-monotonic but still optimal
            if count_f == 0 and count_b == 0:   # Per Shperberg/Siag code for CLB monotonic increase!
                CLB = self.get_new_LB()
                self.GLB = CLB
        return found, CLB

    def calc_expandable(self, add_mwvc=True):
        """ Calculate which buckets in ReadyF, ReadyB are expandable without popping anything. 
            Returns: 
            - Statistics for glevels and gf buckets in Forward.Ready and Backward.Ready: (f_count, f_smallest, |f_smallest|, g_total_count, <GLB edge count, =GLB edge count, edge count, connected_total_count, connected_smallest_count, connected_smallest_count_gf (gD, fD)) 
            - SortedSets of the smallest expandable g-f bucket(s) and smallest expandable glevel(s) (sets since > 1 can be equally small. Leave to tb_select to choose which one to expand)
            - Most connected glevel(s) in Forward and Backward directions ie glvel with most edges and (better) most nodes connected to in other direction
            - Set of expandable edges {(gF, gB), ...} where gF is from Forward and gB is from Backward 
            Additionally, if add_mwvc:
            - Sets of glevels that are in a Weighted Minimum Vertex Cover (WMVC) of the expandable edges in Forward and Backward directions
            - Most connected glevel in a MWVC in Forward and Backward directions
            - Smallest |glevel| in a MWVC in Forward and Backward directions
            Note: Once this is run, no empty expandable buckets or g-levels will be present so downstream select_and_order() code can omit empty checks
        """
        self.forward_expandable_g = {}   # key:g (sorted) val: (f_count, f_smallest, |f_smallest|, g_total_count, <GLB edge count, =GLB edge count, edge count, connected_total_count, connected_smallest_count, connected_smallest_count_gf (gD, fD)) 
        self.backward_expandable_g = {}  # key:g (sorted) val: as above
        self.expandable_edges = set()   # set of (gF, gB)
        self.forward_smallest_expandable_bucket = SortedSet( [(-1, 0, 0)] )     # sorted set of (g, f, count) of smallest expandable buckets fwd (> 1 if smallest equal)
        self.backward_smallest_expandable_bucket = SortedSet( [(-1, 0, 0)] )    # as prior
        self.forward_smallest_expandable_glevel = SortedSet( [(0, 0)] )     # sorted set of (g, count) of smallest expandable glevels
        self.backward_smallest_expandable_glevel = SortedSet( [(0, 0)] )    # as prior
        self.forward_most_interesting_glevel = {'most_edges': -1, 'most_nodes': -1, 'mwvc_most_nodes': -1, 'mwvc_smallest_count': -1, 'lowest': -1 }   # fwd g of glevel with most edges to bwd and edges to most nodes in bwd and corresponding for subset in MWVC
        self.backward_most_interesting_glevel = {'most_edges': -1, 'most_nodes': -1, 'mwvc_most_nodes': -1, 'mwvc_smallest_count': -1, 'lowest': -1}  # as prior
        self.forward_g_mwvc, self.backward_g_mwvc = [], []  # List of tuples of glevels that are in any MWVC
 
        forward_smallest_count = float('inf')
        backward_smallest_count = float('inf')
        forward_smallest_glevel_count = float('inf')
        backward_smallest_glevel_count = float('inf')
        forward_most_edges = 0
        backward_most_edges = 0
        forward_most_nodes = 0
        backward_most_nodes = 0

        forward_g_list = list(self.forward.ready.keys())  # iterate over copy of keys since get_bucket_stats can delete empty g or f buckets
        backward_g_list = list(self.backward.ready.keys())
        found_lowest_gB = False
        lowest_gB = 0

        for gF in forward_g_list:       # loop through forward checking for edges between buckets in forward and backward
            if gF + lowest_gB + self.min_edge_cost > self.GLB:
                break  # if gF + smallest_gB + eps > GLB then no gF + gB + eps will work since gF monotonically increases, so can terminate 

            stats_forward = self.forward.get_bucket_stats(gF)  # {'f_count': ,'f_smallest': , 'f_smallest_count': , 'g_total_count': } 
            if stats_forward['f_count'] == 0:  # gF key was empty and now popped
                continue
            # {'f_count':0 ,'f_smallest':0 , 'f_smallest_count': 0, 'g_total_count': 0, 'under_glb':0, 'eq_glb':0, 'edge_count':0, 'connected_total_count':0, 'connected_smallest_count': float('inf'), 'connected_smallest_count_gf':()}
            stats_forward.update( {'under_glb':0, 'eq_glb':0, 'edge_count':0, 'connected_total_count':0, 'connected_smallest_count': float('inf'), 'connected_smallest_count_gf':()} )

            for gB in backward_g_list:
                if gB not in self.backward.ready:
                    continue
                if gF + gB + self.min_edge_cost <= self.GLB:
                    if gB not in self.backward_expandable_g:
                        stats_backward = self.backward.get_bucket_stats(gB)  # {'f_count': ,'f_smallest': , 'f_smallest_count':, 'g_total_count': } 
                        if stats_backward['f_count'] == 0:  # gB key was empty and now popped
                            continue
                        stats_backward.update( {'under_glb':0, 'eq_glb':0, 'edge_count':0, 'connected_total_count':0, 'connected_smallest_count': float('inf'), 'connected_smallest_count_gf':()} )
                    if not found_lowest_gB:
                        lowest_gB = gB
                        found_lowest_gB = True

                    # if got here, both forward and backward buckets are non-empty
                    if gF not in self.forward_expandable_g:
                        self.forward_expandable_g[gF] = stats_forward
                    if gB not in self.backward_expandable_g:
                        self.backward_expandable_g[gB] = stats_backward
                    if gF + gB + self.min_edge_cost < self.GLB:
                        self.forward_expandable_g[gF]["under_glb"] += 1
                        self.backward_expandable_g[gB]["under_glb"] += 1
                    else:
                        self.forward_expandable_g[gF]["eq_glb"] += 1
                        self.backward_expandable_g[gB]["eq_glb"] += 1
                    self.expandable_edges.add( (gF, gB) )
                    #self.expandable_edges_reversed.add( (gB, gF) )
                    self.forward_expandable_g[gF]["edge_count"] += 1
                    self.backward_expandable_g[gB]["edge_count"] += 1
                    self.forward_expandable_g[gF]["connected_total_count"] += self.backward_expandable_g[gB]["g_total_count"]  # total nodes in other direction connected to this glevel
                    self.backward_expandable_g[gB]["connected_total_count"] += self.forward_expandable_g[gF]["g_total_count"]

                    if self.forward_most_interesting_glevel["lowest"] == -1:
                        self.forward_most_interesting_glevel["lowest"] = gF
                    if self.backward_most_interesting_glevel["lowest"] == -1:
                        self.backward_most_interesting_glevel["lowest"] = gB

                    if forward_most_edges < self.forward_expandable_g[gF]["edge_count"]:
                         forward_most_edges = self.forward_expandable_g[gF]["edge_count"]
                         self.forward_most_interesting_glevel["most_edges"] = gF
                    if backward_most_edges < self.backward_expandable_g[gB]["edge_count"]:
                         backward_most_edges = self.backward_expandable_g[gB]["edge_count"]
                         self.backward_most_interesting_glevel["most_edges"] = gB

                    if forward_most_nodes < self.forward_expandable_g[gF]["connected_total_count"]:
                         forward_most_nodes = self.forward_expandable_g[gF]["connected_total_count"]
                         self.forward_most_interesting_glevel["most_nodes"] = gF
                    if backward_most_nodes < self.backward_expandable_g[gB]["connected_total_count"]:
                         backward_most_nodes = self.backward_expandable_g[gB]["connected_total_count"]
                         self.backward_most_interesting_glevel["most_nodes"] = gB

                    if self.forward_expandable_g[gF]["connected_smallest_count"] > self.backward_expandable_g[gB]["f_smallest_count"]: # smallest bucket in other direction connected to this glevel
                        self.forward_expandable_g[gF]["connected_smallest_count"] = self.backward_expandable_g[gB]["f_smallest_count"]
                        self.forward_expandable_g[gF]["connected_smallest_count_gf"] = (gB, self.backward_expandable_g[gB]["f_smallest"])

                    if self.backward_expandable_g[gB]["connected_smallest_count"] > self.forward_expandable_g[gF]["f_smallest_count"]: # smallest bucket in other direction connected to this glevel
                        self.backward_expandable_g[gB]["connected_smallest_count"] = self.forward_expandable_g[gF]["f_smallest_count"]
                        self.backward_expandable_g[gB]["connected_smallest_count_gf"] = (gF, self.forward_expandable_g[gF]["f_smallest"])

                    if self.forward_expandable_g[gF]["f_smallest_count"] < forward_smallest_count:  # Calc smallest overall expandable bucket fwd
                        forward_smallest_count = self.forward_expandable_g[gF]["f_smallest_count"]
                        self.forward_smallest_expandable_bucket = SortedSet( [(gF, self.forward_expandable_g[gF]["f_smallest"], forward_smallest_count)] ) # [g,f, count]
                    elif self.forward_expandable_g[gF]["f_smallest_count"] == forward_smallest_count:
                        self.forward_smallest_expandable_bucket.add( (gF, self.forward_expandable_g[gF]["f_smallest"], forward_smallest_count) ) # [g,f, count]

                    if self.backward_expandable_g[gB]["f_smallest_count"] < backward_smallest_count:  # Calc smallest overall expandable bucket bwd
                        backward_smallest_count = self.backward_expandable_g[gB]["f_smallest_count"]
                        self.backward_smallest_expandable_bucket = SortedSet( [(gB, self.backward_expandable_g[gB]["f_smallest"], backward_smallest_count)] ) # [g,f, count]
                    elif self.backward_expandable_g[gB]["f_smallest_count"] == backward_smallest_count:  
                        self.backward_smallest_expandable_bucket.add( (gB, self.backward_expandable_g[gB]["f_smallest"], backward_smallest_count) ) # [g,f, count]

                    if self.forward_expandable_g[gF]["g_total_count"] < forward_smallest_glevel_count:  # Calc smallest overall expandable glevel fwd
                        forward_smallest_glevel_count = self.forward_expandable_g[gF]["g_total_count"]
                        self.forward_smallest_expandable_glevel = SortedSet( [(gF, forward_smallest_glevel_count)] ) # [g, count]
                    elif self.forward_expandable_g[gF]["g_total_count"] == forward_smallest_glevel_count:  
                        self.forward_smallest_expandable_glevel.add( (gF, forward_smallest_glevel_count) ) # [g, count]

                    if self.backward_expandable_g[gB]["g_total_count"] < backward_smallest_glevel_count:  # Calc smallest overall expandable glevel bwd
                        backward_smallest_glevel_count = self.backward_expandable_g[gB]["g_total_count"]
                        self.backward_smallest_expandable_glevel = SortedSet( [(gB, backward_smallest_glevel_count)] ) # [g, count]
                    elif self.backward_expandable_g[gB]["g_total_count"] == backward_smallest_glevel_count:  
                        self.backward_smallest_expandable_glevel.add( (gB, backward_smallest_glevel_count) ) # [g, count]
                else:        # gF + gB + self.min_edge_cost <= self.GLB
                    break   #continue # stop inner loop when gF + gB + eps > GLB since gB increases monotonically

        if add_mwvc: # Calc Minimum Weighted Vertex Cover (MWVC) of the expandable edges with algo based on Shaham et al 2017, 2018 and Shperberg et al 2019
            min_val, self.forward_g_mwvc, self.backward_g_mwvc = util.find_minimum_weighted_vertex_cover(
                                                                                                self.forward_expandable_g, 
                                                                                                self.backward_expandable_g, 
                                                                                                self.min_edge_cost, 
                                                                                                self.GLB )
            total_connected_nodes = -1
            smallest_glevel_count = float('inf')
            for g in self.forward_g_mwvc:
                if self.forward_expandable_g[g]["g_total_count"] < smallest_glevel_count:
                    smallest_glevel_count = self.forward_expandable_g[g]["g_total_count"]
                    self.forward_most_interesting_glevel["mwvc_smallest_count"] = g
                if self.forward_expandable_g[g]["connected_total_count"] > total_connected_nodes:
                    total_connected_nodes = self.forward_expandable_g[g]["connected_total_count"]
                    self.forward_most_interesting_glevel["mwvc_most_nodes"] = g
            total_connected_nodes = -1
            smallest_glevel_count = float('inf')
            for g in self.backward_g_mwvc:
                if self.backward_expandable_g[g]["g_total_count"] < smallest_glevel_count:
                    smallest_glevel_count = self.backward_expandable_g[g]["g_total_count"]
                    self.backward_most_interesting_glevel["mwvc_smallest_count"] = g
                if self.backward_expandable_g[g]["connected_total_count"] > total_connected_nodes:
                    total_connected_nodes = self.backward_expandable_g[g]["connected_total_count"]
                    self.backward_most_interesting_glevel["mwvc_most_nodes"] = g

        return

    def select_and_order(self, direction):
        """
         select nodes for expansion in a direction
        """
        if direction == 'F':
            expand_nodes = self.forward.select_and_order(direction, self)
        else:
            expand_nodes = self.backward.select_and_order(direction, self)
        return expand_nodes


    def implicit_tb_dir(self):
        """ Implicit tiebreaker on direction if Explicit tb results in a tie
            Using Alter since its fast and always chooses a unique direction
        """
        if self.last_direction == 'F':
            fwd, bwd = False, True
            self.last_direction = 'B'
        else:
            fwd, bwd = True, False
            self.last_direction = 'F'
        return fwd, bwd

    def calc_direction(self):
        """ Return direction(s) to expand in based on self.tb_dir
        'NBS': Always expand in both directions, 
        'F'/'B': Forward only, backward only
        'A': expand in alternating direction to past time, 
        'P': Pohl: expand direction based on smallest cardinality of open lists, 
        'R': expand in a random direction, 
        'G': expand direction based on lowest expandable g in open lists, 
        'S': expand direction based on which READY_d has smallest expandable |glevel| of any glevel, 
        'S0': smallest |glevel| of the lowest glevel in Fwd and Bwd
        'SM': smallest |glevel| of any glevel in MWVC in Fwd and Bwd
        'SM0': DVCBS: smallest |glevel| of lowest glevel in MWVC in Fwd and Bwd,
        'SB': expand direction based on which READY_d has smallest expandable |g-f bucket| 
        'SBM0': expand direction based on which READY_d has smallest expandable |g-f bucket| in lowest glevel in MWVC
        'EC': Expand direction based on which READY_d has glevel with largest edge count ie is connected with most glevels in other direction
        'LN': Vidal-like: Expand direction based on which READY_d has glevel connected with largest node count in other direction 
        'LN0': Vidal-like: Expand direction based on which READY_d has lowest glevel connected with largest node count in other direction
        'LM': Expand direction based on which READY_d has glevel in MWVC connected with largest node count in other direction 
        'LM0': Expand direction based on which READY_d has lowest glevel in MWVC connected with largest node count in other direction
        'GBF': Expand direction based on which READY_d lowest g + maxoppd_max_g_expanded + self.min_edge_cost <= GLB

        Note: For some tb_dir, calc_expandable() must have been run before calling calc_direction(). This is handled in the search loop

        """
        fwd = False
        bwd = False
        if self.tb_dir == 'NBS':    # always expand in both directions
            fwd, bwd = True, True
            self.last_direction = 'FB' 
        elif self.tb_dir == 'F':    # forward only
            fwd, bwd = True, False
            self.last_direction = 'F' 
        elif self.tb_dir == 'B':    # backward only
            fwd, bwd = False, True
            self.last_direction = 'B'
        elif self.tb_dir == 'A':    # alternating direction to prior step
            if self.last_direction == 'F':
                fwd, bwd = False, True
                self.last_direction = 'B'
            else:
                fwd, bwd = True, False
                self.last_direction = 'F'
        elif self.tb_dir == 'P':    # Pohl: expand direction based on smallest cardinality of open lists
            fval = self.forward.curr_size()
            bval = self.backward.curr_size()
            if fval > bval:
                fwd, bwd = False, True
                self.last_direction = 'B'
            elif fval < bval:
                fwd, bwd = True, False
                self.last_direction = 'F'
            else:
                fwd, bwd = self.implicit_tb_dir()
        elif self.tb_dir == 'R':            # random direction
            if random.choice(['F','B']) == 'F':
                fwd, bwd = True, False
                self.last_direction = 'F'
            else:
                fwd, bwd = False, True
                self.last_direction = 'B'
        elif self.tb_dir == 'G':            # expand direction based on lowest expandable g in fwd vs bwd
            fval = self.forward.peek_ready(priority_only=True)
            bval = self.backward.peek_ready(priority_only=True)
            if fval > bval:
                fwd, bwd = False, True
                self.last_direction = 'B'
            elif fval < bval:
                fwd, bwd = True, False
                self.last_direction = 'F'
            else:
                fwd, bwd = self.implicit_tb_dir()
        elif self.tb_dir == 'S':    # smallest |glevel| of any glevel 
            fval = self.forward_smallest_expandable_glevel[0][-1]  # sorted set of (g, count) of smallest expandable glevels - all counts the same
            bval = self.backward_smallest_expandable_glevel[0][-1]
            if fval > bval:
                fwd, bwd = False, True
                self.last_direction = 'B'
            elif fval < bval:
                fwd, bwd = True, False
                self.last_direction = 'F'
            else:
                fwd, bwd = self.implicit_tb_dir()
        elif self.tb_dir == 'S0':   # smallest |glevel| of the lowest glevel  
            fval = self.forward_expandable_g[ self.forward_most_interesting_glevel['lowest'] ]['g_total_count']
            bval = self.backward_expandable_g[ self.backward_most_interesting_glevel['lowest'] ]['g_total_count']
            if fval > bval:
                fwd, bwd = False, True
                self.last_direction = 'B'
            elif fval < bval:
                fwd, bwd = True, False
                self.last_direction = 'F'
            else:
                fwd, bwd = self.implicit_tb_dir()
        elif self.tb_dir == 'SM':   # smallest |glevel| of any glevel in MWVC 
            if self.forward_g_mwvc:
                fval = self.forward_expandable_g[ self.forward_most_interesting_glevel['mwvc_smallest_count'] ]['g_total_count']
            else:
                fval = float('inf')  # no MWVC covering forward direction
            if self.backward_g_mwvc:
                bval = self.backward_expandable_g[ self.backward_most_interesting_glevel['mwvc_smallest_count'] ]['g_total_count']
            else:
                bval = float('inf')  # no MWVC covering backward direction
            if fval > bval:
                fwd, bwd = False, True
                self.last_direction = 'B'
            elif fval < bval:
                fwd, bwd = True, False
                self.last_direction = 'F'
            else:
                fwd, bwd = self.implicit_tb_dir()
        elif self.tb_dir == 'SM0':   # smallest |glevel| of lowest glevel in MWVC 
            if self.forward_g_mwvc:
                fval = self.forward_expandable_g[ self.forward_most_interesting_glevel['lowest'] ]['g_total_count']
            else:
                fval = float('inf')  # no MWVC covering forward direction
            if self.backward_g_mwvc:
                bval = self.backward_expandable_g[ self.backward_most_interesting_glevel['lowest'] ]['g_total_count']
            else:
                bval = float('inf')  # no MWVC covering backward direction
            if fval > bval:
                fwd, bwd = False, True
                self.last_direction = 'B'
            elif fval < bval:
                fwd, bwd = True, False
                self.last_direction = 'F'
            else:
                fwd, bwd = self.implicit_tb_dir()
        elif self.tb_dir == 'SB':   # smallest |gf bucket| of any expandable glevel in MWVC
            fval = self.forward_smallest_expandable_bucket[0][-1]  # all counts in set will be the same
            bval = self.backward_smallest_expandable_bucket[0][-1]
            if fval > bval:
                fwd, bwd = False, True
                self.last_direction = 'B'
            elif fval < bval:
                fwd, bwd = True, False
                self.last_direction = 'F'
            else:
                fwd, bwd = self.implicit_tb_dir()
        elif self.tb_dir == 'SBM0':   # smallest |gf bucket| of lowest glevel in MWVC
            if self.forward_g_mwvc:
                fval = self.forward_expandable_g[ self.forward_most_interesting_glevel['lowest'] ]['f_smallest_count']
            else:
                fval = float('inf')  # no MWVC covering forward direction
            if self.backward_g_mwvc:
                bval = self.backward_expandable_g[ self.backward_most_interesting_glevel['lowest'] ]['f_smallest_count']
            else:
                bval = float('inf')  # no MWVC covering backward direction
            if fval > bval:
                fwd, bwd = False, True
                self.last_direction = 'B'
            elif fval < bval:
                fwd, bwd = True, False
                self.last_direction = 'F'
            else:
                fwd, bwd = self.implicit_tb_dir()

        elif self.tb_dir == 'EC':       # favour side with expandable glevel that has largest # edges
            fval = self.forward_expandable_g[ self.forward_most_interesting_glevel['most_edges'] ]['edge_count']
            bval = self.backward_expandable_g[ self.backward_most_interesting_glevel['most_edges'] ]['edge_count']
            if fval > bval:  # reversed from other tb_dir since we want to expand the side with the largest edge count
                fwd, bwd = True, False
                self.last_direction = 'F'
            elif fval < bval:
                fwd, bwd = False, True
                self.last_direction = 'B'
            else:
                fwd, bwd = self.implicit_tb_dir()
        elif self.tb_dir == 'LN':  # favor most connected in any expandable glevel
            fval = self.forward_expandable_g[ self.forward_most_interesting_glevel['most_nodes'] ]['connected_total_count']
            bval = self.backward_expandable_g[ self.backward_most_interesting_glevel['most_nodes'] ]['connected_total_count']
            if fval > bval:  # reversed from other tb_dir since we want to expand the side with the largest # of connected nodes
                fwd, bwd = True, False
                self.last_direction = 'F'
            elif fval < bval:
                fwd, bwd = False, True
                self.last_direction = 'B'
            else:
                fwd, bwd = self.implicit_tb_dir()
        elif self.tb_dir == 'LN0':   # favor most connected in lowest expandable glevel
            fval = self.forward_expandable_g[ self.forward_most_interesting_glevel['lowest'] ]['connected_total_count']
            bval = self.backward_expandable_g[ self.backward_most_interesting_glevel['lowest'] ]['connected_total_count']
            if fval > bval:  # reversed from other tb_dir since we want to expand the side with the largest # of connected nodes
                fwd, bwd = True, False
                self.last_direction = 'F'
            elif fval < bval:
                fwd, bwd = False, True
                self.last_direction = 'B'
            else:
                fwd, bwd = self.implicit_tb_dir()
        elif self.tb_dir == 'LM':   # favor most connected in any expandable glevel in MWVC
            if self.forward_g_mwvc:
                fval = self.forward_expandable_g[ self.forward_most_interesting_glevel['mwvc_most_nodes'] ]['connected_total_count']
            else:
                fval = -1  # no MWVC covering forward direction    
            if self.backward_g_mwvc:
                bval = self.backward_expandable_g[ self.backward_most_interesting_glevel['mwvc_most_nodes'] ]['connected_total_count']
            else:
                bval = -1  # no MWVC covering backward direction
            if fval > bval:  # reversed from other tb_dir since we want to expand the side with the largest # of connected nodes
                fwd, bwd = True, False
                self.last_direction = 'F'
            elif fval < bval:
                fwd, bwd = False, True
                self.last_direction = 'B'
            else:
                fwd, bwd = self.implicit_tb_dir()
        elif self.tb_dir == 'LM0':   # favor most connected in lowest expandable glevel in MWVC
            if self.forward_g_mwvc:
                fval = self.forward_expandable_g[ self.forward_most_interesting_glevel['lowest'] ]['connected_total_count']
            else:
                fval = -1  # no MWVC covering forward direction
            if self.backward_g_mwvc:
                bval = self.backward_expandable_g[ self.backward_most_interesting_glevel['lowest'] ]['connected_total_count']
            else:
                bval = -1  # no MWVC covering backward direction
            if fval > bval:  # reversed from other tb_dir since we want to expand the side with the largest # of connected nodes
                fwd, bwd = True, False
                self.last_direction = 'F'
            elif fval < bval:
                fwd, bwd = False, True
                self.last_direction = 'B'
            else:
                fwd, bwd = self.implicit_tb_dir()
        elif self.tb_dir == 'GBF':  # Expand direction based on which READY_d lowest g + self.maxoppd_max_g_expanded + self.min_edge_cost <= GLB
            fwd_lowg = self.forward.peek_ready(priority_only=True)
            bwd_lowg = self.backward.peek_ready(priority_only=True)
            if fwd_lowg + self.backward_max_g_expanded + self.min_edge_cost <= self.GLB:
                fwd = True
            else:
                fwd = False
            if bwd_lowg + self.forward_max_g_expanded + self.min_edge_cost <= self.GLB:
                bwd = True
            else:
                bwd = False
            if fwd and bwd:
                fwd, bwd = self.implicit_tb_dir()
            
        return fwd, bwd

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






"""

print(f"##### FORWARD #####")
print(f"WAIT max:{frontier.forward.wait_max_size} READY max:{frontier.forward.ready_max_size}")
if frontier.data_struct != 'P': print(f"WAIT f keys:{list(frontier.forward.wait.keys())}")
print(f"WAIT:{frontier.forward.wait}")
if frontier.data_struct != 'P': print(f"READY g keys:{list(frontier.forward.ready.keys())}")
print(f"READY:{frontier.forward.ready}") 
print(f"WAIT+READY CURR SIZE: {frontier.forward.curr_size()}")
print(f"##### BACKWARD #####")
print(f"WAIT max:{frontier.backward.wait_max_size} READY max:{frontier.backward.ready_max_size}")
if frontier.data_struct != 'P': print(f"WAIT f keys:{list(frontier.backward.wait.keys())}")
print(f"WAIT:{frontier.backward.wait}")
if frontier.data_struct != 'P': print(f"READY g keys:{list(frontier.backward.ready.keys())}")
print(f"READY:{frontier.backward.ready}") 
print(f"WAIT+READY CURR SIZE: {frontier.backward.curr_size()}")
print("######## CALC EXPANDABLE #######")
print(f"Fwd EXPANDABLE:{frontier.forward_expandable_g}")   # key:g (sorted) val: (f, |f|, <GLB count, =GLB count, edge count) <- track <GLB, =GLB for DVCBS which uses <GLB
print(f"Bwd EXPANDABLE:{frontier.backward_expandable_g}")  # key:g (sorted) val: (f, |f|, <GLB count, =GLB count, edge count) 
print(f"Fwd weights: {[(g, frontier.forward_expandable_g[g]['g_total_count']) for g in frontier.forward_expandable_g ]}")
print(f"Bwd weights: {[(g, frontier.backward_expandable_g[g]['g_total_count']) for g in frontier.backward_expandable_g ]}")
print(f"Edges:{frontier.expandable_edges}")   # set of (gF, gB)
#print(f"Edges Reversed:{frontier.expandable_edges_reversed}")   # set of (gB, gF)
print(f"Fwd Smallest exp bucket:{frontier.forward_smallest_expandable_bucket}")  # [f, g, count] of smallest expandable bucket fwd
print(f"Bwd Smallest exp bucket:{frontier.backward_smallest_expandable_bucket}") # [f, g, count] of smallest expandable bucket bwd
print(f"Fwd Smallest exp glevel:{frontier.forward_smallest_expandable_glevel}")  # [g, count] of smallest expandable glevel fwd
print(f"Bwd Smallest exp glevel:{frontier.backward_smallest_expandable_glevel}") # [g, count] of smallest expandable glevel bwd
print(f"Fwd most connected g: {frontier.forward_most_interesting_glevel}")         # fwd g of glevel with most edges to bwd and edges to most nodes in bwd
print(f"Bwd most connected g: {frontier.backward_most_interesting_glevel}")
print(f"Fwd MWVC: {frontier.forward_g_mwvc}")  # g values in MWVC covering forward direction
print(f"Bwd MWVC: {frontier.backward_g_mwvc}") # g values in MWVC covering backward direction




frontier = LBPairs(version='A', min_edge_cost=1.0, data_struct='B', 
                 tb_dir='SM0', tb_select='LG', tb_order='NONE')
frontier.calc_expandable(add_mwvc=True)
print(frontier.calc_direction())  # (fwd, bwd) (True, False)
print(frontier.last_direction)    #  F

frontier.push('F', [0, 0, 'f0'], 100, float('inf'), float('inf'))
frontier.push('B', [0, 0, 'b0'], 100, float('inf'), float('inf'))
frontier.push('F', [10, 0, 'f1'], 96, float('inf'), float('inf'))
frontier.push('B', [1, 0, 'b1'], 99, float('inf'), float('inf'))
frontier.push('B', [12, 0, 'b1'], 96, prior_f=99, prior_g=1)
frontier.push('B', [23, 0, 'b2'], 96, float('inf'), float('inf'))
frontier.push('F', [10, 0, 'f3'], 96, float('inf'), float('inf'))
frontier.push('B', [10, 0, 'b3'], 96, float('inf'), float('inf'))
frontier.push('F', [11, 0, 'f4'], 96, float('inf'), float('inf'))
frontier.push('B', [1, 0, 'b4'], 99, float('inf'), float('inf'))
frontier.push('B', [11, 0, 'b4'], 96, prior_f=99, prior_g=1)
frontier.push('B', [13, 0, 'b5'], 96, float('inf'), float('inf'))
frontier.push('F', [22, 0, 'f5'], 96, float('inf'), float('inf'))
frontier.push('B', [14, 0, 'b5'], 96, float('inf'), float('inf'))
frontier.push('F', [15, 0, 'f6'], 96, float('inf'), float('inf'))
frontier.push('B', [31, 0, 'b6'], 96, float('inf'), float('inf'))
frontier.push('B', [32, 0, 'b7'], 96, float('inf'), float('inf'))
frontier.push('F', [16, 0, 'f7'], 96, float('inf'), float('inf'))
frontier.push('F', [16, 0, 'f8'], 96, float('inf'), float('inf'))
frontier.push('B', [12, 0, 'b7'], 96, float('inf'), float('inf'))
frontier.push('F', [16, 0, 'f9'], 96, float('inf'), float('inf'))
frontier.push('B', [12, 0, 'b8'], 96, float('inf'), float('inf'))
frontier.push('B', [12, 0, 'b9'], 96, float('inf'), float('inf'))

frontier.prepare_expandable(0) # (True, 96)

frontier.GLB=1
frontier.calc_expandable(True)
print(frontier.calc_direction())  # (fwd, bwd) (True, False)
print(frontier.last_direction)    #  F

frontier.GLB=21
frontier.calc_expandable(True)
print(frontier.calc_direction())  # (fwd, bwd) (False, True)
print(frontier.last_direction)    #  B

frontier.GLB=22
frontier.calc_expandable(True)
print(frontier.calc_direction())  # (fwd, bwd) (True, False)
print(frontier.last_direction)    #  F

frontier.GLB=23
frontier.calc_expandable(True)


frontier.GLB=42
frontier.calc_expandable(True)
print(frontier.calc_direction())  # (fwd, bwd) (True, False)
print(frontier.last_direction)    #  F

frontier.GLB=24
frontier.calc_expandable(True)

frontier.GLB=26
frontier.calc_expandable(True)


frontier.forward.ready[10][96] = SortedKeyList()
frontier.backward.ready[11][96] = SortedKeyList()
frontier.backward.ready[13] = SortedDict()
frontier.forward.ready[16] = SortedDict()

frontier.GLB=42
print(frontier.calc_direction())  # (fwd, bwd) (True, False)
print(frontier.last_direction)    #  F


frontier.GLB=21
frontier.tb_dir = 'P'
print(frontier.calc_direction())  # (fwd, bwd) (True, False)
print(frontier.last_direction)    #  F


frontier.GLB=21
frontier.tb_dir = 'G'
print(frontier.calc_direction())  # (fwd, bwd) (True, False)
print(frontier.last_direction)    #  F




frontier.prepare_expandable(0) # (True, 3)
print(frontier.calc_direction())  # (fwd, bwd) (True, False)
print(frontier.last_direction)    # F


frontier.prepare_expandable(0) # (True, 3)
print(frontier.calc_direction())  # (fwd, bwd) (True, False)
print(frontier.last_direction)    # F

frontier.push('F', [2, 0, 'f8'], 1, float('inf'), float('inf'))
frontier.push('B', [2, 0, 'b10'], 1, float('inf'), float('inf'))
frontier.push('F', [1, 0, 'f9'], 1, float('inf'), float('inf'))
frontier.push('B', [1, 0, 'b11'], 99, float('inf'), float('inf'))
frontier.push('B', [1, 0, 'b11'], 1, prior_f=99, prior_g=1)
frontier.push('B', [1, 0, 'b13'], 1, float('inf'), float('inf'))

################
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