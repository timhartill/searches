"""
Data Structures
"""
import heapq
import random


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
        if not self.isEmpty():
            if self.increment_tb2:
                priority, tiebreaker1, tiebreaker2, item = heapq.heappop(self.heap)
            else:
                priority, tiebreaker1, item = heapq.heappop(self.heap)
                tiebreaker2 = None
            if item_only:
                return item
            return item, priority, tiebreaker1, tiebreaker2
        return None

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
            raise ValueError(f"Invalid tiebreaker1: {self.tiebreaker2}")



