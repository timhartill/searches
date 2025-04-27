"""
Misc Utility fns
"""

import os
import csv
import heapq


def make_prob_serial(prob, prefix="__", suffix=""):
    """ Make csv-friendly key for a problem description eg initial state """
    prob_str = str(prob)
    prob_str = prob_str.replace(" ", "_").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(",", "").replace("'", "").replace("'", "").replace('"', "")
    return prefix + prob_str + suffix

def make_prob_str(file_name='', initial_state=None, goal_state=None, prefix="__", suffix="__"):
    """ Make a string for the problem description eg initial state """
    if file_name:
        file_name = prefix + os.path.basename(file_name)
    prob_str = file_name
    if initial_state is not None:
        prob_str += make_prob_serial(initial_state, prefix=prefix, suffix="")
    if goal_state is not None:
        prob_str += make_prob_serial(goal_state, prefix=prefix, suffix="")
    return prob_str + suffix


def write_jsonl_to_csv(all_results, csv_file_path, del_keys=['path'], 
                       delimiter=',', lineterminator='\n', verbose=True):
    """ Write a list of dictionaries to a CSV file optionally deleting some keys and making the columns
        consistent across all rows by adding header as superset of all keys and adding blanks to rows where necessary.
    """
    all_keys = set()
    for result in all_results:
        if del_keys:
            for del_key in del_keys:
                if del_key in result:
                    del result[del_key] 
        all_keys.update(result.keys())
    for result in all_results:
        for key in all_keys:
            if key not in result:
                result[key] = None
    with open(csv_file_path, 'w') as csv_file:
        fieldnames = all_results[0].keys() if all_results else []
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=delimiter, lineterminator=lineterminator)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)   
    if verbose:
        print(f"Results written to {csv_file_path}")
    return csv_file_path


class PriorityQueue:
    """ Priority Queue implementation supporting 3 levels of priority: heuristic value, tiebreaker1, tiebreaker2
    ie list of tuples (priority, tiebreaker1, tiebreaker2, item)
    """
    def __init__(self, tiebreaker2='FIFO'):
        self.heap = []
        self.tiebreaker2 = tiebreaker2
        if tiebreaker2 == 'FIFO':
            self.increment = 1
        elif tiebreaker2 == 'LIFO':
            self.increment = -1
        else:
            raise ValueError("tiebreaker2 must be either 'FIFO' or 'LIFO'")
        self.count = 0
        self.max_heap_size = 0

    def push(self, item, priority, tiebreaker1=0):
        entry = (priority, tiebreaker1, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += self.increment
        if self.max_heap_size < len(self.heap):
            self.max_heap_size = len(self.heap)

    def pop(self, item_only=True):
        if not self.isEmpty():
            priority, tiebreaker1, tiebreaker2, item = heapq.heappop(self.heap)
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



