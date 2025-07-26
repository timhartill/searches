

# --- Minimum Vertex Cover runnable example ---
# Finds all Minimum Weighted Vertex Covers over a bipartite graph

# We define them here with placeholder values so the script can be run.
# 'forward_cluster' and 'backward_cluster' are assumed to be lists of tuples
# where each tuple is (value, weight) and both forward and backward are assumed to be sorted by ascending value ("g") 
# and weight is the the number of states with this g-value in this direction. 
#forward_cluster = [(10, 1), (11, 1), (15, 1)]
#backward_cluster = [(10, 1), (11, 1), (13, 1), (14,1)]
#forward_cluster = [(10, 2), (11, 1), (15, 1)]
#backward_cluster = [(10, 1), (11, 1), (13, 1), (14,1)]
#forward_cluster = [(10, 1), (11, 1), (15, 3)]
#backward_cluster = [(10, 1), (11, 1), (13, 1), (14,1)]
#forward_cluster = [(10, 4), (11, 1), (15, 3)]
#backward_cluster = [(10, 1), (11, 1), (13, 1), (14,1)]
#forward_cluster = [(10, 4), (11, 1), (15, 1)]
#backward_cluster = [(10, 2), (11, 1), (13, 2), (14,1)]
forward_cluster = [(11, 10), (15, 2)]
backward_cluster = [(10, 2), (11, 2), (13, 1), (14,7)]



min_edge_cost = 1.0   # Epsilon is the minimum edge cost.
# GLB can also be C* if running over entire GMX.
#GLB = 23.0
GLB = 26.0

# translate (g, weight) tuples into our dictionary format (critical key only) from data_structures.lb_pairs.calc_expandable() for testing
forward_expandable_g = {g: {"g_total_count": weight} for g, weight in forward_cluster}
backward_expandable_g = {g: {"g_total_count": weight} for g, weight in backward_cluster}
# self.forward_expandable_g[gF]["g_total_count"]


def find_minimum_weighted_vertex_cover(forward_expandable_g, backward_expandable_g, min_edge_cost, GLB, return_covers=False, verbose=False):
    """
    Finds all minimum weighted vertex covers (MWVC) based on weighted nodes where a node is a g-level and weight is the number of states in that g.
    Returns the minimum value ie |MVC|, lists of ascending forward/backward g levels in any mvc 
    and a list of tuples representing the minimal vertex covers. Each tuple contains the indices (i, j) of the forward and backward clusters.
    Nodes in the MWVC for a given tuple (i,j) are forward_expandable_g.keys()[:i+1] and backward_expandable_g.keys()[:j+1].
    An i or j value of -1 indicates no nodes in that direction are in the MWVC for that tuple.

    Key Args: 2 dictionaries with keys as g levels and values as dictionaries with at least the key "g_total_count" (weight):
    - forward_expandable_g: key: gF (sorted ascending), value: {"g_total_count": weight} <- there are other keys in the value if calling from calc_expandable() but "g_total_count" is the only required one here
    - backward_expandable_g: key: gB (sorted ascending), value: {"g_total_count": weight}
    """
    forward_g_list = list(forward_expandable_g.keys())
    backward_g_list = list(backward_expandable_g.keys())
    forward_g_mwvc = []   # forward g levels in MWVC
    backward_g_mwvc = []
    min_value = float('inf') 
    minimum_vertex_covers = []   # A list of tuples will store the (i, j) pairs
    max_i = -1
    max_j = -1
    
    num_forward_in_vc = 0
    for i in range(-1, len(forward_g_list)):       # Iterates from -1 up to the number of forward g levels.
        if i > -1:
            num_forward_in_vc += forward_expandable_g[ forward_g_list[i] ]['g_total_count']  # Accumulate the count from the forward g level.

        num_backward_in_vc = 0
        for j in range(-1, len(backward_g_list)):  # Iterates from -1 up to the number of backward g levels.
            if j > -1:
                num_backward_in_vc += backward_expandable_g[ backward_g_list[j] ]['g_total_count']   # Accumulate the weight from the backward g level.
            should_break = False
            current_sum = 0
            
            if i == len(forward_g_list)-1:       # Condition 1: We are at the last element of the forward cluster.
                current_sum = num_backward_in_vc + num_forward_in_vc
                should_break = True
                if verbose: print(f"1. i={i}, j={j} num_backward_in_vc={num_backward_in_vc} num_forward_in_vc={num_forward_in_vc} (GLB={GLB})")
                        
            elif j == len(backward_g_list)-1:    # Condition 2: We are at the last element of the backward cluster.
                current_sum = num_backward_in_vc + num_forward_in_vc
                should_break = True
                if verbose: print(f"2. i={i}, j={j} num_backward_in_vc={num_backward_in_vc} num_forward_in_vc={num_forward_in_vc} (GLB={GLB})")
                        
            elif (backward_g_list[j+1] + forward_g_list[i+1] + min_edge_cost) > GLB:  # Condition 3: No more edges ie gF + gB + eps > GLB
                current_sum = num_backward_in_vc + num_forward_in_vc
                should_break = True
                if verbose: print(f"3. i={i}, j={j} num_backward_in_vc={num_backward_in_vc} num_forward_in_vc={num_forward_in_vc} (GLB={GLB})")

            if should_break:
                if current_sum < min_value:   # If we found a new absolute minimum, discard the old list of covers
                    min_value = current_sum
                    minimum_vertex_covers = [(i, j)]
                    max_i = i
                    max_j = j
                elif current_sum == min_value:  # If we found a value equal to the current minimum, add it to the list.
                    minimum_vertex_covers.append((i, j))
                    max_i = max(max_i, i)
                    max_j = max(max_j, j)
                break       # Break the inner loop
    
    if max_i > -1:
        forward_g_mwvc = forward_g_list[:max_i+1]
    if max_j > -1:
        backward_g_mwvc = backward_g_list[:max_j+1]

    if not return_covers:  
        return min_value, forward_g_mwvc, backward_g_mwvc
    return min_value, minimum_vertex_covers, forward_g_mwvc, backward_g_mwvc, max_i, max_j

# --- Execution ---
if __name__ == "__main__":
    min_val, covers, forward_g_mwvc, backward_g_mwvc, max_i, max_j = find_minimum_weighted_vertex_cover(forward_expandable_g, backward_expandable_g, min_edge_cost, GLB, return_covers=True, verbose=True)
    size_mwvc = covers[0][0] + 1 + covers[0][1] + 1 

    print(f"|MWVC|: {min_val}. Number of vertices in 1st MWVC: {size_mwvc}")
    print(f"Number of Minimum Vertex Covers: {len(covers)}")
    print("Minimum Vertex Covers (i, j pairs):")
    for cover in covers:
        print(f"  {cover}")
    print(f"Forward nodes in any MWVC up to and including idx:{max_i} and Backward nodes up to and including idx:{max_j}")
    print(f"Forward g levels in any MWVC: {forward_g_mwvc}")
    print(f"Backward g levels in any MWVC: {backward_g_mwvc}")

    min_val, forward_g_mwvc, backward_g_mwvc = find_minimum_weighted_vertex_cover(forward_expandable_g, backward_expandable_g, min_edge_cost, GLB, return_covers=False, verbose=False)

    print(f"|MWVC|: {min_val}.")
    print(f"Forward g levels in any MWVC: {forward_g_mwvc}")
    print(f"Backward g levels in any MWVC: {backward_g_mwvc}")


