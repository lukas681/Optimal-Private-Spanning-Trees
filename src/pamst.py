import math
import numpy as np
from itertools import count

## Inspired by https://github.com/networkx/networkx/blob/main/networkx/algorithms/tree/mst.py
# @nx._dispatch(edge_attrs="weight", preserve_edge_attrs="data")
def pamst(G, minimum=False, sensitivity=1, rho=1):
    """Iterate over edges of Prim's algorithm min/max spanning tree.
    Primitive Implementation,.

    Parameters
    ----------
    G : NetworkX Graph
        The graph holding the tree of interest.

    minimum : bool (default: True)
        Find the minimum (True) or maximum (False) spanning tree.

    """
    # push = heappush ## Can't do it any more
    # pop = heappop ## Cant use it any more

    nodes = set(G)
    rounds = len(nodes) - 1
    c = count()
    noisy_scale = (2 * sensitivity * math.sqrt(rounds))/(math.sqrt(2 * rho))
    sign = 1 if minimum else -1
    while nodes:
        u = nodes.pop()
        frontier = [] # Creates a heap
        visited = {u} # set of visited vertices
        for v, d in G.adj[u].items():
            wt = d.get("weight", 1) * sign
            frontier.append((wt, next(c),u,v,d)) # Is it slow?
        while nodes and frontier:
            W, z, u, v, d = report_noisy_max_zcdp(frontier, noisy_scale)
            frontier.remove((W,z,u,v,d)) # Might be expensive.

            if v in visited or v not in nodes:
                # Delete from
                continue
            yield u, v, d # Return the next edge
            # update frontier
            visited.add(v)
            nodes.discard(v)
            for w, d2 in G.adj[v].items():
                if w in visited:
                    continue
                new_weight = d2.get("weight", 1) * sign
                frontier.append((new_weight, next(c),v, w, d2))

def report_noisy_max_zcdp(frontier, noise_scale):
    noisy_max_edge = (None,None)

    for e in frontier:
        noisy_value = (e[0] + np.random.exponential(noise_scale)) # e[0] = w
        is_new = noisy_value if (
                noisy_max_edge[1] is None) \
            else (noisy_value < noisy_max_edge[1])
        noisy_max_edge = noisy_max_edge if not is_new else (e, noisy_value)

    return noisy_max_edge[0]