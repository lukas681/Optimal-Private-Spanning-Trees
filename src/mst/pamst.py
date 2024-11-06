import math
import numpy as np
from itertools import count

## Inspired by https://github.com/networkx/networkx/blob/main/networkx/algorithms/tree/mst.py
# @nx._dispatch(edge_attrs="weight", preserve_edge_attrs="data")
def pamst(G, noise_lambda=1):
    """Iterate over edges of Prim's algorithm to return a MINIMUM spanning tree
    Primitive Implementation

    Parameters
    ----------
    G : NetworkX Graph
        The graph holding the tree of interest.

    minimum : bool (default: True)
        Find the minimum (True) or maximum (False) spanning tree.
    """
    nodes = set(G)
    c = count()

    while nodes:
        u = nodes.pop()
        frontier = [ ]  # Creates a heap
        visited = { u } # set of visited vertices
        for v, d in G.adj[u].items():
            wt = d.get("weight", 1)
            frontier.append((wt, next(c),u,v,d)) # Is it slow?

        while nodes and frontier:
            W, z, u, v, d = report_noisy_max_zcdp(frontier, noise_lambda)
            frontier.remove((W,z,u,v,d)) # Might be expensive.

            if v in visited or v not in nodes:
                continue # Ignore
            yield u, v, d # Return the next edge
            # update frontier
            visited.add(v)
            nodes.discard(v)
            for w, d2 in G.adj[v].items():
                if w in visited:
                    continue
                new_weight = d2.get("weight", 1)
                frontier.append((new_weight, next(c),v, w, d2))

def report_noisy_max_zcdp(frontier, noise_lambda):
    """
    Takes a frontier and returns a noisy edge.
    Uses Expüonential Noise.
    """
    noisy_max_edge = (None,None)
    for e in frontier:
        noisy_value = (e[0] + np.random.exponential(noise_lambda))
        is_new = noisy_value if (
                noisy_max_edge[1] is None) \
            else (noisy_value < noisy_max_edge[1])
        noisy_max_edge = noisy_max_edge if not is_new else (e, noisy_value)
    return noisy_max_edge[0]
def comp_mst_weight(edges):
    return sum(e[2]["weight"] for e in edges)
