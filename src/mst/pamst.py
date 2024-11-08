import math
import numpy as np
from itertools import count

## Inspired by https://github.com/networkx/networkx/blob/main/networkx/algorithms/tree/mst.py
# @nx._dispatch(edge_attrs="weight", preserve_edge_attrs="data")
def pamst(G, noise_scale=1):
    """Iterate over edges of Prim's algorithm to return a MINIMUM spanning tree
    Primitive Implementation

    Parameters
    ----------
    G : NetworkX Graph
        The graph holding the tree of interest.
    """
    nodes = set(G)
    c = count()

    while nodes:
        u = nodes.pop()
        frontier = [ ]  # Creates a heap
        visited = { u } # set of visited vertices
        for v, d in G.adj[u].items():
            wt = d.get("weight", 1)
            frontier.append((-wt, next(c),u,v,d))
        while nodes and frontier:
            W, z, u, v, d = report_noisy_max(frontier, noise_scale)
            print()
            frontier.remove((W,z,u,v,d))

            if v in visited or v not in nodes:
                continue
            yield u, v, d # Return the next edge

            # update frontier
            visited.add(v)
            nodes.discard(v)
            for w, d2 in G.adj[v].items():
                if w in visited:
                    continue
                new_weight = d2.get("weight", 1)
                frontier.append((-new_weight, next(c),v, w, d2))

def report_noisy_max(frontier, noise_scale):
    """
    Implementation of Report Noisy Max using Exponential Noise
    """
    noisy_max_edge = (None, None)
    for e in frontier:
        noise = np.random.exponential(noise_scale)
        found_new_max = (noisy_max_edge[1] is None) or (noise + e[0] > noisy_max_edge[1])
        noisy_max_edge = noisy_max_edge if not found_new_max \
            else (e, noise)
    return noisy_max_edge[0]

def comp_mst_weight(edges):
    return sum(e[2]["weight"] for e in edges)
