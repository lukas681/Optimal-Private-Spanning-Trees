import math
import numpy as np
from itertools import count
from networkx import Graph

## Inspired by https://github.com/networkx/networkx/blob/main/networkx/algorithms/tree/mst.py
# @nx._dispatch(edge_attrs="weight", preserve_edge_attrs="data")
def pamst(G:Graph, noise_scale=1):
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
        frontier = {}  # Dictionary should be faster at least for constant time lookup
        visited = { u } # set of visited vertices
        for v, d in G.adj[u].items():
            wt = d.get("weight",1)
            frontier[(u, v)] = (-wt, next(c), d)
            #frontier.append((-wt, next(c),u,v,d))
        while nodes and frontier:
            noisy_edge = report_noisy_max(frontier, noise_scale)
            (u, v), (W, _, d) = noisy_edge
            del frontier[(u, v)]  # O(1) deletion from the dictionary

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
                frontier[(v, w)] = (-new_weight, next(c), d2)
#                frontier.append((-new_weight, next(c),v, w, d2))

def report_noisy_max(frontier, noise_scale):
    """
    Implementation of Report Noisy Max using Exponential Noise.

    Parameters
    ----------
    frontier : dict
        A dictionary where keys are edges (u, v) and values are tuples of
        (-weight, counter, edge_data).
    noise_scale : float
        Scale of the exponential noise.

    Returns
    -------
    tuple
        The edge with the maximum noisy weight and its associated data.
    """
    noisy_max_edge = None
    max_noisy_score = float('-inf')

    for edge, (weight, counter, data) in frontier.items():
        noise = np.random.exponential(noise_scale)
        noisy_score = weight + noise

        if noisy_score > max_noisy_score:
            max_noisy_score = noisy_score
            noisy_max_edge = (edge, (weight, counter, data))

#    for e, (weight, counter, data) in frontier.items():
#        noise = np.random.exponential(noise_scale)
#        found_new_max = (noisy_max_edge[1] is None) or (noise + e[0] > noisy_max_edge[1])
#        noisy_max_edge = noisy_max_edge if not found_new_max \
#            else (e, noise + e[0])
    return noisy_max_edge

def comp_mst_weight(edges):
    return sum(e[2]["weight"] for e in edges)
