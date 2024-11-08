import math
import random
import numpy as np
import networkx as nx
from networkx import Graph

def generate_random_complete_graph(n):
    """
    Generates a complete graph with uniformly drawn random weights
    """
    G: Graph = nx.complete_graph(n)

    # Initializing the Edge Weights
    for (u, v, w) in G.edges(data=True):
        w['weight'] = random.random()
    return G
def generate_hard_instance(n, E):
    """
    Generates the hard instance as described in [Sealfon 2016]
    The graph constains of a center in the middle surrounded by triangles.
    """
    # G: Graph = nx.Graph(n + (n-1))
    G: Graph = nx.Graph()
    # Center = v_0
    for i in range(1, 2*n+1):
        G.add_edge(0, i,weight = (0 if i%2==0 else E))
    for i in range(1, n+1):
        G.add_edge(2*i-1, 2*(i), weight =  0)
    return G

def compute_real_mst_weight(G, alg='prim'):
    """
    """
    T  = nx.minimum_spanning_tree(G, algorithm = alg)
    weight = 0
    for _,_,d in T.edges(data=True):
        weight += d['weight']
    return weight

def compute_input_perturbation(G, noise_fkt, alg='prim'):
    """
    Implementation of Input Perturbation Mechanisms.
    Covers our approach and Sealfon's postprocessing.
    """
    weights = 0
    old_weights = {}

    # Storing old values might be inefficient for large graphs, but we need them to restore to default
    # as our mechanism is not just addive noise.
    # + Avoids floating issues
#    old_weights = {(u, v): w['weight'] for u, v, w in G.edges(data=True)}

    for (u, v, w) in G.edges(data=True):
        old_weights[(u,v)] = w['weight']
        w['weight'] = noise_fkt(w['weight'])

    T  = nx.minimum_spanning_tree(G, algorithm=alg)
    for u,v,d in T.edges(data=True):
        weights += (old_weights[(u,v)]) # MST on the ground graph
    return weights
