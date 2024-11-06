import math
import random
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

def compute_real_mst_weight(G, alg='prim'):
    """

    """
    T  = nx.minimum_spanning_tree(G, algorithm = alg)
    weight = 0
    for _,_,d in T.edges(data=True):
        weight += d['weight']
    return weight

def compute_input_perturbation(G, noise_level, noise_fkt):
    """
    """
    G_copy = G.copy() ## Enables modification of edges = noise addition
    noise_terms = {}
    weight = 0
    for (u, v, w) in G_copy.edges(data=True): # Takes care of symmetry
        noise_terms[(u, v)] = noise_fkt(0, noise_level)
        w['weight'] +=  noise_terms[(u,v)]

    T  = nx.minimum_spanning_tree(G_copy, algorithm ='prim')
    for u,v,d in T.edges(data=True):
        weight += (d['weight']-noise_terms[(u,v)] ) # subtracts the noise again.
    return weight