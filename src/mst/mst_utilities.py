import math
import random
import numpy as np
import networkx as nx
from networkx import Graph
from .pamst import comp_mst_weight, pamst

def generate_random_complete_graph(n):
    """
    Generates a complete graph with uniformly drawn random weights
    """
    G: Graph = nx.complete_graph(n)

    # Initializing the Edge Weights
    for (u, v, w) in G.edges(data=True):
        w['weight'] = random.random()
    return G

def generate_mi_instance(n, p):
    """
    Generates a realistic MI scenario: Given a dataset with n features X_i,
    we set X_i = 1- X_{i-1} w.p p and X_{i-1} otherwise
    The overall structure of the MI matrix can be computed directly
    """
    A= np.zeros((n,n))
    for i in range(0, n):
        for j in range (0, i):
            k = i-j
            p00 = (1 + (1-2*p)**k)
            p10 = (1 - (1-2*p)**k)
            mi = 1/2 * p00 * np.log2(p00) + 1/2 * p10 * np.log2(p10)
            A[i][j] = mi
            A[j][i] = mi
    G: Graph = nx.from_numpy_array(A=A,parallel_edges=False, create_using=nx.Graph)
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


def compute(G, sensitivity=1, rho_values=[1], run_real=True, run_sealfon=True, run_pamst=True, run_our=True):
    """
    Allows to run a subset of MST algorithm
    Returns a dictionary containing keys 'sealfon','pamst', 'our' and 'real'
    """
    n = G.number_of_edges()  # Does it work?
    results = {}

    ### Real Spanning Tree ###
    # Simply make an array to make visualization easier
    if run_real:
        results['real'] = [compute_real_mst_weight(G)] * len(rho_values)

    ### Pinot's PAMST Algorithm ###
    if run_pamst:
        results['pamst'] = []
        for rho in rho_values:
            noise_level = (1/2 * sensitivity * math.sqrt( (n-1)/(2 * rho))) # Should be ok
            print("pamst: " + str(noise_level))
            pamst_edges = pamst(G, noise_scale=noise_level) # Gives an iterator which should only be executed once!
            results['pamst'] += [comp_mst_weight(pamst_edges)]

    ### Sealfon's Post Processing Technique ###
    if run_sealfon:
        results['sealfon'] = []
        for rho in rho_values:
            std_deviation = sensitivity * math.sqrt(G.number_of_edges() / (2*rho))
            gaussNoise = lambda edge_weight: edge_weight + np.random.normal(0, std_deviation)
            results['sealfon'] += [compute_input_perturbation(G.copy(), gaussNoise)]

    ### Finally: Our Approach ###
    if run_our:
        results['our'] = []
        for rho in rho_values:
            noise_lambda = 2* 1/(sensitivity) * math.sqrt(2*rho/(n-1)) ## TODO: Check, whether 1/sensitivity is correct, Previously 2* sens+ rest
            expNoise = lambda edge_weight: np.log(np.random.exponential(1) ) + noise_lambda * edge_weight
            results['our'] += [compute_input_perturbation(G.copy(), expNoise, alg='prim')]
    return results