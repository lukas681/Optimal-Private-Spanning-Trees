import logging
logger = logging.getLogger(__name__)

from time import perf_counter_ns, perf_counter
import math
import random
import numpy as np
import networkx as nx
from networkx import Graph
import src.mst.pamst as pt

def generate_random_complete_graph(n, upper=1):
    """
    Generates a complete graph with uniformly drawn random weights
    :param n: size of the graph
    :param range: Draws each edge weight from Uni(0, range)
    :return: nx.Graph with the specified parameters
    """

    G: Graph = nx.complete_graph(n)

    # Initializing the Edge Weights
    for (u, v, w) in G.edges(data=True):
        w['weight'] = random.random() * upper
    return G


def generate_random_erdos_reny_graph(n=1, p=1, max_edge_weight=1, force_connected=True):
    """
    Generates a complete graph using the Erdos-Renyi Random Graph model G(n, p).
    Might be slower than the corresponding generate_random_complete_graph for p=1

    :param max_edge_weight: Draws each edge weight from Uni(0, max_edge_weight)
    :param n: size of the graph
    :param force_connected: Force a connected instance. Will just run again if it is not
    :return: Graph constructed from the desired parameters.
    """
    start = perf_counter_ns()
    G = nx.Graph()
    while True:
        G.add_nodes_from(range(n)) # How slow is this?

        for i in range(0, n):
            for j in range(0, i):
                if random.random() < p:
                    G.add_edge(i, j, weight=1)

        # Initializing the Edge Weights
        for (u, v, w) in G.edges(data=True):
            w['weight'] = random.random() * max_edge_weight
        logger.debug(f'Initialization took: {perf_counter_ns() - start}')
        if nx.is_connected(G) or not force_connected:
            break
    return G


def generate_mi_instance(n, p):
    """
    Generates a realistic MI scenario: Given a dataset with n features X_i,
    we set X_i = 1- X_{i-1} w.p p and X_{i-1} otherwise
    The overall structure of the MI matrix can be computed directly
    :param n:
    :param p:
    :return:
    """
    A = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, i):
            k = i - j
            p00 = (1 + (1 - 2 * p) ** k)
            p10 = (1 - (1 - 2 * p) ** k)
            mi = 1 / 2 * p00 * np.log2(p00) + 1 / 2 * p10 * np.log2(p10)
            A[i][j] = -mi  # Minimum Spanning Tree Finds Max Mutual Information
            A[j][i] = -mi
    G: Graph = nx.from_numpy_array(A=A, parallel_edges=False, create_using=nx.Graph)
    return G


def generate_hard_instance(n, E):
    """
    Generates the hard instance as described in [Sealfon 2016]
    The graph constains of a center in the middle surrounded by triangles.
    :param n:
    :param E:
    :return:
    """

    G: Graph = nx.Graph()
    # Center = v_0
    for i in range(1, 2 * n + 1):
        G.add_edge(0, i, weight=(0 if i % 2 == 0 else E))
    for i in range(1, n + 1):
        G.add_edge(2 * i - 1, 2 * (i), weight=0)
    return G


def compute_real_mst_weight(G, alg='prim'):
    """
    Returns the weight of a minimum spanning tree in G
    :param G: Networkx Graph
    :param alg: Supports ["prim", "kruskal"]
    :return:
    """
    T = nx.minimum_spanning_tree(G, algorithm=alg)
    weight = 0
    for _, _, d in T.edges(data=True):
        weight += d['weight']
    return weight


def compute_input_perturbation(G, noise_fkt, alg='prim'):
    """
    Implementation of Input Perturbation Mechanisms.
    Covers our approach and Sealfon's postprocessing.
    :param G:
    :param noise_fkt: A function /(f:weights \rightarrow R/)
    :param alg: The weight of the MST using input perturbation
    :return:
    """

    weights = 0
    old_weights = {}

    # Storing old values might be inefficient for large graphs, but we need them to restore to default
    # as our mechanism is not just addive noise.
    # + Avoids floating issues
    #    old_weights = {(u, v): w['weight'] for u, v, w in G.edges(data=True)}

    for (u, v, w) in G.edges(data=True):
        old_weights[(u, v)] = w['weight']
        w['weight'] = noise_fkt(w['weight'])

    T = nx.minimum_spanning_tree(G, algorithm=alg)
    for u, v, d in T.edges(data=True):
        weights += (old_weights[(u, v)])  # MST on the ground graph
    return weights



def compute_approximate_dp(G: Graph, sensitivity=1, rho_values=[1], run_real=True, run_sealfon=True, run_pamst=True, run_our=True):
    """
    The main entrypoint for the experiments.
     Allows to run a subset of MST algorithm
    Returns a dictionary containing keys 'sealfon','pamst', 'our' and 'real'
    :param G: NetworkX Graph
    :param sensitivity: the $\ell_\infty$v sensitivity of the graph.
    :param rho_values:
    :param run_real: Flag indicating whether the REAL mst should be computed
    :param run_sealfon: Flag indicating whether we run SEALFON's postprocessing alg.
    :param run_pamst: Flag indicating whether to run PAMST
    :param run_our: Flag indicating whether to run OUR approach
    :return:
    """
    n = G.number_of_nodes()  # Does it work?
    results = {}

    ### Real Spanning Tree ###
    # Simply make an array to make visualization easier
    results['real'] = []
    start = perf_counter_ns()
    if run_real:
        results['real'] = [compute_real_mst_weight(G)] * len(rho_values)
    logger.debug(f'computing the real MST took: {perf_counter_ns() - start}')

    ### Pinot's PAMST Algorithm ###
    results['pamst'] = []
    if run_pamst:
        for rho in rho_values:
            start = perf_counter_ns()
            noise_level = (2 * sensitivity * math.sqrt((n - 1) / (2 * rho)))  # Should be ok
            pamst_edges = pt.pamst(G.copy(),
                                noise_scale=noise_level)  # Gives an iterator which should only be executed once!
            results['pamst'] += [pt.comp_mst_weight(pamst_edges)]
            logger.debug(f'computing PAMST MST took: {perf_counter_ns() - start}')

    ### Sealfon's Post Processing Technique ###
    results['sealfon'] = []
    if run_sealfon:
        for rho in rho_values:
            start = perf_counter_ns()
            std_deviation = sensitivity * math.sqrt(G.number_of_edges() / (2 * rho))
            gaussNoise = lambda edge_weight: edge_weight + np.random.normal(0, std_deviation)
            results['sealfon'] += [compute_input_perturbation(G.copy(), gaussNoise)]
            logger.debug(f'computing SEALFON took: {perf_counter_ns() - start}')

    ### Finally: Our Approach ###
    results['our'] = []
    if run_our:
        for rho in rho_values:
            start = perf_counter_ns()
            noise_lambda = math.sqrt(2 * rho / (n - 1)) / (2 * sensitivity)
            expNoise = lambda edge_weight: np.log(np.random.exponential(1)) + noise_lambda * edge_weight
            results['our'] += [compute_input_perturbation(G.copy(), expNoise, alg='prim')]
            logger.debug(f'computing OUR APPROACH took: {perf_counter_ns() - start}')
    return results

def compute_different_densities_approximate_dp(n, edge_probabilities, sensitivity, maximum_edge_weight, rho, number_of_runs=1):
    """
    Generates a G(n, p) random graph with random edge weights from Uni(0, maximum_edge_weight).
    Then runs all algorithms number_of_runs times.

    :param n: Size of the graph to test
    :param edge_probabilities: list of probabilities of edges to test, densitiv
    :param sensitivity:
    :param maximum_edge_weight:
    :param number_of_runs: number of runs per instance
    :param rho: Privacy level in $\rho$-zCDP
    :return: primitive list with all datapoints. Made easy to parse by matplotloib / seaborn
    """
    results = []
    for edge_p in edge_probabilities:
        logger.debug(f'working on G({n},{edge_p})')
        if edge_p < np.log2(n)/n: logger.warning("Graph might not be connected as p < log(n)/n!")
        G = generate_random_erdos_reny_graph(n, p=edge_p, max_edge_weight=maximum_edge_weight)
        for i in range(1, number_of_runs+1):
            start = perf_counter_ns()
            res = compute_approximate_dp(G=G, sensitivity=sensitivity, rho_values=rho)
            for key in res:
                real_mst = res['real'][0] # normalization
                logger.debug(dict(p=edge_p, type=key, value=res[key][0]/real_mst))
                results += [(dict(p=edge_p, type=key, value=res[key][0]/real_mst))]
            logger.debug(f'A complete run finshed after {perf_counter_ns() - start}ms')
    logger.info("computation complete. Initializing the plots.")
    return results