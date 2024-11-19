import matplotlib.pyplot as plt
import numpy as np
import logging
logger = logging.getLogger(__name__)

def encode_experiment_one(results, edge_probabilites, rho_values, n, maximum_edge_weight, sensitivity):
    return
def save_results(results, filename="save/test.npy"):
    """
    Takes a dictionary and saves all entries to a file.
    :param results:
    :param filename:
    :return:
    """
    with open(filename, 'wb') as f:
        np.savez(f, **results)

def save_plot_and_data_experiment_one(file_name, results_complete, edge_probabilities, rho_values, n, maximum_edge_weight, sensitivity):
    save_results(filename=file_name + ".npy", results=
    {
        "results": results_complete,
        "rho_values": rho_values,
        "edge_probabilities": edge_probabilities,
        "n": n,
        "maximum_edge_weight": maximum_edge_weight,
        "sensitivity": sensitivity
    })
    plt.savefig(file_name + ".png")
    plt.savefig(file_name + ".pdf", format='pdf', bbox_inches='tight')

def save_plot_and_data_experiment_two(results_complete, rho_values, n, sensitivity):
    file_name = f'save/mutual-information_n{n}_sens{sensitivity}'
    save_results(filename=file_name + ".npy", results=
    {
        "results": results_complete,
        "rho_values": rho_values,
        "n": n,
        "sensitivity": sensitivity
    })
    plt.savefig(file_name + ".png")
    plt.savefig(file_name + ".pdf", format='pdf', bbox_inches='tight')

def load_results(filename):
    logger.debug(f'Loading from disk. file {filename}')
    return np.load(filename, allow_pickle=True)