import matplotlib.pyplot as plt
import numpy as np
def encode_experiment_one(results, edge_probabilites, rho_values, n, maximum_edge_weight, sensitivity):
    return {
        "results": results,
        "rho_values": rho_values,
        "edge_probabilities": edge_probabilites,
        "n": n,
        "maximum_edge_weight": maximum_edge_weight,
        "sensitivity": sensitivity
    }
def save_results(results, filename="save/test.npy"):
    """
    Takes a dictionary and saves all entries to a file.
    :param results:
    :param filename:
    :return:
    """
    with open(filename, 'wb') as f:
        np.savez(f, **results)

def save_plot_and_data(results_complete, edge_probabilities, rho_values, n, maximum_edge_weight, sensitivity):
    file_name = "save/complete_n{}_sens{}_range0-{}x".format(n, sensitivity, maximum_edge_weight)
    save_results(filename=file_name + ".npy", results=encode_experiment_one(
        results_complete, edge_probabilities, rho_values,n, maximum_edge_weight, sensitivity))
    plt.savefig(file_name + ".png")


def load_results(filename):
    return np.load(filename, allow_pickle=True)
