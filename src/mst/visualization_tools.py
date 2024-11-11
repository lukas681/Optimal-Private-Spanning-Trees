import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def init_plot(results, rho_values, title):
    """
    Requires a plt.show() later.
    :param results:
    :param rho_values:
    :return:
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 2x2 grid
    fig.suptitle(title)
    (sealfon, our, pamst, real) = (results['sealfon'], results['our'], results['pamst'], results['real'])
    # Error of the MST
    sns.lineplot(x=rho_values, y=sealfon, marker='o', label="Sealfon", ax=axs[0])
    # sns.lineplot(x=rho_values, y=upper, marker='o', label="upperbound", ax=axs[0])
    sns.lineplot(x=rho_values, y=our, marker='o', label="$\\textbf{Our}$", ax=axs[0])
    sns.lineplot(x=rho_values, y=pamst, marker='o', label="PAMST", ax=axs[0])
    sns.lineplot(x=rho_values, y=real, marker='o', label="Real MST", ax=axs[0])
    axs[0].set_title("Absolute Weight of the MST")
    # Error of the MST
    sns.lineplot(x=rho_values, y=[sealfon - real for sealfon, real in zip(sealfon, real)], marker='o', label="Sealfon",
                 ax=axs[1])
    sns.lineplot(x=rho_values, y=[ours - real for ours, real in zip(our, real)], marker='o', label="$\\textbf{Our}$",
                 ax=axs[1])
    sns.lineplot(x=rho_values, y=[pamst - real for pamst, real in zip(results['pamst'], real)], marker='o',
                 label="PAMST", ax=axs[1])

    axs[1].set_title("Additive Error of the MST")
    axs[0].set_ylabel("Absolute Weight")
    # axs[0].set_ylim([0,100])

    axs[0].set_xlabel("$\\rho$")
    axs[1].set_ylabel("Additive Error")

    plt.legend(title="Comparing Private MST Algs")

def init_multiplot(all_results, rho_values, meta_params, columns=2):

    number_of_rows = int(np.ceil(min(columns, len(all_results)) / columns))
    fig, axs = plt.subplots(number_of_rows, columns, figsize=(20, 17))  # Grid layout
    fig.suptitle(meta_params["title"])

    # Turn off all axes by default
    for ax in axs.flat:
        ax.axis('off')

    for index, results in enumerate(all_results):
        (sealfon, our, pamst, real) = (results['sealfon'], results['our'], results['pamst'], results['real'])
        # Error of the MST
        # sns.lineplot(x=rho_values, y=sealfon, marker='o', label="Sealfon", ax=axs[index])
        # sns.lineplot(x=rho_values, y=upper, marker='o', label="upperbound", ax=axs[0])
        # sns.lineplot(x=rho_values, y=our, marker='o', label="$\\textbf{Our}$", ax=axs[index])
        # sns.lineplot(x=rho_values, y=pamst, marker='o', label="PAMST", ax=axs[index])
        # sns.lineplot(x=rho_values, y=real, marker='o', label="Real MST", ax=axs[index])
        # axs[index].set_title("Absolute Weight of the MST")
        # Error of the MST
        (figureX, figureY) = (index // columns, index % columns)
        sns.lineplot(x=rho_values, y=[sealfon - real for sealfon, real in zip(sealfon, real)], marker='o',
                     label="Sealfon",
                     ax=axs[figureX][figureY])
        sns.lineplot(x=rho_values, y=[ours - real for ours, real in zip(our, real)], marker='o',
                     label="$\\textbf{Our}$",
                     ax=axs[figureX][figureY])
        sns.lineplot(x=rho_values, y=[pamst - real for pamst, real in zip(results['pamst'], real)], marker='o',
                     label="PAMST", ax=axs[figureX][figureY])
        title = "G({}, {})".format(meta_params['graph-size'], round(meta_params['edge_probabilities'][index], 1)
                                   )
        axs[figureX][figureY].set_title(title)
        axs[figureX][figureY].set_xlabel("$\\rho$")
        axs[figureX][figureY].set_ylabel("MST Error")
        axs[figureX][figureY].axis('on')

        # axs[0].set_ylim([0,100])
        # axs[index].set_ylabel("Additive Error")

#     plt.legend(title="Comparing Private MST Algs")


def convert_results(results):
    return ([i * -1 for i in results['sealfon']],
            [i * -1 for i in results['our']],
            [i * -1 for i in results['pamst']],
            [i * -1 for i in results['real']])
