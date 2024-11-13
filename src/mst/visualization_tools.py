import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def init_plot(results, rho_values, meta_params):
    """
    Requires a plt.show() later.
    :param results:
    :param rho_values:
    :return:
    """
    fig, ax= plt.subplots(figsize=(10, 10))  # 2x2 grid
    # fig = plt.plot( figsize=(10, 5))  # 2x2 grid
    ax.set_title(meta_params["title"])
    (sealfon, our, pamst, real) = (results['sealfon'], results['our'], results['pamst'], results['real'])
    sns.lineplot(x=rho_values, y=[sealfon - real for sealfon, real in zip(sealfon, real)], marker='o', label="Sealfon",
                 ax=ax)
    sns.lineplot(x=rho_values, y=[ours - real for ours, real in zip(our, real)], marker='o', label="$\\textbf{Our}$",
                ax=ax)
    sns.lineplot(x=rho_values, y=[pamst - real for pamst, real in zip(results['pamst'], real)], marker='o',
                 label="PAMST", ax=ax)
    ax.set_ylabel("Additive Error")
    # plt.legend(title="Comparing Private MST ")
   # plt.subplots_adjust(hspace = 0.4)

def init_multiplot(all_results, rho_values, meta_params, columns=2):
    number_of_rows = max(1,
                         int(
                             np.ceil(len(all_results)) / columns)) + 1
    fig, axs = plt.subplots(number_of_rows, columns, figsize=(20, 20))  # Grid layout
    fig.suptitle(meta_params["title"])

    # we only want to see used axes
    for ax in axs.flat:
        ax.axis('off')

    for index, results in enumerate(all_results):
        (sealfon, our, pamst, real) = (results['sealfon'], results['our'], results['pamst'], results['real'])

        # Error of the MST
        (figureX, figureY) = (index // columns, index % columns)

        axs[figureX][figureY].axis('on')
        sns.lineplot(x=rho_values, y=[sealfon - real for sealfon, real in zip(sealfon, real)], marker='o',
                     label="Sealfon",
                     ax=axs[figureX][figureY])
        sns.lineplot(x=rho_values, y=[ours - real for ours, real in zip(our, real)], marker='o',
                     label="$\\textbf{Our}$",
                     ax=axs[figureX][figureY])
        sns.lineplot(x=rho_values, y=[pamst - real for pamst, real in zip(results['pamst'], real)], marker='o',
                     label="PAMST", ax=axs[figureX][figureY])
        title = "G({}, {})".format(meta_params['graph-size'], round(meta_params['edge_probabilities'][index], 1))
        axs[figureX][figureY].set_title(title)
        axs[figureX][figureY].set_xlabel("$\\rho$")
        axs[figureX][figureY].set_ylabel("MST Error")
        axs[figureX][figureY].set_yscale("log")
        # axs[0].set_ylim([0,100])
        # axs[index].set_ylabel("Additive Error")

    plt.subplots_adjust(hspace = 0.4)
    return (fig, axs)


def convert_results(results):
    return ([i * -1 for i in results['sealfon']],
            [i * -1 for i in results['our']],
            [i * -1 for i in results['pamst']],
            [i * -1 for i in results['real']])
