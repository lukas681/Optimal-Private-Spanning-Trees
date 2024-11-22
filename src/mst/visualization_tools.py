import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def prepare_data_exp_three(results, rho_values):
    """
    Highly specialst utility function.
    :param results:
    :rho_values: The range of rho values
    :return:
    """

    mi_results = []
    for ix, p in enumerate(rho_values): # Try to inline this!
        for i in range(len(results)):
            for k in results[i].keys():
                mi_results += [(dict(rho=p, alg=k, value=results[i][k][ix]))]
    return pd.DataFrame(list(mi_results))

def init_plot_exp_three(df, meta_params):
    """
    Visualizes the experiment with the
    :param df:
    :param meta_params:
    :return:
    """

    plt.figure(figsize=(12, 8))
    aggregated = df.groupby(['rho', 'alg'])['value'].agg(['median', 'min', 'max']).reset_index()
    for alg_name, group in aggregated.groupby('alg'):
        plt.plot(group['rho'], group['median'], label=f'{alg_name}', linewidth=1.5, marker="o", linestyle='-')  # Dashed line for distinctiveness
        plt.fill_between(group['rho'], group['min'], group['max'], alpha=0.2)  # Matching fill color

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'Mutual Information Graph with  \nwith $\\Delta_\\infty = {meta_params["sensitivity"]}$')

#    desired_ticks = [minx, 10**-1, 10**0]
    #    plt.xscale('log')
    ax = plt.gca()
#    ax.set_xscale('log')
#    ax.set_xticks(desired_ticks)
    plt.xlabel("$\\rho$")
    plt.ylabel("Weight of MST of negated graph")
    plt.legend()
    plt.grid(True)

#@warnings.deprecated
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
    sns.lineplot(x=rho_values, y=[ours - real for ours, real in zip(our, real)], marker='o', label="$\\textbf{Our}$",            ax=ax)
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
    # axs[figureX][figureY].set_yscale("log")
    # axs[0].set_ylim([0,100])
    # axs[index].set_ylabel("Additive Error")

    plt.subplots_adjust(hspace = 0.4)
    return (fig, axs)

def normalize(df):
    """
    Very specific function. Adds a normlazed value to df.
    Normalized sealfon, our and pamst to divide it by the real weight.
    This is very specific to our use case.
    :param df:
    :return:
    """
    # Extract the 'real' values for each 'p'
    real_values = df[df['type'] == 'real'][['p', 'value']].rename(columns={'value': 'real_value'})
    df_intermediate = df.merge(real_values, on='p')
    df_intermediate['normalized_value'] = df_intermediate['value'] / df_intermediate['real_value']
    return df_intermediate


def init_plot_densities(df, meta_params, minx):
    """
    Plotting the effect of the density
    :param aggregated_data:
    :param meta_params:
    :return:
    """
    n, max_edge_weight = meta_params['n'], meta_params['max_edge_weight']

    plt.figure(figsize=(12, 8))
    df = normalize(df)
    aggregated = df.groupby(['p', 'type'])['normalized_value'].agg(['median', 'min', 'max']).reset_index()

    for type_name, group in aggregated.groupby('type'):
        plt.plot(group['p'], group['median'], label=f'{type_name}', linewidth=1.5, marker="o", linestyle='-')  # Dashed line for distinctiveness
        plt.fill_between(group['p'], group['min'], group['max'], alpha=0.2)  # Matching fill color
    # Add gridlines for readability
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'$G({n}, p)$ where $w_e \\sim U(0, {max_edge_weight})$ \nwith $\\Delta_\\infty = {meta_params["sensitivity"]}$ and $\\rho = {meta_params["rho"]}$')

    desired_ticks = [minx, 10**-1, 10**0]
    #    plt.xscale('log')
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_xticks(desired_ticks)

    plt.xlabel("density $p$")
    plt.ylabel("Normalized Weights")
    plt.legend()
    plt.grid(True)

def convert_results(results):
    return ([i * -1 for i in results['sealfon']],
            [i * -1 for i in results['our']],
            [i * -1 for i in results['pamst']],
            [i * -1 for i in results['real']])
