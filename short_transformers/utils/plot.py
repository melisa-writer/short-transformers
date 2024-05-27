import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import matplotlib

def draw_diagram(results, output_file_path, title=None, normalized=True):
    plt.clf()

    mask = np.zeros_like(results)
    mask[np.triu_indices_from(mask, k=1)] = True
    mask = np.flip(mask, axis=0)

    # @TODO make row-wise normalization optional
    # rescale scores to 0-1 for each cut_layers value
    
    if normalized:
        masked_results = np.ma.masked_array(results, mask)
        max_dist = np.ma.max(masked_results, axis=1)[:, np.newaxis]
        min_dist = np.ma.min(masked_results, axis=1)[:, np.newaxis]

        results = (results - min_dist) / (max_dist - min_dist)

    ax = sns.heatmap(results, linewidth=0.5, mask=mask, cmap="viridis_r")
    ax.invert_yaxis()
    ax.set_xticklabels(ax.get_xticklabels(), ha="left")
    ax.set_yticklabels(ax.get_yticklabels(), va="bottom")

    plt.xlabel("Layer number, l")
    plt.ylabel("Block size, n")

    if title:
        ax.set_title(title)

    plt.savefig(output_file_path)


def draw_layers_heatmap(results, metric_name, title, output_path, block_size=0, cmap=matplotlib.cm.viridis_r):

    # @TODO make row-wise normalization?
    if block_size:
        data = results[0, :-block_size]
    else:
        data = results[0, :]
    data = np.asarray(data).reshape(data.shape[0],1)

    fig = plt.figure(figsize = (5,5))
    ax = sns.heatmap(data, annot=False, cmap=cmap)

    ax.set_xlabel(metric_name)
    ax.set_title(title)
    plt.savefig(output_path)