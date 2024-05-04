import matplotlib.pylab as plt
import numpy as np
import seaborn as sns


def draw_diagram(input_file_path, output_file_path):
    plt.clf()

    memory = np.load(input_file_path)

    result = memory["result"]

    mask = np.zeros_like(result)
    mask[np.triu_indices_from(mask, k=1)] = True
    mask = np.flip(mask, axis=0)

    # rescale scores to 0-1 for each cut_layers value
    masked_result = np.ma.masked_array(result, mask)
    max_dist = np.ma.max(masked_result, axis=1)[:, np.newaxis]
    min_dist = np.ma.min(masked_result, axis=1)[:, np.newaxis]

    result = (result - min_dist) / (max_dist - min_dist)

    ax = sns.heatmap(result, linewidth=0.5, mask=mask, cmap="viridis_r")
    ax.invert_yaxis()
    ax.set_xticklabels(ax.get_xticklabels(), ha="left")
    ax.set_yticklabels(ax.get_yticklabels(), va="bottom")

    plt.savefig(output_file_path)
