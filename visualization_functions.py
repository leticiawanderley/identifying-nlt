import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random

from constant import CONFUSION_MATRIX_AXES
from utils import create_confusion_data, process_scatter_plot_data


def losses(all_losses, info):
    """Plot model losses and save figure in a file."""
    plt.figure()
    plt.plot(all_losses)
    plt.savefig('figures/all_losses_' + info + '.png')


def confusion_matrix(dataset_file, ground_truth_column,
                     guess_column, info):
    """Plot confusion matrix and save figure in a file.
    """
    confusion = create_confusion_data(dataset_file, ground_truth_column,
                                      guess_column)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion)
    fig.colorbar(cax)

    for (i, j), z in np.ndenumerate(confusion):
        ax.text(j, i, '{:0.4f}'.format(z), ha='center', va='center')

    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))

    ax.set_xticklabels(CONFUSION_MATRIX_AXES)
    ax.set_yticklabels(CONFUSION_MATRIX_AXES)

    plt.xlabel('Predicted')
    plt.ylabel('True')

    fig.savefig('figures/confusion_matrix_' + info + '.png')


def scatter_plot(dataset_file, count_column, ground_truth_column,
                 results_column, info):
    df = process_scatter_plot_data(dataset_file, results_column)
    ground_truth_values = [True, False]
    choices = [0.1, 0.001, 0, -0.001, -0.1]
    colours = ['blue', 'yellow']
    fig, ax = plt.subplots()
    for index, ground_truth in enumerate(ground_truth_values):
        data = df[df[ground_truth_column] == ground_truth]
        x = data[count_column] + random.choice(choices)
        y = data[results_column] + random.choice(choices)
        ax.scatter(x, y, c=colours[index], label=ground_truth,
                   alpha=0.3, edgecolors='none')

    ax.legend()
    ax.grid(True)

    plt.show()
    fig.savefig('figures/scatter_plot' + info + '.png')
