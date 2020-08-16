import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from constant import CONFUSION_MATRIX_AXES
from utils import create_confusion_data


def losses(all_losses, output_filename):
    """Plot model losses and save figure in a file."""
    plt.figure()
    plt.plot(all_losses)
    plt.savefig('figures/' + output_filename)


def confusion_matrix(dataset_file, gold_label, guess_column, output_filename):
    """Plot confusion matrix and save figure in a file."""
    confusion = create_confusion_data(dataset_file, gold_label, guess_column)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion)
    fig.colorbar(cax)

    for (i, j), z in np.ndenumerate(confusion):
        ax.text(j, i, '{:0.4f}'.format(z), ha='center', va='center')

    ax.set_xticklabels([''] + CONFUSION_MATRIX_AXES)
    ax.set_yticklabels([''] + CONFUSION_MATRIX_AXES)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.xlabel('Predicted')
    plt.ylabel('True')

    fig.savefig('figures/' + output_filename)
