import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from utils import create_confusion_data


def losses(all_losses, output_filename):
    plt.figure()
    plt.plot(all_losses)
    plt.savefig(output_filename)


def confusion_matrix(dataset_file, gold_label, guess_column,
                     all_categories, output_filename):
    confusion = create_confusion_data(dataset_file, gold_label, guess_column)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion)
    fig.colorbar(cax)

    for (i, j), z in np.ndenumerate(confusion):
        ax.text(j, i, '{:0.4f}'.format(z), ha='center', va='center')

    ax.set_xticklabels([''] + all_categories)
    ax.set_yticklabels([''] + all_categories)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.xlabel('Predicted')
    plt.ylabel('True')

    fig.savefig(output_filename)
