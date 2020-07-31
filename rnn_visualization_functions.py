import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def losses(all_losses, output_filename):
    plt.figure()
    plt.plot(all_losses)
    plt.savefig(output_filename)


def confusion_matrix(confusion, all_categories, output_filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.savefig(output_filename)
