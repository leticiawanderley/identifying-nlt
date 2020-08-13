import copy
import random
import torch

from rnn_data_preprocessing import sequence_to_tensor


def category_from_output(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


class Data:
    def __init__(self, data, tags):
        self.tags = tags
        self.n_tags = len(self.tags)
        self.data = copy.deepcopy(data)
        self.categories = list(data.keys())
        size = 0
        for category in self.categories:
            random.shuffle(self.data[category])
            size += len(self.data[category])
        self.size = size

    def random_training_datapoint(self):
        category = random_choice(self.categories)
        while not self.data[category]:
            category = random_choice(self.categories)
        sequence = self.data[category].pop()
        category_tensor = torch.tensor([self.categories.index(category)],
                                       dtype=torch.long)
        sequence_tensor = sequence_to_tensor(sequence, self.n_tags, self.tags)
        return category, sequence, category_tensor, sequence_tensor


def setup_data(dataset, columns, feature, gold_label):
    categories = columns.keys()
    data_dict = {}
    for col in categories:
        data_dict[columns[col]] = \
            dataset[dataset[gold_label] == col][feature].to_list()
        for i in range(len(data_dict[columns[col]])):
            data_dict[columns[col]][i] = data_dict[columns[col]][i].split()
    return data_dict
