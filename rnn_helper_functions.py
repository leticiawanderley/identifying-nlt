import copy
import random
import pandas as pd
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


def setup_testing_data(dataset_file, categories):
    data_dict = {}
    test_data = pd.read_csv(dataset_file)
    data_dict['zhs_ud'] = \
        test_data[test_data['Negative transfer?'] == True]\
        ['incorrect_trigram_ud'].to_list()
    data_dict['en_ud'] = \
        test_data[test_data['Negative transfer?'] == False]\
        ['incorrect_trigram_ud'].to_list()
    for cat in categories:
        for i in range(len(data_dict[cat])):
            data_dict[cat][i] = data_dict[cat][i].split()
    return data_dict
