import copy
import random
import pandas as pd
import torch

from utils import split_sentences


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


def read_data(filenames, columns):
    data = {}
    for column in columns:
        data[column] = []
    for filename in filenames:
        df = pd.read_csv(filename)
        for c in columns:
            data[c] += split_sentences(df[c].to_list())
    return data


def get_all_tags(languages_tag_files):
    all_tags = {}
    for tag_file in languages_tag_files:
        df = pd.read_csv(tag_file, index_col=0)
        for tag in df['ngram'].to_list():
            if tag not in all_tags:
                all_tags[tag] = True
    return sorted(all_tags.keys())


def tag_to_index(tag, all_tags):
    return all_tags.index(tag)


def tag_to_tensor(tag, n_tags, all_tags):
    tensor = torch.zeros(1, n_tags)
    tensor[0][tag_to_index(tag, all_tags)] = 1
    return tensor


def sequence_to_tensor(sequence, n_tags, all_tags):
    tensor = torch.zeros(len(sequence), 1, n_tags)
    for li, tag in enumerate(sequence):
        tensor[li][0][tag_to_index(tag, all_tags)] = 1
    return tensor


class Data:
    def __init__(self, data, tags, setup):
        self.tags = tags
        self.n_tags = len(self.tags)
        self.data = copy.deepcopy(data)
        self.categories = list(data.keys())
        self.setup = setup
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
        category_tensor = self.compute_category_tensor(category)
        sequence_tensor = sequence_to_tensor(sequence, self.n_tags, self.tags)
        return category, sequence, category_tensor, sequence_tensor

    def compute_category_tensor(self, category):
        if self.setup == 'NLLoss':
            category_tensor = torch.tensor([self.categories.index(category)],
                                           dtype=torch.long)
        else:
            category_tensor = torch.zeros(1, len(self.categories))
            category_tensor[0][self.categories.index(category)] = 1
        return category_tensor


def setup_data(dataset, columns, feature, ground_truth):
    categories = columns.keys()
    data_dict = {}
    for col in categories:
        data_dict[columns[col]] = \
            dataset[dataset[ground_truth] == col][feature].to_list()
        for i in range(len(data_dict[columns[col]])):
            data_dict[columns[col]][i] = data_dict[columns[col]][i].split()
    return data_dict


def category_from_output(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i
