import random
import pandas as pd
import torch

from utils import split_sentences


def read_data(filenames, columns):
    data = {}
    for column in columns:
        data[column] = []
    for filename in filenames:
        df = pd.read_csv(filename)
        for c in columns:
            data[c] += split_sentences(df[c].to_list())
    return data


def split_data(data_dict, split):
    train = {}
    test = {}
    for key in data_dict:
        ind = int(len(data_dict[key]) * split)
        random.shuffle(data_dict[key])
        train[key] = data_dict[key][ind:]
        test[key] = data_dict[key][:ind]
    return train, test


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


def sequence_to_tensor(sequence, n_tags, all_tags, max_length):
    tensor = torch.zeros(max_length, 1, n_tags)
    for li, tag in enumerate(sequence):
        tensor[li][0][tag_to_index(tag, all_tags)] = 1
    return tensor


def category_from_output(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


class Dataset:
    def __init__(self, data_dict, tags, setup):
        self.tags = tags
        self.setup = setup
        self.categories = ['en_ud', 'zhs_ud']
        x = data_dict['en_ud'] + data_dict['zhs_ud']
        y = ['en_ud'] * len(data_dict['en_ud']) + \
            ['zhs_ud'] * len(data_dict['zhs_ud'])
        self.x_lengths = [len(s) for s in x]
        max_length = max(self.x_lengths)
        self.x_train = [sequence_to_tensor(s, len(tags), tags,
                        max_length) for s in x]
        self.y_train = [self.compute_category_tensor(c) for c in y]

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx], self.x_lengths[idx]

    def compute_category_tensor(self, category):
        if self.setup == 'NLLoss':
            category_tensor = torch.tensor([self.categories.index(category)],
                                           dtype=torch.long)
        else:
            category_tensor = torch.zeros(1, len(self.categories))
            category_tensor[0][self.categories.index(category)] = 1
        return category_tensor
