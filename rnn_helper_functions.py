import copy
import random
import torch
from rnn_data_preprocessing import sentence_to_tensor


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
        sentence = self.data[category].pop()
        category_tensor = torch.tensor([self.categories.index(category)],
                                       dtype=torch.long)
        sentence_tensor = sentence_to_tensor(sentence, self.n_tags, self.tags)
        return category, sentence, category_tensor, sentence_tensor
