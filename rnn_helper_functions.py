import random
import torch
from rnn_data_preprocessing import sentence_to_tensor


def category_from_output(output, all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


def random_choice(l):
    return l[random.randint(0, len(l) - 1)]


def random_training_example(all_categories, data, n_tags, all_tags):
    category = random_choice(all_categories)
    sentence = random_choice(data[category])
    category_tensor = torch.tensor([all_categories.index(category)],
                                   dtype=torch.long)
    sentence_tensor = sentence_to_tensor(sentence, n_tags, all_tags)
    return category, sentence, category_tensor, sentence_tensor
