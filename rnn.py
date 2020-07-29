import pandas as pd
import torch

def read_data(filenames, columns):
    data = {}
    for column in columns:
        data[column] = []
    for filename in filenames:
        df = pd.read_csv(filename)
        for c in columns:
            data[c].append(df[c].to_list())
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


def sentence_to_tensor(sentence, n_tags, all_tags):
    sentence = sentence.split()
    tensor = torch.zeros(len(sentence), 1, n_tags)
    for li, tag in enumerate(sentence):
        tensor[li][0][tag_to_index(tag, all_tags)] = 1
    return tensor


# read_data(['data/training data/tagged_sentences_dataset_sentences.csv',
#           'data/training data/tagged_sentences_1000sents.csv'],
#          ['en_ud', 'es_ud'])

all_tags = get_all_tags(['data/training data/es_ud_0_vocab.csv',
                         'data/training data/en_ud_0_vocab.csv'])


n_tags = len(all_tags)

print(tag_to_tensor('VERB', n_tags, all_tags))

print(sentence_to_tensor('DET NOUN VERB ADP NUM PROPN ', n_tags, all_tags).size())