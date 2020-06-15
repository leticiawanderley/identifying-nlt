import csv
import pandas as pd

from constant import CCONJ, CONJ, SPANISH


def split_sentences(dataset):
    """Create list of tags from each of the datasets' rows
    appending two end of sentence markers to the end of the lists."""
    clean_dataset = []
    for i in range(len(dataset)):
        if type(dataset[i]) == str:
            clean_dataset.append(dataset[i].split() + ['_', '_'])
    return clean_dataset


def extract_vocabs(dataset, n):
    """For each gram from unigram to n-gram, create a dictionary in which
    the keys are the n-grams that exist on the dataset
    and the values are their occurence counts."""
    vocabs = {}
    for length in range(n):
        vocabs[length] = {}
    for sent in dataset:
        for i in range(len(sent)):
            for j in range(n):
                right_index = i + j + 1
                if right_index <= len(sent):
                    key = ' '.join(sent[i:right_index])
                    if key not in vocabs[j].keys():
                        vocabs[j][key] = 0
                    vocabs[j][key] += 1
    return vocabs


def get_count(tags, vocabs):
    """Retrive number of occurences of tag sequence
    from the pre-processed n-gram vocabularies."""
    vocab = vocabs[len(tags) - 1]
    tags_key = ' '.join(tags)
    return vocab[tags_key] if tags_key in vocab.keys() else 0


def tags_mapping(filename):
    """Read csv tags mapping and convert it to a dictionary."""
    mapping_dict = {}
    with open(filename, newline='') as csvfile:
        mapping = csv.reader(csvfile, delimiter=',')
        for row in mapping:
            mapping_dict[row[0]] = row[1]
    return mapping_dict


def unpack_poss_and_tags(series):
    """Unpack series of parsed pos and tags into two lists."""
    poss = []
    tags = []
    for row in series.to_list():
        poss.append(row[0])
        tags.append(row[1])
    return poss, tags


def process_tags(input_filename, output_filename):
    """Create csv file from txt file in which the values are
    separated by commas."""
    f = open(input_filename, "r")
    lines = f.readlines()
    with open(output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for line in lines:
            tags = line.split(',')
            for tag in tags:
                csvwriter.writerow([tag.strip()])


def save_vocabs_to_csv(vocabs, lang, affix):
    """Save ngram counts to a csv file."""
    for n in vocabs.keys():
        df = pd.DataFrame(vocabs[n].items(), columns=['ngram', 'count'])
        df.to_csv(lang + '_' + affix + '_' + str(n) + '_vocab.csv')


def tag_sentences(model, sentence, language=None, mapping=None):
    """Part-of-speeh tag dataframe sentence."""
    poss = ''
    tags = ''
    if type(sentence) == str:
        doc = model(sentence)
        for token in doc:
            pos = token.pos_
            if token.pos_ == CONJ:
                pos = CCONJ if language and language == SPANISH else pos
            poss += pos + ' '
            tag = token.tag_
            if language == SPANISH and mapping:
                tag = mapping[tag]
            tags += tag + ' '
    return (poss, tags)
