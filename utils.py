def split_sentences(dataset):
    """Create list of tags from each of the datasets' rows
    appending two end of sentence markers to the end of the lists."""
    for i in range(len(dataset)):
        dataset[i] = dataset[i].split() + ['_', '_']
    return dataset


def extract_vocabs(dataset, n):
    """For each gram from unigram to n-gram, create a dictionary in which
    the keys are the n-grams that exist on the dataset
    and the values are their occurence counts."""
    vocabs = {}
    for length in range(n):
        vocabs[length] = {}
    dataset_size = 0
    for sent in dataset:
        dataset_size += len(sent)
        for i in range(len(sent)):
            for j in range(n):
                right_index = i + j + 1
                if right_index <= len(sent):
                    key = ' '.join(sent[i:right_index])
                    if key not in vocabs[j].keys():
                        vocabs[j][key] = 0
                    vocabs[j][key] += 1
    return vocabs, dataset_size


def get_count(tags, vocabs):
    """Retrive number of occurences of tag sequence
    from the pre-processed n-gram vocabularies."""
    vocab = vocabs[len(tags) - 1]
    tags_key = ' '.join(tags)
    return vocab[tags_key] if tags_key in vocab.keys() else 0
