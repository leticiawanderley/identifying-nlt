import argparse
import math
import pandas as pd

from constant import INTERPOLATION, LAPLACE, UNSMOOTHED, OOV_TAG,\
                     NGRAM_METHODS
from utils import extract_vocabs, get_count, split_sentences


def pre_process_data(filenames, languages):
    """Format training data

    Concatenate values with a pipe separators between sentences.
    """
    datasets = {}
    for filename in filenames:
        df = pd.read_csv(filename)
        for language in languages:
            if language not in datasets.keys():
                datasets[language] = []
            datasets[language] += split_sentences(df[language + '_tags'].
                                                  tolist())
    return datasets


def extract_vocabulary(tag_counts, dataset_size):
    """Extract pos tag alphabet.

    Removing less frequente tags (frequency < 0.005%.)"""
    vocabulary = []
    tag_set = tag_counts.keys()
    threshold = 0.00005 * dataset_size
    for tag in tag_set:
        if tag_counts[tag] > threshold:
            vocabulary.append(tag)
    return vocabulary


def replace_oov_train(vocabs, vocabulary):
    """Replace out of vocabulary tags with a special symbol."""
    updated_vocabs = {}
    for n in vocabs.keys():
        vocab = {}
        for n_gram in vocabs[n].keys():
            updated_key = []
            for tag in n_gram.split():
                if tag not in vocabulary:
                    updated_key.append(OOV_TAG)
                else:
                    updated_key.append(tag)
            updated_key = ' '.join(updated_key)
            if updated_key not in vocab.keys():
                vocab[updated_key] = 0
            vocab[updated_key] += vocabs[n][n_gram]
        updated_vocabs[n] = vocab
    return updated_vocabs


def replace_oov_test(sentence, vocabulary):
    """Replace out of vocabulary tags with a special symbol."""
    for i in range(len(sentence)):
        if sentence[i] not in vocabulary:
            sentence[i] = OOV_TAG
    return sentence


def unsmoothed(n, tags, vocabs):
    """Compute n-gram probability without smoothing."""
    gram_count = get_count(tags, vocabs)
    previous_count = (len(vocabs[n-1]) if n == 1
                      else get_count(tags[:-1], vocabs))
    return math.log(gram_count/previous_count)


def laplace(n, tags, vocabs, vocabulary_size):
    """Compute n-gram probability with add-one smoothing."""
    gram_count = get_count(tags, vocabs) + 1
    previous_count = (len(vocabs[n-1]) if n == 1
                      else get_count(tags[:-1], vocabs)) + vocabulary_size
    return math.log(gram_count/previous_count)


def deleted_interpolation(n, vocabs, dataset_size):
    """Compute interpolation weights using deleted interpolation."""
    gamas = [0.0]*n
    for n_gram in vocabs[n-1].keys():
        best_count = 0
        best_gama = 0
        best_p = 0
        for i in range(n - 1, -1, -1):
            n_gram_list = n_gram.split()
            numerator = get_count(n_gram_list[:i + 1], vocabs) - 1
            denominator = (dataset_size if i == 0
                           else get_count(n_gram_list[:i], vocabs)) - 1
            p = numerator/denominator if denominator > 0 else 0
            if p > best_p:
                best_p = p
                best_count = numerator + 1
                best_gama = i
        gamas[best_gama] += best_count
    gama_sum = sum(gamas)
    return [g/gama_sum for g in gamas]


def interpolation(n, tags, vocabs, gamas, dataset_size):
    """Compute n-gram probability with interpolation smoothing."""
    prob = 0
    for i in range(n):
        numerator = get_count(tags[:n-i], vocabs)
        denominator = (dataset_size if i == (n - 1)
                       else get_count(tags[:n-i-1], vocabs))
        prob += (gamas[i] * (numerator/denominator)) if denominator > 0 else 0
    return math.log(prob)


def pre_process_training_data(dataset, n):
    """Pre-process training dataset."""
    vocabs, dataset_size = extract_vocabs(dataset, n)
    vocabulary = extract_vocabulary(vocabs[0], dataset_size)
    vocabs = replace_oov_train(vocabs, vocabulary)
    return vocabulary, vocabs, dataset_size


def pre_process_test(ngram, vocabulary):
    """Pre-process test document."""
    return replace_oov_test(ngram, vocabulary)


def process_training_data(datasets_filenames, method, n, languages):
    """Process training data depending on which smoothing method was chosen."""
    datasets = pre_process_data(datasets_filenames, languages)
    langs = {}
    for lang in datasets.keys():
        dataset = datasets[lang]
        vocabulary, vocabs, dataset_size = pre_process_training_data(dataset,
                                                                     n)
        if method == INTERPOLATION:
            langs[lang] = [vocabs, vocabulary,
                           deleted_interpolation(n, vocabs, dataset_size),
                           dataset_size]
        else:
            langs[lang] = vocabs, vocabulary
    return langs


def test_ngram(method, n, ngram, language_model):
    """Compute probability on n-gram."""
    if method == UNSMOOTHED:
        prob = unsmoothed(n, ngram, language_model[0])
    elif method == LAPLACE:
        prob = laplace(n, ngram, language_model[0], len(language_model[1]))
    else:
        prob = interpolation(n, ngram, language_model[0],
                             language_model[2], language_model[3])
    return prob


def main(method, dataset_filenames, test_ngrams, languages):
    n = NGRAM_METHODS[method][0]
    langs = process_training_data(dataset_filenames, method, n, languages)
    for ngram in test_ngrams:
        for l in langs.keys():
            processed_ngram = pre_process_test(ngram, langs[l][1])
            probability = 0
            if len(processed_ngram) > n:
                for i in range(0, len(processed_ngram)):
                    probability += test_ngram(method, n,
                                              processed_ngram[i:i+n], langs[l])
            else:
                probability += test_ngram(method, n, processed_ngram, langs[l])
            print(l, probability)


def parse_arg_list():
    """Uses argparse to parse the required parameters"""
    parser = argparse.ArgumentParser(
                description='',
                formatter_class=argparse.RawTextHelpFormatter)
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument(
        '-m', '--method',
        help='Smoothing method. One of unsmoothed, laplace, or interpolation',
        required=True)
    required_args.add_argument(
        '-d', '--dataset_file',
        help='name of the file which contains the training data',
        required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg_list()
    test_ngrams = [['DET', 'NOUN', 'VERB'],
                   ['PUNCT', 'VERB', 'DET'],
                   ['|', 'VERB', 'DET']]
    languages = ['en', 'es']
    main(args.method, [args.dataset_file], test_ngram, languages)
