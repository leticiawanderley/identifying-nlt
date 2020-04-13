import argparse
import math
import pandas as pd

from utils import sublist_count

def pre_process_data(filename):
  """Format training data

  Concatenate values with a pipe separators between sentences.
  """
  df = pd.read_csv(filename)
  datasets = {}
  datasets['en'] = df.en_pos.str.cat(sep='| ')
  datasets['es'] = df.es_pos.str.cat(sep='| ')
  return datasets

def extract_vocabulary(dataset):
  """Extract pos tag alphabet.

  Removing less frequente tags (frequency < 0.5%.)"""
  vocabulary = []
  tag_set = set(dataset)
  threshold = 0.005*len(dataset)
  for tag in tag_set:
    if dataset.count(tag) > threshold:
      vocabulary.append(tag)
  return vocabulary

def replace_oov(dataset, vocabulary):
  """Replace out of vocabulary tags with a special symbol (#)."""
  for i in range(len(dataset)):
    if dataset[i] not in vocabulary:
      dataset[i] = '#'
  return dataset

def unsmoothed(n, tags, dataset):
  """Compute n-gram probability without smoothing."""
  gram_count = sublist_count(tags, dataset)
  previous_count = len(dataset) if n == 1 else sublist_count(tags[:-1], dataset)
  return math.log(gram_count/previous_count)

def laplace(n, tags, dataset, vocabulary_size):
  """Compute n-gram probability with add-one smoothing."""
  gram_count = sublist_count(tags, dataset) + 1
  previous_count = (sublist_count(tags[:-1], dataset) + vocabulary_size)
  return math.log(gram_count/previous_count)

def deleted_interpolation(n, dataset):
  """Compute interpolation weights using deleted interpolation."""
  gamas = [0.0]*n
  for i in range(0, len(dataset) - n):
    best_count = 0
    best_gama = 0
    best_p = 0
    for j in range(n):
      candidate_count = sublist_count(dataset[i:i+(n-j)], dataset)
      candidate_p = 0
      if sublist_count(dataset[i:i+(n-j-1)], dataset) > 1:
        candidate_p = (candidate_count - 1)/(sublist_count(dataset[i:i+(n-j-1)], dataset) - 1)
      if candidate_p > best_p:
        best_count = candidate_count
        best_gama = j
        best_p = candidate_p
    gamas[best_gama] += best_count
  gama_sum = sum(gamas)
  return [g/gama_sum for g in gamas]

def interpolation(n, tags, dataset, gamas):
  """Compute n-gram probability with interpolation smoothing."""
  prob = 0
  for i in range(n):
    if sublist_count(tags[:n-i-1], dataset) > 0:
      prob += gamas[i] * (sublist_count(tags[:n-i], dataset)/sublist_count(tags[:n-i-1], dataset))
  return math.log(prob)

def pre_process_training_data(dataset):
  """Pre-process training dataset."""
  vocabulary = extract_vocabulary(dataset)
  return replace_oov(dataset, vocabulary), vocabulary

def pre_process_test(ngram, vocabulary):
  """Pre-process test document."""
  return replace_oov(ngram, vocabulary)

def process_training_data(datasets_filename, method, n):
  """Process training data depending on which smoothing method was chosen."""
  datasets = pre_process_data(datasets_filename)
  langs = {}
  for lang in datasets.keys():
    dataset = datasets[lang].split()
    dataset, vocabulary = pre_process_training_data(dataset)
    if method == 'interpolation':
      langs[lang] = [dataset, vocabulary, deleted_interpolation(n, dataset)]
    else:
      langs[lang] = dataset, vocabulary
  return langs

def compute_perplexity(doc_size, prob):
  """Compute perplexity based on a log probability."""
  return math.exp(-(prob/doc_size))

def test_ngram(method, n, ngram, language_model):
  """Compute probability on n-gram."""
  if method == 'unsmoothed':
    prob = unsmoothed(n, ngram, language_model[0])
  elif method == 'laplace':
    prob = laplace(n, ngram, language_model[0], len(language_model[1]))
  else:
    prob = interpolation(n, ngram, language_model[0], language_model[2])
  return prob

def main(method, dataset_filename, test_ngrams):
  methods = {'unsmoothed': [1, 'unsmoothed'], 'laplace': [3, 'add-one'], 'interpolation': [3, 'interpolation']}
  n = methods[method][0]
  langs = process_training_data(dataset_filename, method, n)
  for ngram in test_ngrams:
    for l in langs.keys():
      processed_ngram = pre_process_test(ngram, langs[l][1])
      probability = 0
      if len(processed_ngram) > n:
        for i in range(0, len(processed_ngram)):
          probability += test_ngram(method, n, processed_ngram[i:i+n], langs[l])
      else:
        probability += test_ngram(method, n, processed_ngram, langs[l])
      print(l, probability)

def parse_arg_list():
    """Uses argparse to parse the required parameters"""
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.RawTextHelpFormatter)
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('-m', '--method', help='Smoothing method. One of unsmoothed, laplace, or interpolation', required=True)
    required_args.add_argument('-d', '--dataset_file', help='name of the file which contains the training data', required=True)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
  args = parse_arg_list()
  test_ngrams = [['DET', 'NOUN', 'VERB'], ['PUNCT', 'VERB', 'DET'], ['|', 'VERB', 'DET']]
  main(args.method, args.dataset_file, test_ngram)