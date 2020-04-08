import pandas as pd

from n_gram_model import process_training_data, pre_process_test, test_ngram

def pre_process_data(filename):
  df = pd.read_csv(filename)
  df = df[['student_id','language', 'error_type',
           'correct_trigram_tags', 'incorrect_trigram_tags',
           'correct_trigram_poss', 'incorrect_trigram_poss',
           'correct_trigram', 'incorrect_trigram',
           'correct_sentence', 'incorrect_sentence']]
  syntactic_errors_df = df[df['correct_trigram_tags'] != df['incorrect_trigram_tags']]
  es_syntactic_errors_df = syntactic_errors_df[syntactic_errors_df['language'] == 'Spanish']
  return es_syntactic_errors_df

def test(train_dataset, method, test_df):
  data_dict = {
    'student_id': [], 'error_type': [],
    'en': [], 'es': [],
    'correct_trigram_poss': [], 'incorrect_trigram_poss': [],
    'correct_trigram': [], 'incorrect_trigram': [],
    'correct_sentence': [], 'incorrect_sentence': []
  }
  methods = {'unsmoothed': [1, 'unsmoothed'], 'laplace': [3, 'add-one'], 'interpolation': [3, 'interpolation']}
  n = methods[method][0]
  langs = process_training_data(train_dataset, method, n)
  for index, row in test_df.iterrows():
    data_dict['student_id'].append(row['student_id'])
    data_dict['error_type'].append(row['error_type'])
    data_dict['correct_trigram_poss'].append(row['correct_trigram_poss'])
    data_dict['incorrect_trigram_poss'].append(row['incorrect_trigram_poss'])
    data_dict['correct_trigram'].append(row['correct_trigram'])
    data_dict['incorrect_trigram'].append(row['incorrect_trigram'])
    data_dict['correct_sentence'].append(row['correct_sentence'])
    data_dict['incorrect_sentence'].append(row['incorrect_sentence'])
    for l in langs.keys():
      processed_ngram = pre_process_test(row['incorrect_trigram_poss'].split(), langs[l][1])
      probability = 0
      if len(processed_ngram) > n:
        for i in range(0, len(processed_ngram)):
          probability += test_ngram(method, n, processed_ngram[i:i+n], langs[l])
      else:
        probability += test_ngram(method, n, processed_ngram, langs[l])
      data_dict[l].append(probability)
  df = pd.DataFrame.from_dict(data_dict)
  df.to_csv('data/results.csv')

df = pre_process_data('data/main_parser.csv')
test('data/tagged_sentences.csv', 'interpolation', df)