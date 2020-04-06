import pandas as pd

from n_gram_model import process_training_data, pre_process_test, test_ngram

def pre_process_data(filename):
  df = pd.read_csv(filename)
  df = df[['student_id','language', 'error_type',
           'correct_trigram_poss', 'incorrect_trigram_poss',
           'correct_sentence', 'incorrect_sentence']]
  syntactic_errors_df = df[df['correct_trigram_poss'] != df['incorrect_trigram_poss']]
  es_syntactic_errors_df = syntactic_errors_df[syntactic_errors_df['language'] == 'Spanish']
  return es_syntactic_errors_df

def test(train_dataset, method, test_df):
  methods = {'unsmoothed': [1, 'unsmoothed'], 'laplace': [3, 'add-one'], 'interpolation': [3, 'interpolation']}
  n = methods[method][0]
  langs = process_training_data(train_dataset, method, n)
  test_df = test_df.head()
  for index, row in test_df.iterrows():
    print(row['error_type'], row['incorrect_sentence'], row['incorrect_trigram_poss'], row['correct_trigram_poss'])
    for l in langs.keys():
      processed_ngram = pre_process_test(row['incorrect_trigram_poss'].split(), langs[l][1])
      probability = 0
      if len(processed_ngram) > n:
        for i in range(0, len(processed_ngram)):
          probability += test_ngram(method, n, processed_ngram[i:i+n], langs[l])
      else:
        probability += test_ngram(method, n, processed_ngram, langs[l])
      print(l, probability)

df = pre_process_data('main_parser.csv')
test('tagged_sentences.csv', 'laplace', df)