import pandas as pd

from n_gram_model import process_training_data, pre_process_test, test_ngram

def pre_process_data(filename, language):
  df = pd.read_csv(filename)
  df = df[['student_id','language', 'error_type',
           'correct_trigram_tags', 'incorrect_trigram_tags',
           'correct_trigram_poss', 'incorrect_trigram_poss',
           'correct_trigram', 'incorrect_trigram',
           'correct_sentence', 'incorrect_sentence']]
  es_errors_df = df[df['language'] == language]
  return es_errors_df

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
  fields = ['student_id', 'error_type',
            'correct_trigram_poss', 'incorrect_trigram_poss'
            'correct_trigram', 'incorrect_trigram',
            'correct_sentence', 'incorrect_sentence']
  for index, row in test_df.iterrows():
    populate_dict(data_dict, row, fields)
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

def populate_dict(dict, row, fields):
  for field in fields:
    dict[field].append(row[field])

def main():
  language = 'Spanish'
  training_df = pre_process_data('data/main_parser.csv')
  method = 'interpolation'
  test('data/tagged_sentences_1000sents.csv', method, training_df)

if __name__ == "__main__":
  main()