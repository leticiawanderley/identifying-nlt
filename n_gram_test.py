import pandas as pd

from constant import NGRAM_METHODS, INTERPOLATION, LAPLACE, UNSMOOTHED,\
                     PENN_NGRAMS_FILES, UD_NGRAMS_FILES, UD_NGRAMS_FILES_GV,\
                     LEARNER_ENGLISH_FIELDS
from n_gram_model import pre_process_test, process_training_data, test_ngram
from utils import get_structural_errors


def create_dict(fields):
    dict = {}
    for field in fields:
        dict[field] = []
    return dict


def populate_dict(dict, row, fields):
    for field in fields:
        dict[field].append(row[field])


def pre_process_data(filename, fields, language=None):
    df = pd.read_csv(filename)
    df = df[fields]
    if language:
        df = df[df['language'] == language]
    return df


def test(train_dataset_filenames, method, test_df, languages, test_df_fields,
         test_column, output_file):
    data_dict = create_dict(test_df_fields + languages)
    n = NGRAM_METHODS[method][0]
    langs = process_training_data(train_dataset_filenames, method, n,
                                  languages)
    structural_errors = get_structural_errors()
    for index, row in test_df.iterrows():
        if row['error_type'] == '_' or row['error_type'] in structural_errors:
            populate_dict(data_dict, row, test_df_fields)
            for l in langs.keys():
                processed_ngram = pre_process_test(row[test_column].
                                                   split(), langs[l][1])
                probability = 0
                if len(processed_ngram) > n:
                    for i in range(0, len(processed_ngram) - n):
                        probability += test_ngram(method, n,
                                                  processed_ngram[i:i+n],
                                                  langs[l])
                else:
                    probability += test_ngram(method, n, processed_ngram,
                                              langs[l])
                data_dict[l].append(probability)
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(output_file)


def main():
    vocab_files = PENN_NGRAMS_FILES
    language = 'Spanish'
    fields = ['student_id', 'language', 'error_type',
              'correct_trigram_penn', 'incorrect_trigram_penn',
              'correct_trigram', 'incorrect_trigram',
              'correct_sentence', 'incorrect_sentence']
    fields = LEARNER_ENGLISH_FIELDS
    test_df = pre_process_data('data/testing data/'
                               'parsed_learner_english_sentences_.csv',
                               fields)
    languages = ['en', 'es']
    method = INTERPOLATION
    test_column = 'penn'
    output_file = 'data/results_learner_english_' + test_column + '_' +\
                  method + '.csv'
    test(vocab_files, method, test_df, languages,
         fields, test_column, output_file)


def test_fce_annotated_data():
    vocab_files = UD_NGRAMS_FILES_GV
    fields = ['error_type', 'Negative transfer?',
              'correct_sentence', 'correct_trigram_ud',
              'incorrect_sentence', 'incorrect_trigram_ud']
    test_df = pre_process_data('data/testing data/annotated_FCE/'
                               'chinese_annotated_errors.csv',
                               fields)
    languages = ['en', 'zhs']
    method = INTERPOLATION
    test_column = 'incorrect_trigram_ud'
    output_file = 'data/results_chinese_fce_' + test_column + '_' +\
                  method + '.csv'
    test(vocab_files, method, test_df, languages,
         fields, test_column, output_file)

if __name__ == "__main__":
    test_fce_annotated_data()
