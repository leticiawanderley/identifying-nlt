import pandas as pd

from constant import NGRAM_METHODS, INTERPOLATION, LAPLACE, UNSMOOTHED,\
                     PENN_NGRAMS_FILES, UD_NGRAMS_FILES, UD_NGRAMS_FILES_ZH,\
                     LEARNER_ENGLISH_FIELDS, ANNOTATED_FCE_FIELDS,\
                     CHINESE, ENGLISH, SPANISH, UD_NGRAMS_FILES_SPLIT
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


def evaluation(train_dataset_filenames, eval_df, method, n, languages,
               test_column, output_file):
    data_dict = create_dict(languages + [test_column])
    langs = process_training_data(train_dataset_filenames, method, n,
                                  languages)
    for index, row in eval_df.iterrows():
        populate_dict(data_dict, row, [test_column])
        for l in langs.keys():
            probability = 0
            if isinstance(row[test_column], str):
                processed_ngram = pre_process_test(row[test_column].
                                                   split(), langs[l][1])
                if len(processed_ngram) > n:
                    for i in range(0, len(processed_ngram) - n):
                        partial_probability = \
                            test_ngram(method, n, processed_ngram[i:i+n],
                                       langs[l])
                        if partial_probability == 0:
                            probability = float('-inf')
                            break
                        probability += partial_probability
                elif len(processed_ngram) == n:
                    if processed_ngram:
                        probability = test_ngram(method, n, processed_ngram,
                                                 langs[l])
                        if probability == 0:
                            probability = float('-inf')
            data_dict[l].append(probability)
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(output_file)


def test(train_dataset_filenames, method, n, test_df, languages, test_df_fields,
         test_column, output_file):
    data_dict = create_dict(test_df_fields + languages)
    langs = process_training_data(train_dataset_filenames, method, n,
                                  languages)
    structural_errors = get_structural_errors()
    for index, row in test_df.iterrows():
        if row['error_type'] == '_' or row['error_type'] in structural_errors:
            populate_dict(data_dict, row, test_df_fields)
            for l in langs.keys():
                probability = 0
                if isinstance(row[test_column], str):
                    processed_ngram = pre_process_test(row[test_column].
                                                       split(), langs[l][1])
                    if len(processed_ngram) > n:
                        for i in range(0, len(processed_ngram) - n):
                            partial_probability = \
                                test_ngram(method, n, processed_ngram[i:i+n],
                                           langs[l])
                            if partial_probability == 0:
                                probability = float('-inf')
                                break
                            probability += partial_probability
                    elif len(processed_ngram) == n:
                        if processed_ngram:
                            probability = test_ngram(method, n,
                                                     processed_ngram,
                                                     langs[l])
                            if probability == 0:
                                probability = float('-inf')
                data_dict[l].append(probability)
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(output_file)


def main():
    vocab_files = UD_NGRAMS_FILES
    language = 'Spanish'
    fields = ['student_id', 'language', 'error_type',
              'correct_trigram_penn', 'incorrect_trigram_penn',
              'correct_trigram', 'incorrect_trigram',
              'correct_sentence', 'incorrect_sentence']
    fields = LEARNER_ENGLISH_FIELDS
    test_df = pre_process_data('data/testing data/'
                               'parsed_learner_english_sentences_.csv',
                               fields)
    languages = [ENGLISH, SPANISH]
    method = INTERPOLATION
    test_column = 'ud'
    output_file = 'data/results_learner_english_' + test_column + '_' +\
                  method + '.csv'
    test(vocab_files, method, test_df, languages,
         fields, test_column, output_file)


def test_fce_annotated_data():
    vocab_files = UD_NGRAMS_FILES_ZH
    fields = ANNOTATED_FCE_FIELDS
    test_df = pre_process_data('data/testing data/fce_processed_data.csv',
                               fields)
    languages = [ENGLISH, CHINESE]
    method = INTERPOLATION
    n = 2
    test_column = 'incorrect_ud_tags_bigram'
    output_file = 'data/results_chinese_fce_' + test_column + '_' +\
                  method + '.csv'
    test(vocab_files, method, n, test_df, languages,
         fields, test_column, output_file)


def parameter_tuning():
    vocab_files = UD_NGRAMS_FILES_SPLIT
    eval_df = pd.read_csv('data/training data/splits/eval_split.csv')
    languages = [ENGLISH, CHINESE]
    methods = [UNSMOOTHED, LAPLACE, INTERPOLATION]
    for n in range(1, 6):
        for method in methods:
            for test_column in ['en_ud', 'zhs_ud']:
                output_file = test_column + '_' + method + '_' + str(n) + \
                              '.csv'
                print(output_file)
                evaluation(vocab_files, eval_df, method, n, languages,
                           test_column, output_file)


if __name__ == "__main__":
    test_fce_annotated_data()
