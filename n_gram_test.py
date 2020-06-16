import pandas as pd

from constant import NGRAM_METHODS, INTERPOLATION, LAPLACE, UNSMOOTHED,\
                     TAGS_NGRAMS_FILES, POSS_NGRAMS_FILES
from n_gram_model import pre_process_test, process_training_data, test_ngram


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
    for index, row in test_df.iterrows():
        populate_dict(data_dict, row, test_df_fields)
        for l in langs.keys():
            processed_ngram = pre_process_test(row[test_column].
                                               split(), langs[l][1])
            probability = 0
            if len(processed_ngram) > n:
                for i in range(0, len(processed_ngram) - n):
                    probability += test_ngram(method, n,
                                              processed_ngram[i:i+n], langs[l])
            else:
                probability += test_ngram(method, n, processed_ngram, langs[l])
            data_dict[l].append(probability)
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(output_file)


def main():
    vocab_files = TAGS_NGRAMS_FILES
    language = 'Spanish'
    fields = ['student_id', 'language', 'error_type',
              'correct_trigram_tags', 'incorrect_trigram_tags',
              'correct_trigram', 'incorrect_trigram',
              'correct_sentence', 'incorrect_sentence']
    fields = ['sentence', 'correct', 'tags', 'tags_trigram', 'poss',
              'poss_trigram']
    test_df = pre_process_data('data/testing data/'
                               'parsed_learner_english_sentences_.csv',
                               fields)
    languages = ['en', 'es']
    method = INTERPOLATION
    test_column = 'tags_trigram'
    output_file = 'data/results_learner_english_trigrams_interpolation.csv'
    test(vocab_files, method, test_df, languages,
         fields, test_column, output_file)


if __name__ == "__main__":
    main()
