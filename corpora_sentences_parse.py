import pandas as pd
import spacy

from utils import tags_mapping, tag_sentences, unpack_ud_and_penn_tags


def pre_process_data(filename, new_column_names, selected_columns=None):
    """Create pandas dataframe with selected columns."""
    df = pd.read_csv(filename)
    if selected_columns:
        df = df[selected_columns]
    df.columns = new_column_names
    return df


def pos_tag(models, df, languages_columns, mapping_filename):
    """Add part-of-speech tags columns to dataframe."""
    mapping = tags_mapping(mapping_filename)
    for lang in languages_columns.keys():
        series = df.apply(lambda x: tag_sentences(models[lang],
                                                  x[languages_columns[lang]],
                                                  lang, mapping), axis=1)
        ud, penn = unpack_ud_and_penn_tags(series)
        df[lang + '_ud'] = ud
        df[lang + '_penn'] = penn
    return df


def main(models, input_filename, output_filename, languages_columns,
         new_column_names, selected_columns):
    df = pre_process_data(input_filename, new_column_names, selected_columns)
    df = pos_tag(models, df, languages_columns,
                 'data/spaCy tags/spacy_spanish_tags_.csv')
    df.to_csv(output_filename, index=True)


if __name__ == "__main__":
    input_filename = 'data/training data/dataset_sentences.csv'
    output_filename = ('data/training data/'
                       'tagged_sentences_dataset_sentences.csv')
    models = {
        'en': spacy.load("en_core_web_md"),
        'es': spacy.load("es_core_news_md")
    }
    new_column_names = ['english', 'spanish']
    selected_columns = ['english', 'spanish']
    languages_columns = {'en': 'english', 'es': 'spanish'}
    main(models, input_filename, output_filename, languages_columns,
         new_column_names, selected_columns)
