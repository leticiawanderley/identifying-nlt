import pandas as pd
import spacy

from constant import CHINESE, ENGLISH, SPANISH
from utils import tags_mapping, tag_sentences, unpack_ud_and_penn_tags


def pre_process_data(filename, new_column_names, selected_columns=None):
    """Create pandas dataframe with selected columns."""
    df = pd.read_csv(filename)
    if selected_columns:
        df = df[selected_columns]
    df.columns = new_column_names
    return df


def pos_tag(models, df, languages_columns, mapping_filename=None):
    """Add part-of-speech tags columns to dataframe."""
    mapping = None
    if mapping_filename:
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
         new_column_names, selected_columns, mapping=None):
    df = pre_process_data(input_filename, new_column_names, selected_columns)
    df = pos_tag(models, df, languages_columns)
    df = df.dropna()
    df.to_csv(output_filename, index=True)



if __name__ == "__main__":
    l1 = CHINESE
    if l1 == SPANISH:
        input_filename = 'data/training data/dataset_sentences.csv'
        output_filename = ('data/training data/'
                           'tagged_sentences_dataset_sentences.csv')
        models = {
            ENGLISH: spacy.load("en_core_web_md"),
            SPANISH: spacy.load("es_core_news_md")
        }
        new_column_names = ['english', 'spanish']
        selected_columns = ['english', 'spanish']
        languages_columns = {ENGLISH: 'english', SPANISH: 'spanish'}
        mapping = 'data/tags/spacy_spanish_tags_.csv'
        main(models, input_filename, output_filename, languages_columns,
             new_column_names, selected_columns, mapping)
    elif l1 == CHINESE:
        input_filename = 'data/training data/wmt-news_en-zh.csv'
        output_filename = ('data/training data/'
                           'tagged_wmt-news_en-zh.csv')
        models = {
            ENGLISH: spacy.load("en_core_web_lg"),
            CHINESE: spacy.load("zh_core_web_lg")
        }
        new_column_names = ['english', 'chinese']
        selected_columns = [ENGLISH, CHINESE]
        languages_columns = {ENGLISH: 'english', CHINESE: 'chinese'}
        main(models, input_filename, output_filename, languages_columns,
             new_column_names, selected_columns)
