import pandas as pd
import spacy

from utils import tags_mapping, unpack_poss_and_tags


CONJ = 'CONJ'
CCONJ = 'CCONJ'
SPANISH = 'es'


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
        series = df.apply(lambda x: tag_sentences(models,
                                                  x[languages_columns[lang]],
                                                  lang, mapping), axis=1)
        poss, tags = unpack_poss_and_tags(series)
        df[lang + '_poss'] = poss
        df[lang + '_tags'] = tags
    return df


def tag_sentences(models, sentence, language, mapping):
    """Part-of-speeh tag dataframe sentence."""
    poss = ''
    tags = ''
    if type(sentence) == str:
        nlp = models[language]
        doc = nlp(sentence)
        for token in doc:
            pos = token.pos_
            if token.pos_ == CONJ:
                pos = CCONJ if language == SPANISH else pos
            poss += pos + ' '
            tag = token.tag_
            if language == SPANISH:
                tag = mapping[tag]
            tags += tag + ' '
    return (poss, tags)


def main(models, input_filename, output_filename):
    new_column_names = ['english', 'spanish']
    selected_columns = ['english', 'spanish']
    df = pre_process_data(input_filename, new_column_names, selected_columns)
    languages_columns = {'en': 'english', 'es': 'spanish'}
    df = pos_tag(models, df, languages_columns, 'data/spacy_spanish_tags_.csv')
    df.to_csv(output_filename, index=True)


if __name__ == "__main__":
    input_filename = 'data/dataset_sentences.csv'
    output_filename = 'data/tagged_sentences_dataset_sentences.csv'
    models = {
        'en': spacy.load("en_core_web_md"),
        'es': spacy.load("es_core_news_md")
    }
    main(models, input_filename, output_filename)
