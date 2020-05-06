import pandas as pd
import spacy

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


def pos_tag(models, df, languages_columns):
    """Add part-of-speech tags columns to dataframe."""
    for lang in languages_columns.keys():
        df[lang + '_pos'] = df.apply(lambda x:
                                     tag_sentences(models,
                                                   x[languages_columns[lang]],
                                                   lang), axis=1)
    return df


def tag_sentences(models, sentence, language):
    """Part-of-speeh tag dataframe sentence."""
    tags = ''
    if type(sentence) == str:
        nlp = models[language]
        doc = nlp(sentence)
        for token in doc:
            pos = token.pos_
            if token.pos_ == CONJ:
                pos = CCONJ if language == SPANISH else pos
            tags += pos + ' '
    return tags


def main(models, input_filename, output_filename):
    new_column_names = ['english', 'spanish']
    df = pre_process_data(input_filename, new_column_names)
    languages_columns = {'en': 'english', 'es': 'spanish'}
    df = pos_tag(models, df, languages_columns)
    df.to_csv(output_filename, index=True)


if __name__ == "__main__":
    input_filename = 'data/1000sents.csv'
    output_filename = 'data/tagged_sentences_1000sents.csv'
    models = {
        'en': spacy.load("en_core_web_md"),
        'es': spacy.load("es_core_news_md")
    }
    main(models, input_filename, output_filename)
