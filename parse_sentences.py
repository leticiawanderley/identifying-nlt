import pandas as pd
import spacy

models = {
  'en': spacy.load("en_core_web_md"),
  'es': spacy.load("es_core_news_md")
}

def pre_process_data(filename, selected_columns, new_column_names):
    """Create pandas dataframe with selected columns."""
    df = pd.read_csv(filename)
    df = df[selected_columns]
    df.columns = new_column_names
    return df


def pos_tag(df, languages_columns):
    """Add part-of-speech tags columns to dataframe."""
    for lang in languages_columns.keys():
        df[lang + '_pos'] = df.apply(lambda x : tag_sentences(x[languages_columns[lang]], lang), axis=1)
    return df


def tag_sentences(sentence, language):
    """Part-of-speeh tag dataframe sentence."""
    tags = ''
    if type(sentence) == str:
        nlp = models[language]
        doc = nlp(sentence)
        for token in doc:
            tags += token.pos_ + ' '
    return tags


def main():
    selected_columns = ['english', 'spanish']
    new_column_names = ['english', 'spanish']
    filename = 'data/dataset_sentences.csv'
    df = pre_process_data(filename, selected_columns, new_column_names)

    languages_columns = {'en': 'english', 'es': 'spanish'}
    df = pos_tag(df, languages_columns)

    results_filename = 'data/tagged_sentences_dataset_sentences.csv'
    df.to_csv(results_filename, index=True)


if __name__ == "__main__":
    main()
