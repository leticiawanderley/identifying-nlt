import pandas as pd
import spacy

models = {
  'en': spacy.load("en_core_web_sm"),
  'es': spacy.load("es_core_news_sm")
}

def read_sentences(filename):
  df = pd.read_csv(filename)
  df = df[['EXAMPLE (EN)', 'EXAMPLE (ES)']]
  df.columns = ['english', 'spanish']
  return df

def pos_tag(df):
  df['en_pos'] = df.apply( lambda x : get_tags(x['english'], 'en'), axis=1)
  df['es_pos'] = df.apply( lambda x : get_tags(x['spanish'], 'es'), axis=1)
  return df

def get_tags(sentence, language):
  nlp = models[language]
  doc = nlp(sentence)
  tags = ''
  for token in doc:
    tags += token.pos_ + ' '
  return tags

def main():
  df = read_sentences('1000sents.csv')
  df = pos_tag(df)
  df.to_csv('tagged_sentences.csv', index=True)

if __name__ == "__main__":
  main()