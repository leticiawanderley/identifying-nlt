import pandas as pd

def pre_process_data(filename):
  df = pd.read_csv(filename)
  df = df[df['es'] > df['en']]
  return df

def data_summary_count(df, columns):
  counter = columns[0]
  grouped = df.groupby(columns)[counter].count().reset_index(name='count')
  grouped = grouped.sort_values('count', ascending=False)
  grouped.to_csv('data/top_' + '_'.join(columns) + '.csv')

def create_summaries(df, columns):
  for column in columns:
    data_summary_count(df, column)

create_summaries(pre_process_data('data/results.csv'),
                [['error_type'],
                 ['incorrect_trigram_poss'],
                 ['incorrect_trigram_poss', 'correct_trigram_poss'],
                 ['incorrect_trigram_poss', 'error_type'],
                 ['correct_trigram_poss', 'error_type']])