import pandas as pd
from utils import get_structural_errors


error_meaning = pd.read_csv('data/error_codes/error_type_meaning.csv')

padded = pd.read_csv(
    'data/results_metrics/metrics_kenlm_5_incorrect_ud_tags_padded.csv',
    index_col=[0])
unigram = pd.read_csv(
    'data/results_metrics/metrics_kenlm_5_incorrect_ud_tags_unigram.csv',
    index_col=[0])
bigram = pd.read_csv(
    'data/results_metrics/metrics_kenlm_5_incorrect_ud_tags_bigram.csv',
    index_col=[0])


padded['span'] = 'Padded error'
unigram['span'] = 'Error + unigram'
bigram['span'] = 'Error + bigram'

padded = padded[padded['id'].isin(get_structural_errors())]
unigram = unigram[unigram['id'].isin(get_structural_errors())]
bigram = bigram[bigram['id'].isin(get_structural_errors())]

padded = padded.round(2)
unigram = unigram.round(2)
bigram = bigram.round(2)

padded['error_type'] = padded.apply(lambda row: list(error_meaning[error_meaning['error_type']==row['id']]['meaning'])[0].capitalize() + ' \\break (n = ' + str(row['total']) + ')', axis=1)
unigram['error_type'] = unigram.apply(lambda row: list(error_meaning[error_meaning['error_type']==row['id']]['meaning'])[0].capitalize() + ' \\break (n = ' + str(row['total']) + ')', axis=1)
bigram['error_type'] = bigram.apply(lambda row: list(error_meaning[error_meaning['error_type']==row['id']]['meaning'])[0].capitalize() + ' \\break (n = ' + str(row['total']) + ')', axis=1)

result = pd.concat([padded, unigram, bigram])

result = result.sort_values(by=['total', 'error_type'], ascending=False)

print(result)

result.to_csv('data/thesis_tables/error_type_table_ngram.csv')