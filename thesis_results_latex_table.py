import pandas as pd


def bold_highest(values):
    max_value = max(values)
    new_list = ['\\textbf{' + str(value) + '}' if value == max_value and \
                max_value > 0 else str(value) for value in values]
    return new_list


df = pd.read_csv('data/thesis_tables/error_type_table_ngram.csv')
error_types = list(df['error_type'].unique())

f = open('data/thesis_tables/error_type_analysis_ngram.txt', 'w')

for e in error_types:
    sub_df = df[df['error_type'] == e]
    padded = sub_df[sub_df['span'] == 'Padded error']
    unigram = sub_df[sub_df['span'] == 'Error + unigram']
    bigram = sub_df[sub_df['span'] == 'Error + bigram']
    precisions = bold_highest([list(padded['precision'])[0], list(unigram['precision'])[0], list(bigram['precision'])[0]])
    recall = bold_highest([list(padded['recall'])[0], list(unigram['recall'])[0], list(bigram['recall'])[0]])
    f1 = bold_highest([list(padded['f1'])[0], list(unigram['f1'])[0], list(bigram['f1'])[0]])
    line_padded = '\multirow{3}{=}{' + e + '} & Padded error & ' + precisions[0] + ' & ' + recall[0] + ' & ' + f1[0] + '\\\ \n'
    line_unigram = '& Error + unigram  & ' + precisions[1] + ' & ' + recall[1] + ' & ' + f1[1] + '\\\ \n'
    line_bigram = '& Error + bigram  & ' + precisions[2] + ' & ' + recall[2] + ' & ' + f1[2]  + '\\\ \hline \n'
    f.write(line_padded)
    f.write(line_unigram)
    f.write(line_bigram)

f.close()
