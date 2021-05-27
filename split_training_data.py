import pandas as pd

from sklearn.model_selection import train_test_split


def create_splits(datasets):
    dfs = []
    for dataset in datasets:
        df = pd.read_csv(dataset, index_col=[0])
        dfs.append(df)
    final_df = pd.concat(dfs)
    X_train, X_test, y_train, y_test = \
        train_test_split(final_df, final_df, test_size=0.2, random_state=42)
    X_train.to_csv('./data/training data/splits/train_split.csv')
    X_test.to_csv('./data/training data/splits/eval_split.csv')


training_datasets = [
    'data/training data/tagged_wmt-news_en-zh.csv',
    'data/training data/tagged_globalvoices_sentences.csv']
create_splits(training_datasets)
