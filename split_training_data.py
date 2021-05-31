import pandas as pd

from sklearn.model_selection import KFold, train_test_split


def create_splits(datasets):
    dfs = []
    for dataset in datasets:
        df = pd.read_csv(dataset, index_col=[0])
        dfs.append(df)
    final_df = pd.concat(dfs)
    X_train, X_test, y_train, y_test = \
        train_test_split(final_df, final_df, test_size=0.2, random_state=42)
    X_train.to_csv('./data/training_data/chinese_english_splits/train_split.csv')
    X_test.to_csv('./data/training_data/chinese_english_splits/eval_split.csv')


def create_folds(datasets):
    dfs = []
    for dataset in datasets:
        df = pd.read_csv(dataset, index_col=[0])
        dfs.append(df)
    final_df = pd.concat(dfs)
    kf = KFold(n_splits=5, shuffle=True)
    i = 1
    for train_index, test_index in kf.split(final_df):
        train_fold, test_fold = final_df.iloc[list(train_index)], \
            final_df.iloc[list(test_index)]
        test_fold.to_csv(
            'data/training_data/chinese_english_splits/test_fold_' +
            str(i) + '.csv')
        en_train_file = open(
            'data/training_data/chinese_english_splits/en_train_fold_' +
            str(i) + '.txt', 'w')
        zhs_train_file = open(
            'data/training_data/chinese_english_splits/zhs_train_fold_' +
            str(i) + '.txt', 'w')
        en_train_file.writelines('\n'.join(list(train_fold['en_ud'])))
        zhs_train_file.writelines('\n'.join(list(train_fold['zhs_ud'])))
        en_train_file.close()
        zhs_train_file.close()
        i += 1


training_datasets = [
    'data/training_data/chinese_english/tagged_wmt-news_en-zh.csv',
    'data/training_data/chinese_english/tagged_globalvoices_sentences.csv']
create_folds(training_datasets)
