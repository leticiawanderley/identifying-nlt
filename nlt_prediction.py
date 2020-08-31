import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from constant import GROUND_TRUTH


def concat_dummies(df, dummies):
    for dummy in list(dummies):
        df[dummy] = dummies[dummy]
    return list(dummies)


def create_dummies(df, column, prefix):
    dummies = pd.get_dummies(df[column], prefix=prefix)
    return concat_dummies(df, dummies)


def process_trigram(data):
    data['ud_0'], data['ud_1'], data['ud_2'] = \
        data['incorrect_trigram_ud'].str.split(' ').str
    ud_0 = create_dummies(data, 'ud_0', 'ud_0')
    ud_1 = create_dummies(data, 'ud_1', 'ud_1')
    ud_2 = create_dummies(data, 'ud_2', 'ud_2')
    error_type = create_dummies(data, 'error_type', 'error_type')
    columns = list(ud_0) + list(ud_1) + list(ud_2) + list(error_type)
    return data, columns


def train_random_forest():
    df = pd.read_csv('data/testing data/annotated_FCE/' +
                     'chinese_annotated_errors.csv')
    df, columns = process_trigram(df)
    x_train, x_test, y_train, y_test = train_test_split(df, df[GROUND_TRUTH],
                                                        test_size=0.1)
    fit_data = x_train[columns]
    clf = RandomForestClassifier(n_estimators=500,
                                 max_depth=30,
                                 random_state=0)
    clf.fit(fit_data, y_train)
    print(clf.score(x_test[columns], y_test))


train_random_forest()
