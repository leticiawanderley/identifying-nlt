import pandas as pd

from utils import get_structural_errors


def select_structural_errors(dataset_filename):
    df = pd.read_csv(dataset_filename, index_col=[0])
    df = df[df['error_type'].isin(get_structural_errors())]
    nlt_map = {'Y': True, 'N': False}
    df['Negative transfer?'] = df['Negative transfer?'].map(nlt_map)
    df.to_csv('data/test_data/zhs_structural_errors.csv')


select_structural_errors('data/test_data/fce_processed_data.csv')
