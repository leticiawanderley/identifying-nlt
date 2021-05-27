import pandas as pd


def pre_process_data(filename, columns):
    """Filter dataset rows.

    Select columns in which the first column value
    is larger than the second column value."""
    df = pd.read_csv(filename)
    df = df[df[columns[0]] > df[columns[1]]]
    return df


def data_summary_count(df, columns):
    """Summarize dataset grouping and counting it by columns."""
    counter = columns[0]
    grouped = df.groupby(columns)[counter].count().reset_index(name='count')
    grouped = grouped.sort_values('count', ascending=False)
    grouped.to_csv('data/summaries/more_common_in_es_' + '_'.join(columns) +
                   '_.csv')


def create_summaries(df, columns):
    """Create counts summaries of selected columns."""
    for column in columns:
        data_summary_count(df, column)


def main():
    columns = ['es', 'en']
    results_file = 'data/results_main_parser_interpolation.csv'
    create_summaries(pre_process_data(results_file, columns),
                     [['error_type'],
                      ['incorrect_trigram_penn'],
                      ['incorrect_trigram_penn', 'correct_trigram_penn'],
                      ['incorrect_trigram_penn', 'error_type'],
                      ['correct_trigram_penn', 'error_type']])


if __name__ == "__main__":
    main()
