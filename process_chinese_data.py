import re
import pandas as pd

from constant import CHINESE, ENGLISH


def extract_sentences(filename):
    source_re = r'(?<=<source>)(.*?)(?=<\/source>)'
    target_re = r'(?<=<target>)(.*?)(?=<\/target>)'
    data_dict = {CHINESE: [], ENGLISH: []}
    file = open(filename).readlines()
    for line in file:
        if '<source>' in line:
            source_match = re.search(source_re, line, re.MULTILINE)
            data_dict[CHINESE].append(source_match.groups()[0])
        elif '<target>' in line:
            target_match = re.search(target_re, line, re.MULTILINE)
            data_dict[ENGLISH].append(target_match.groups()[0])
    return data_dict


def main(input_filename, output_filename):
    data_dict = extract_sentences(input_filename)
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(output_filename)


if __name__ == "__main__":
    input_filename = 'data/training data/globalvoices.zhs-en.xliff'
    output_filename = ('data/training data/'
                       'globalvoices_sentences.csv')
    main(input_filename, output_filename)
