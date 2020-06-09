import os
import spacy
from parse_sentences import main

NUMBER_OF_ROWS = 10000
TAGGED = '_tagged.csv'


def split_csv(filename):
    file = open(filename + '.csv').readlines()
    file_index = 1
    for i in range(len(file)):
        if i % NUMBER_OF_ROWS == 0:
            batch_filename = filename + '_' + str(file_index) + '.csv'
            open(batch_filename, 'w+').writelines(file[i:i+NUMBER_OF_ROWS])
            file_index += 1


if __name__ == "__main__":
    folder = 'data/europarl'
    files = os.listdir(folder)
    models = {
        'en': spacy.load("en_core_web_md"),
        'es': spacy.load("es_core_news_md")
    }
    for file in files:
        if TAGGED not in file:
            output_file = file.split('.')[0] + TAGGED
            if output_file not in files:
                print(output_file)
                main(models, folder + '/' + file, folder + '/' + output_file)
