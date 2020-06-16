import csv
import spacy

from utils import tag_sentences


def parse_sentences(input_file, output_file):
    nlp = spacy.load("en_core_web_lg")
    with open(input_file, newline='') as csvfile:
        sentences = csv.reader(csvfile, delimiter=',')
        with open(output_file, 'w', newline='') as csvfile2:
            parsed = csv.writer(csvfile2, delimiter=',')
            parsed.writerow(['sentence', 'correct', 'tags', 'poss'])
            for row in sentences:
                parsed_sentence = tag_sentences(nlp, row[0])
                parsed.writerow([row[0], row[1], parsed_sentence[1],
                                 parsed_sentence[0]])


parse_sentences('data/testing data/learner_english_sentences.csv',
                'data/testing data/parsed_learner_english_sentences.csv')
