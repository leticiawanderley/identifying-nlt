import csv
import spacy


def parse_sentences(input_file, output_file):
    nlp = spacy.load("en_core_web_lg")
    with open(input_file, newline='') as csvfile:
        sentences = csv.reader(csvfile, delimiter=',')
        with open(output_file, 'w', newline='') as csvfile2:
            parsed = csv.writer(csvfile2, delimiter=',')
            parsed.writerow(['sentence', 'correct', 'tags'])
            for row in sentences:
                parsed.writerow([row[0], row[1],
                                 tag_sentences(nlp, row[0])])


def tag_sentences(nlp, sentence):
    """Part-of-speeh tag a sentence."""
    tags = ''
    doc = nlp(sentence)
    for token in doc:
        tags += token.tag_ + ' '
    return tags


parse_sentences('data/learner_english_sentences.csv', 'data/parsed_learner_english_sentences.csv')