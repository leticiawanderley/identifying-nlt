import csv


def process_tags(input_filename, output_filename):
    f = open(input_filename, "r")
    lines = f.readlines()
    with open(output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for line in lines:
            tags = line.split(',')
            for tag in tags:
                csvwriter.writerow([tag.strip()])


process_tags('spacy_spanish_tags.txt', 'spacy_spanish_tags.csv')
