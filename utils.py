def sublist_count(sublist, dataset):
  """Count the amount of times a sublist appears inside a list."""
  count = 0
  for sentence in dataset:
    count += sum(sentence[i:i + len(sublist)] == sublist for i in range(len(sentence)))
  return count

def split_sentences(dataset):
  for i in range(len(dataset)):
    dataset[i] = dataset[i].split()

def create_vocab_dict(dataset):
  dataset_size = 0
  count_dict = {}
  for sent in dataset:
    for tag in sent:
      dataset_size += 1
      if tag not in count_dict.keys():
        count_dict[tag] = 0
      count_dict[tag] += 1
  return count_dict, dataset_size

def iterate_rows():
  return