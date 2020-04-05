def sublist_count(sublist, list):
  return sum(list[i:i + len(sublist)] == sublist for i in range(len(list)))