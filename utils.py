def sublist_count(sublist, list):
  """Count the amount of times a sublist appears inside a list."""
  return sum(list[i:i + len(sublist)] == sublist for i in range(len(list)))