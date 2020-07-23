## Results file data description

Each line in this dataset represents one annotated learner error.

| Column | Type   | Description |
|--------|:------:|-------------|
| student_id | String | Test taker identification |
| error_type | String | Tag associated with the error, see [Nicholls 2003](http://ucrel.lancs.ac.uk/publications/CL2003/papers/nicholls.pdf) |
| en | Float | Probability of part-of-speech sequence belong to English |
| es | Float | Probability of part-of-speech sequence belong to Spanish |
| correct_trigram_ud | String | Universal part-of-speech tags corresponding to the correction sequence |
| incorrect_trigram_ud | String | Universal part-of-speech tags corresponding to the error sequence |
| correct_trigram | String | Sequence of tree words that begins at the correction index |
| incorrect_trigram | String | Sequence of tree words that begins at the error index |
| correct_sentence | String | Sentence with all the errors replaced by their corrections |
| incorrect_sentence | String | Sentence with all the errors replaced by their corrections, **but the error represented by the row** |
