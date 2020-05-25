## Parser 2 output description

Each line in this dataset represents an annotated learner error.
One paragraph line in the essay XML file can contain more than one annotated error.

| Column | Type   | Description |
|--------|:------:|-------------|
| student_id | String | Test taker identification |
| error_type | String | Tag associated with the error. See [Nicholls 2003](http://ucrel.lancs.ac.uk/publications/CL2003/papers/nicholls.pdf) |
| en | Float | Probability of part-of-speech sequence belong to English |
| es | Float | Probability of part-of-speech sequence belong to Spanish |
| correct_trigram_poss | String | Universal part-of-speech tags corresponding to the correction sequence |
| incorrect_trigram_poss | String | Universal part-of-speech tags corresponding to the error sequence |
| correct_trigram | String | Sequence of tree words that begins at the correction index |
| incorrect_trigram | Float | Sequence of tree words that begins at the error index |
| correct_sentence | Float | Sentence with all the errors replaced by their corrections |
| incorrect_sentence | Float | Sentence with all the errors replaced by their corrections, **but the error represented by the row** |
