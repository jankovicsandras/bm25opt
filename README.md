# BM25opt
## faster BM25 search algorithms in Python
####  based on https://github.com/dorianbrown/rank_bm25 by Dorian Brown
####  Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0
----
## Usage:
#### Input:
 - ```corpus``` is a list of strings, e.g. ```[ 'bla bla bla', 'this is document two', ... ]```
 - ```question``` is a string, e.g. ```'which text contains the word two?'```
 - optional arguments:
   - ```algo``` : BM25 algorithm, the default is ```'okapi'```; ```'l'``` and ```'plus'``` available
   - ```tokenizer_function``` : the default is ```tokenizer_default``` which is split-on-whitespace, lowercase, remove common punctiations
   - ```idf_algo``` : default uses the same IDF as ```rank_bm25```; values ```'okapi'```, ```'l'``` and ```'plus'``` can override to fix https://github.com/dorianbrown/rank_bm25/issues/35
   - ```k1```, ```b```, ```epsilon```, ```delta``` : constants with standard default values, see https://en.wikipedia.org/wiki/Okapi_BM25
#### Example 1:
This example uses the default tokenizer and the default BM25Okapi algorithm and returns the top 5 highest scoring document ids and scores.
```python
bm25opt_index = BM25opt( corpus )
results = bm25opt_index.topk( question, 5 )
print( 'results[0] id', results[0][0], 'results[0] score', results[0][1], 'results[0] document', corpus[ results[0][0] ] )
```
#### Example 2:
This example returns the list of document scores (order is the same as the document order in corpus), shows algoritm selection and custom tokenizer function.
```python
bm25opt_index = BM25opt( corpus, algo='plus', tokenizer_function=some_tokenizer_function )
doc_scores = bm25opt_index.get_scores( question )
```
#### Example 3: comparison with rank_bm25
This example shows the score list and the similarity with [```rank_bm25```](https://github.com/dorianbrown/rank_bm25), but NOTE: BM25opt input is not tokenized beforehand.
```python
corpus = [ ... ]
question = '...'
tokenized_corpus = [ tokenizer_default(document) for document in corpus ]
tokenized_question = tokenizer_default( question )

rank_bm25_index = BM25Okapi( tokenized_corpus )
bm25opt_index = BM25opt( corpus, algo='okapi' )

rank_bm25_scores = rank_bm25_index.get_scores( tokenizedquestion )
bm25opt_scores = bm25opt_index.get_scores( question )
```
----
### Notes:
This is an optimized variant of rank_bm25 where the key insight is that we can calculate almost everything at index creation time in ```__init__()``` , resulting a words * documents-score dict, e.g.
```python
wsmap = {
  'word1': [ word1_doc1_score, word1_doc2_score, ... ],
  'word2': [ word2_doc1_score, word2_doc2_score, ... ],
  ...
}
```
then the query function is just adding the score lists for each word in the question, e.g. 
```python
question = 'word1 word2'
doc_scores = [ wsmap['word1'][0] + wsmap['word2'][0], wsmap['word1'][1] + wsmap['word2'][1], ... ]
```
Another important change is the un-tokenized inputs and registration of the tokenizer function, which is important to avoid situations where the corpus would be tokenized with a different function than the queries later. A simple ```tokenizer_default()``` function is provided as a default.

