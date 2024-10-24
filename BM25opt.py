################################################################################
#
#  BM25opt : faster BM25 search algorithms in Python
#  by András Jankovics  https://github.com/jankovicsandras  andras@jankovics.net 
#  based on https://github.com/dorianbrown/rank_bm25 by Dorian Brown
#  Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0
#
################################################################################


import math, numpy as np


# default tokenizer: split-on-whitespace + lowercase + remove common punctiation
def tokenizer_default( s ) :
  ltrimchars = ['(','[','{','<','\'','"']
  rtrimchars = ['.', '?', '!', ',', ':', ';', ')', ']', '}', '>','\'','"']
  if type(s) != str : return []
  wl = s.lower().split()
  for i,w in enumerate(wl) :
    if len(w) < 1 : continue
    si = 0
    ei = len(w)
    while si < ei and w[si] in ltrimchars : si += 1
    while ei > si and w[ei-1] in rtrimchars : ei -= 1
    wl[i] = wl[i][si:ei]
  wl = [ w for w in wl if len(w) > 0 ]
  return wl


# simple split-on-whitespace tokenizer, not recommended
def tokenizer_whitespace( s ) :
  return s.split()


# integrated BM25 class with multiple algoritms and options
class BM25opt :
  def __init__(self, corpus, algo='okapi', tokenizer_function=tokenizer_default, idf_algo=None, k1=None, b=None, epsilon=None, delta=None ):
    # tokenizing input
    self.tokenizer_function = tokenizer_function
    tokenized_corpus = [ self.tokenizer_function( document ) for document in corpus ]
    # algoritm selection
    self.algo = algo.strip().lower()
    if self.algo not in ['okapi','l','plus'] : self.algo = 'okapi'

    # constants
    if algo == 'okapi' :
      self.k1 = k1 if k1 is not None else 1.5
      self.b = b if b is not None else 0.75
      self.epsilon = epsilon if epsilon is not None else 0.25
      self.delta = delta if delta is not None else 1
    if algo == 'l' :
      self.k1 = k1 if k1 is not None else 1.5
      self.b = b if b is not None else 0.75
      self.delta = delta if delta is not None else 0.5
      self.epsilon = epsilon if epsilon is not None else 0.25
    if algo == 'plus' :
      self.k1 = k1 if k1 is not None else 1.5
      self.b = b if b is not None else 0.75
      self.delta = delta if delta is not None else 1
      self.epsilon = epsilon if epsilon is not None else 0.25

    # common
    self.corpus_len = len(corpus)
    self.avg_doc_len = 0
    self.word_freqs = []
    self.idf = {}
    self.doc_lens = []
    word_docs_count = {}  # word -> number of documents with word
    total_word_count = 0

    for document in tokenized_corpus:
      # doc lengths and total word count
      self.doc_lens.append(len(document))
      total_word_count += len(document)
      # word frequencies in this document
      frequencies = {}
      for word in document:
        if word not in frequencies:
          frequencies[word] = 0
        frequencies[word] += 1
      self.word_freqs.append(frequencies)
      # number of documents with word count
      for word, freq in frequencies.items():
        try:
          word_docs_count[word] += 1
        except KeyError:
          word_docs_count[word] = 1

    # average document length
    self.avg_doc_len = total_word_count / self.corpus_len

    # IDF 
    # https://github.com/dorianbrown/rank_bm25/issues/35 : atire IDF correction is possible if idf_algo is set
    self.idf_algo = idf_algo if idf_algo is not None else algo
    if self.idf_algo == 'okapi' :
      # Calculates frequencies of terms in documents and in corpus.
      # This algorithm sets a floor on the idf values to eps * average_idf
      # collect idf sum to calculate an average idf for epsilon value
      # collect words with negative idf to set them a special epsilon value.
      # idf can be negative if word is contained in more than half of documents
      idf_sum = 0
      negative_idfs = []
      for word, freq in word_docs_count.items():
        idf = math.log(self.corpus_len - freq + 0.5) - math.log(freq + 0.5)
        self.idf[word] = idf
        idf_sum += idf
        if idf < 0:
          negative_idfs.append(word)
      self.average_idf = idf_sum / len(self.idf)
      # assign epsilon
      eps = self.epsilon * self.average_idf
      for word in negative_idfs:
        self.idf[word] = eps
    if self.idf_algo == 'l' : # IDF for BM25L
      for word, doccount in word_docs_count.items():
        self.idf[word] = math.log(self.corpus_len + 1) - math.log(doccount + 0.5)
    if self.idf_algo == 'plus' : # IDF for BM25Plus
      for word, doccount in word_docs_count.items():
        self.idf[word] = math.log(self.corpus_len + 1) - math.log(doccount)

    # "half divisor"
    self.hds = [  ( 1-self.b + self.b*doc_len/self.avg_doc_len) for doc_len in self.doc_lens ]

    # words * documents score map
    self.wsmap = {}
    for word in self.idf :
      self.wsmap[word] = np.zeros( self.corpus_len )
      for di in range(0,self.corpus_len) :
        twf = (self.word_freqs[di].get(word) or 0)
        if algo == 'okapi' : 
          self.wsmap[word][di] = self.idf[word] * ( twf * (self.k1 + 1) / ( twf + self.k1 * self.hds[di] ) )
        if algo == 'l' : 
          self.wsmap[word][di] = self.idf[word] * twf * (self.k1 + 1) * ( twf/self.hds[di] + self.delta) / (self.k1 + twf/self.hds[di] + self.delta)
        if algo == 'plus' : 
          self.wsmap[word][di] = self.idf[word] * (self.delta + ( twf * (self.k1 + 1) / ( twf + self.k1 * self.hds[di] ) ))

    ### End of __init__()
    

  # get a list of scores for every document
  def get_scores( self, query ):
    tokenizedquery = self.tokenizer_function( query )
    # zeroes list of scores
    scores = np.zeros( self.corpus_len )
    # for each word in tokenizedquery, if word is in wsmap, lookup and add word score for every documents' scores
    for word in tokenizedquery:
      if word in self.wsmap :
        scores += self.wsmap[word]
    # return scores list (not sorted)
    return scores


  # return [id,score] for the top k documents
  def topk( self, query, k=None ):
    docscores = self.get_scores( query )
    sisc = [ [i,s] for i,s in enumerate(docscores) ]
    sisc.sort(key=lambda x:x[1],reverse=True)
    if k :
      sisc = sisc[:k]
    return sisc

  ### End of class BM25opt
