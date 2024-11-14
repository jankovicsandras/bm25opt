################################################################################
#
#  BM25opt : faster BM25 search algorithms in Python
#  version 1.1.0
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
  def __init__( self, corpus, algo='okapi', tokenizer_function=tokenizer_default, idf_algo=None, k1=None, b=None, epsilon=None, delta=None ) :
    # version
    self.version = '1.1.0'
    # tokenizing input
    self.tokenizer_function = tokenizer_function
    tokenized_corpus = [ self.tokenizer_function( document ) for document in corpus ]
    # algoritm selection
    self.algo = algo.strip().lower()
    if self.algo not in ['okapi','l','plus'] : self.algo = 'okapi'
    self.idf_algo = idf_algo if idf_algo is not None else algo
    if self.idf_algo not in ['okapi','l','plus'] : self.idf_algo = 'okapi'

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
    self.doc_lens = []
    self.word_docs_count = {}  # word -> number of documents with word
    self.total_word_count = 0

    for document in tokenized_corpus:
      # doc lengths and total word count
      self.doc_lens.append(len(document))
      self.total_word_count += len(document)
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
          self.word_docs_count[word] += 1
        except KeyError:
          self.word_docs_count[word] = 1

    # create wsmap
    self.createwsmap()

    ### End of __init__()


  # creating the words * documents score map
  def createwsmap(self) :

    # average document length
    self.avg_doc_len = self.total_word_count / self.corpus_len

    # IDF
    self.idf = {}
    # https://github.com/dorianbrown/rank_bm25/issues/35 : atire IDF correction is possible if idf_algo is set
    if self.idf_algo == 'okapi' :
      idf_sum = 0
      negative_idfs = []
      for word, freq in self.word_docs_count.items():
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
      for word, doccount in self.word_docs_count.items():
        self.idf[word] = math.log(self.corpus_len + 1) - math.log(doccount + 0.5)
    if self.idf_algo == 'plus' : # IDF for BM25Plus
      for word, doccount in self.word_docs_count.items():
        self.idf[word] = math.log(self.corpus_len + 1) - math.log(doccount)

    # "half divisor"
    self.hds = [ ( 1-self.b + self.b*doc_len/self.avg_doc_len) for doc_len in self.doc_lens ]

    # words * documents score map
    self.wsmap = {}
    for word in self.idf :
      self.wsmap[word] = np.zeros( self.corpus_len )
      for di in range(0,self.corpus_len) :
        twf = (self.word_freqs[di].get(word) or 0)
        if self.algo == 'okapi' :
          self.wsmap[word][di] = self.idf[word] * ( twf * (self.k1 + 1) / ( twf + self.k1 * self.hds[di] ) )
        if self.algo == 'l' :
          self.wsmap[word][di] = self.idf[word] * twf * (self.k1 + 1) * ( twf/self.hds[di] + self.delta) / (self.k1 + twf/self.hds[di] + self.delta)
        if self.algo == 'plus' :
          self.wsmap[word][di] = self.idf[word] * (self.delta + ( twf * (self.k1 + 1) / ( twf + self.k1 * self.hds[di] ) ))

    ### End of createwsmap()


  # get a list of scores for every document
  def get_scores( self, query ) :
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
  def topk( self, query, k=None ) :
    docscores = self.get_scores( query )
    sisc = [ [i,s] for i,s in enumerate(docscores) ]
    sisc.sort(key=lambda x:x[1],reverse=True)
    if k : sisc = sisc[:k]
    return sisc


  # updating index by adding documents
  def add_documents( self, documents ) :
    new_tokenized_documents = [ self.tokenizer_function( document ) for document in documents ]
    self.corpus_len += len(documents)

    # loop new documents
    for tokenized_document in new_tokenized_documents:
      # doc lengths and total word count
      self.doc_lens.append(len(tokenized_document))
      self.total_word_count += len(tokenized_document)
      # word frequencies in this document
      frequencies = {}
      for word in tokenized_document:
        if word not in frequencies:
          frequencies[word] = 0
        frequencies[word] += 1
      self.word_freqs.append(frequencies)
      # number of documents with word count
      for word, freq in frequencies.items():
        try:
          self.word_docs_count[word] += 1
        except KeyError:
          self.word_docs_count[word] = 1
    
    # create wsmap
    self.createwsmap()

    ### End of add_documents()


  # updating index by deleting documents
  def delete_documents( self, document_ids ) :
    self.corpus_len -= len(document_ids)
    document_ids.sort(reverse=True) # important to delete documents in reverse order

    # loop document-to-delete ids
    for d_id in document_ids :
      if d_id < len(self.doc_lens) :
        # doc lengths and total word count
        self.total_word_count -= self.doc_lens[d_id]
        del self.doc_lens[d_id]
        # word frequencies
        for word in self.word_freqs[d_id] :
          self.word_docs_count[word] -= 1
          # number of documents with word count
          if self.word_docs_count[word] < 1 : del self.word_docs_count[word]
        del self.word_freqs[d_id]
    
    # create wsmap
    self.createwsmap()

    ### End of delete_documents()


  # updating index by changing documents
  def update_documents( self, document_ids, documents ) :
    new_tokenized_documents = [ self.tokenizer_function( document ) for document in documents ]

    # loop document-to-update ids
    for i, u_id in enumerate(document_ids) :
      if i < len(new_tokenized_documents) and u_id < len(self.doc_lens) :
        # doc lengths and total word count
        self.total_word_count -= self.doc_lens[u_id]
        self.doc_lens[u_id] = len(new_tokenized_documents[i])
        self.total_word_count += len(new_tokenized_documents[i])
        # word frequencies : remove old
        for word in self.word_freqs[u_id] :
          self.word_docs_count[word] -= 1
          if self.word_docs_count[word] < 1 : del self.word_docs_count[word]
        # word frequencies : add new
        self.word_freqs[u_id] = {}
        for word in new_tokenized_documents[i]:
          if word not in self.word_freqs[u_id]:
            self.word_freqs[u_id][word] = 0
          self.word_freqs[u_id][word] += 1
        # number of documents with word count
        for word, freq in self.word_freqs[u_id].items():
          try:
            self.word_docs_count[word] += 1
          except KeyError:
            self.word_docs_count[word] = 1

    # create wsmap
    self.createwsmap()

    ### End of update_documents()


  ### End of class BM25opt
