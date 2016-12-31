import gensim
from gensim import corpora, models, similarities
from os import listdir
from os.path import isfile, join
import numpy as np
from collections import defaultdict
class bow():
	def get_docLabels(self, path):
		
		docLabels = []
		docLabels = [f for f in listdir(path) if 
			f.endswith('.txt')]
		return docLabels


		#print docLabels
	def read_doc2list(self, path, docLabels):
		documents = []
		for doc in docLabels:
			documents.append(open(path + doc).read())

		return documents

	def remove_stopwords(self, documents):
		stoplist = set('for a of the and to in'.split())
		texts = [[word for word in document.lower().split() if word not in stoplist]
				for document in documents]

		return texts

	def freq_count_of_docs(self, texts):
		frequency = defaultdict(int)
		for text in texts:
		    for token in text:
		        frequency[token] += 1

		texts = [[token for token in text if frequency[token] > 1]
		          for text in texts]

		from pprint import pprint  # pretty-printer
		return texts

	def save_dictionary(self, texts, name):
		dictionary = corpora.Dictionary(texts)
		dictionary.save(name)
		print "done"

		#print(dictionary.token2id)
	def load_dictionary(seld, name):
		dictionary = corpora.Dictionary.load(name)
		return dictionary


	def word_mapings(self, dictionary):
		
		return dictionary.token2id
		
	def get_vector(self, dictionary, new_doc):
		#new_doc = "I want to be great in Machine Learning"
		new_vec = dictionary.doc2bow(new_doc.lower().split())
		return new_vec

	def get_corpus(self, texts, dictionary):
		corpus = [dictionary.doc2bow(text) for text in texts]
		
		return corpus
		
		
	def corpus_2_matrix(self,corpus):
		
		#numpy_matrix = np.random.randint(10, size=[5,2])  # random matrix as an example
		numpy_matrix = np.matrix(corpus)
		return numpy_matrix
		
		
	def get_vec_lsi(self, corpus, dictionary):
		corpora.MmCorpus.serialize('mydict.mm', corpus)
		lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=12)
		vec_bow = corpus
		vec_lsi = lsi[vec_bow]
		return vec_lsi