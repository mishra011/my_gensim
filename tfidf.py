import gensim
from gensim import corpora, models, similarities

class tfidf_gensim():
	def get_tfidf_count(self, corpus):
		tfidf = models.TfidfModel(corpus)
		return tfidf

	def get_tfidf_of_doc(self, tfidf, some_doc_index):
		tfidf_vector = tfidf[some_doc_index]	# some_doc_index is the indec of doc in corpus
		return tfidf_vector

	def save_tfidf_model(self, tfidf, name):
		tfidf.save(name)
		print "Saved"

	def get_sim(self,tfidf, corpus,doc):
		
		index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=10)
		#print index
		sims = index[doc]
		return list(enumerate(sims))

