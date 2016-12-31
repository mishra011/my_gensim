
from bow import bow
from tfidf import tfidf_gensim
def main():
	b = bow()
	t = tfidf_gensim()
	path = "/home/deepak/Work/D2V/docs/"
	docLabels =  b.get_docLabels(path)
	#print docLabels
	documents = b.read_doc2list(path, docLabels)
	#print documents
	texts = b.remove_stopwords(documents)
	#print texts[1]
	texts = b.freq_count_of_docs(texts)
	#print texts
	#b.save_dictionary(texts, "mydictionary.dict")
	
	dictionary = b.load_dictionary("mydictionary.dict")
	v = b.get_vector(dictionary, "hello to my new world")
	#print v
	x = b.word_mapings(dictionary)
	#print x
	corpus = b.get_corpus(texts, dictionary)
	#print corpus

	numpy_matrix = b.corpus_2_matrix(corpus)

	#print (corpus == numpy_matrix)
	
	tfidf = t.get_tfidf_count(corpus)
	#print tfidf
	d = t.get_tfidf_of_doc(corpus, 2)
	#print d
	
	name = "tfidf_model"
	#t.save_tfidf_model(tfidf, name)
	#vec_lsi = b.get_vec_lsi(corpus, dictionary)
	#print vec_lsi
	
	s = t.get_sim(tfidf, corpus,v)
	print s
	




if __name__ == "__main__":
	main()