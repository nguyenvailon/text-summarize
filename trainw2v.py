import pickle
from gensim.models import Word2Vec
sentences = pickle.load(open('./sentences.pkl', 'rb'))
input_gensim = []
for sen in sentences:
    input_gensim.append(sen.split())
# model = Word2Vec(input_gensim, vector_size=128, window=5, min_count=0, workers=4, sg=1)
# model.wv.save("w2v.model")
import gensim.models.keyedvectors as word2vec
w2v_model = word2vec.KeyedVectors.load('w2v.model')
vocabulary = []
# for word in w2v_model.index_to_key:
#     vocabulary.append(word)
print(len(sentences))
print(len(vocabulary))