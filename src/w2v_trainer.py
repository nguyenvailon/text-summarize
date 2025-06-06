import pickle
from gensim.models import Word2Vec
import os # Import os để xử lý đường dẫn
import gensim.models.keyedvectors as word2vec # Giữ nguyên import này như trong code gốc của bạn

# Lấy đường dẫn thư mục gốc của dự án
project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Cập nhật đường dẫn đọc file pkl từ data/processed
# Giữ nguyên tên file 'sentences.pkl' như code gốc của bạn đã sử dụng trong biến path
sentences_pkl_path = os.path.join(project_root, 'data', 'processed', 'sentences.pkl') # Hoặc 'sentences1.pkl'

sentences = pickle.load(open(sentences_pkl_path, 'rb'))
input_gensim = []
for sen in sentences:
    input_gensim.append(sen.split())

w2v_model_save_path = os.path.join(project_root, 'models', 'w2v.model')

os.makedirs(os.path.dirname(w2v_model_save_path), exist_ok=True)

# Bỏ comment để huấn luyện
# model = Word2Vec(input_gensim, vector_size=128, window=5, min_count=0, workers=4, sg=1)
# model.wv.save(w2v_model_save_path)

w2v_model = word2vec.KeyedVectors.load(w2v_model_save_path)


vocabulary = []
for word in w2v_model.index_to_key:
    vocabulary.append(word)

print(len(sentences))
print(len(vocabulary))
