import pickle
from gensim.models import Word2Vec
import os # Import os để xử lý đường dẫn

# Lấy đường dẫn thư mục gốc của dự án
project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Cập nhật đường dẫn đọc file pkl từ data/processed
# Tôi sẽ giả định bạn đã đổi tên sentences1.pkl thành sentences.pkl cho nhất quán.
# Nếu bạn muốn giữ nguyên sentences1.pkl, hãy thay 'sentences.pkl' thành 'sentences1.pkl'
sentences_pkl_path = os.path.join(project_root, 'data', 'processed', 'sentences.pkl') # Hoặc 'sentences1.pkl'

sentences = pickle.load(open(sentences_pkl_path, 'rb'))
input_gensim = []
for sen in sentences:
    input_gensim.append(sen.split())

# Cập nhật đường dẫn lưu mô hình vào models/
w2v_model_save_path = os.path.join(project_root, 'models', 'w2v.model')

# Bỏ comment dòng huấn luyện mô hình nếu bạn muốn chạy nó
model = Word2Vec(input_gensim, vector_size=128, window=5, min_count=0, workers=4, sg=1)
model.wv.save(w2v_model_save_path)

# Dòng tải mô hình này có vẻ không cần thiết trong file trainer nếu bạn chỉ huấn luyện.
# Nếu bạn muốn kiểm tra ngay sau khi lưu, có thể giữ lại.
# w2v_model = word2vec.KeyedVectors.load(w2v_model_save_path) # Cần import gensim.models.keyedvectors as word2vec

# Để code gốc của bạn hoạt động, bạn cần import word2vec
import gensim.models.keyedvectors as word2vec
w2v_model = word2vec.KeyedVectors.load(w2v_model_save_path)


vocabulary = []
# for word in w2v_model.index_to_key: # Dòng này bị comment trong code gốc
#    vocabulary.append(word)
print(len(sentences))
print(len(vocabulary))