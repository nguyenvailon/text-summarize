from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import os # Thêm import os để xử lý đường dẫn

# Cập nhật import để trỏ đến text_processing.py trong cùng package src
from .text_processing import preprocess_user_text # Hàm tiền xử lý bạn viết

# Lấy đường dẫn thư mục gốc của dự án
# Từ 'src/word_embedding.py', ta đi lên 1 cấp ('src') để đến thư mục gốc của dự án.
project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Load mô hình đã huấn luyện
# Cập nhật đường dẫn để trỏ tới models/w2v.model
model = KeyedVectors.load(os.path.join(project_root, 'models', 'w2v.model'))

def get_sentence_vectors(user_input):
    processed_sentences = preprocess_user_text(user_input)
    sentence_vectors = []
    valid_sentence_indexes = [] # Khởi tạo danh sách để lưu các chỉ số câu hợp lệ

    # Lặp qua các câu đã xử lý cùng với chỉ số của chúng
    for idx, tokens in enumerate(processed_sentences):
        # Lọc các từ có trong mô hình để tránh lỗi KeyError
        vectors = [model[word] for word in tokens if word in model]
        if vectors:
            avg_vector = np.mean(vectors, axis=0)
            sentence_vectors.append(avg_vector)
            valid_sentence_indexes.append(idx) # Lưu chỉ số của câu nếu nó có vector hợp lệ
    
    # Trả về cả vector câu, chỉ số hợp lệ và các câu đã xử lý (tokens)
    return sentence_vectors, valid_sentence_indexes, processed_sentences
