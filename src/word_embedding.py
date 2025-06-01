from gensim.models.keyedvectors import KeyedVectors
import numpy as np
<<<<<<< HEAD
<<<<<<< HEAD
from text_processing import preprocess_user_text  # Hàm tiền xử lý bạn viết

model = KeyedVectors.load("w2v.model")
=======
=======
>>>>>>> 83c1b7c5599fe9ced85fdeb8dc711c50b6e71a2e
import os # Thêm import os để xử lý đường dẫn

# Cập nhật import để trỏ đến text_processing.py trong cùng package src
from text_processing import preprocess_user_text

# Lấy đường dẫn thư mục gốc của dự án
# Từ 'src/word_embedding.py', ta đi lên 1 cấp ('src') để đến thư mục gốc của dự án.
project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Load mô hình đã huấn luyện
# Cập nhật đường dẫn để trỏ tới models/w2v.model
model = KeyedVectors.load(os.path.join(project_root, 'models', 'w2v.model'))
<<<<<<< HEAD
>>>>>>> 83c1b7c5599fe9ced85fdeb8dc711c50b6e71a2e
=======
>>>>>>> 83c1b7c5599fe9ced85fdeb8dc711c50b6e71a2e

def get_sentence_vectors(user_input):
    processed_sentences = preprocess_user_text(user_input)
    sentence_vectors = []
<<<<<<< HEAD

    for tokens in processed_sentences:
=======
    valid_sentence_indexes = []

    for idx, tokens in enumerate(processed_sentences):
>>>>>>> 83c1b7c5599fe9ced85fdeb8dc711c50b6e71a2e
        vectors = [model[word] for word in tokens if word in model]
        if vectors:
            avg_vector = np.mean(vectors, axis=0)
            sentence_vectors.append(avg_vector)
<<<<<<< HEAD

    return sentence_vectors
=======
            valid_sentence_indexes.append(idx)
    
    return sentence_vectors, valid_sentence_indexes, processed_sentences
>>>>>>> 83c1b7c5599fe9ced85fdeb8dc711c50b6e71a2e
