from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from text_processing import preprocess_user_text  # Import hàm bạn đã viết
# Load mô hình đã huấn luyện
model = KeyedVectors.load("w2v.model")

def get_sentence_vectors(user_input):
    processed_sentences = preprocess_user_text(user_input)
    sentence_vectors = []
    valid_sentence_indexes = []

    for idx, tokens in enumerate(processed_sentences):
        vectors = [model[word] for word in tokens if word in model]
        if vectors:
            avg_vector = np.mean(vectors, axis=0)
            sentence_vectors.append(avg_vector)
            valid_sentence_indexes.append(idx)
    
    return sentence_vectors, valid_sentence_indexes, processed_sentences