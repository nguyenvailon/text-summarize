from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from text_processing import preprocess_user_text  # Hàm tiền xử lý bạn viết

model = KeyedVectors.load("w2v.model")

def get_sentence_vectors(user_input):
    processed_sentences = preprocess_user_text(user_input)
    sentence_vectors = []

    for tokens in processed_sentences:
        vectors = [model[word] for word in tokens if word in model]
        if vectors:
            avg_vector = np.mean(vectors, axis=0)
            sentence_vectors.append(avg_vector)

    return sentence_vectors