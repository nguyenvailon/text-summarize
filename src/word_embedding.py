from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import os

from .text_processing import preprocess_user_text

project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

model = KeyedVectors.load(os.path.join(project_root, 'models', 'w2v.model'))

def get_sentence_vectors(user_input):
    processed_sentences = preprocess_user_text(user_input)
    sentence_vectors = []

    for tokens in processed_sentences:
        vectors = [model[word] for word in tokens if word in model]
        if vectors:
            avg_vector = np.mean(vectors, axis=0)
            sentence_vectors.append(avg_vector)

    return sentence_vectors
