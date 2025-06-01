
from .word_embedding import get_sentence_vectors
from underthesea import sent_tokenize 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

def summarize_text_k_means(original_text, num_sentences_desired):
    """
    Tóm tắt văn bản sử dụng K-Means clustering.

    Args:
        original_text (str): Văn bản gốc cần tóm tắt.
        num_sentences_desired (int): Số câu tóm tắt mong muốn.

    Returns:
        str: Văn bản tóm tắt.
        list: Danh sách các câu gốc đã được tách (sentences_raw).
        str: Thông báo lỗi nếu có.
    """
    sentences_raw = sent_tokenize(original_text)
    
    sentence_vectors, valid_indexes, _ = get_sentence_vectors(original_text)

    if not sentence_vectors:
        return None, sentences_raw, "Không có câu nào có từ trong mô hình Word2Vec. Vui lòng kiểm tra lại văn bản hoặc mô hình."
    
    if num_sentences_desired > len(sentence_vectors):
        return None, sentences_raw, f"Số câu tóm tắt ({num_sentences_desired}) không thể lớn hơn số câu hợp lệ trong văn bản ({len(sentence_vectors)})."

    X = np.array(sentence_vectors)

    kmeans = KMeans(n_clusters=num_sentences_desired, random_state=42, n_init=10)
    kmeans.fit(X)

    avg = []
    for j in range(num_sentences_desired):
        idx = np.where(kmeans.labels_ == j)[0]
        if len(idx) > 0: 
            avg_index = np.mean(idx)
            avg.append(avg_index)
        else:
            avg.append(float('inf')) 

    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)

    valid_order_indices = [i for i, val in enumerate(avg) if val != float('inf')]
    ordering = sorted(valid_order_indices, key=lambda k: avg[k])

    summary_sentences = [sentences_raw[valid_indexes[closest[idx]]] for idx in ordering]
    summary_text = ' '.join(summary_sentences)
    
    return summary_text, sentences_raw, None 

if __name__ == "__main__":
    
    user_input = input("Nhập văn bản tiếng Việt:\n")

    _, valid_indexes, _ = get_sentence_vectors(user_input) 

    if not valid_indexes: 
        print("⚠️ Không có câu nào có từ trong mô hình Word2Vec. Vui lòng kiểm tra lại văn bản hoặc mô hình.")
        exit()

    while True:
        try:
            n_clusters = int(input(f"Nhập số câu tóm tắt (tối đa {len(valid_indexes)}): "))
            if 1 <= n_clusters <= len(valid_indexes):
                break
            else:
                print(f"Vui lòng nhập số hợp lệ trong khoảng từ 1 đến {len(valid_indexes)}.")
        except ValueError:
            print("Vui lòng nhập số nguyên hợp lệ.")
    
    summary, _, error = summarize_text_k_means(user_input, n_clusters)
    
    if error:
        print(f"Lỗi: {error}")
    else:
        print("\nKết quả tóm tắt:\n")
        print(summary)
