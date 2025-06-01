# Cập nhật import để trỏ đến word_embedding.py trong cùng package src
from .word_embedding import get_sentence_vectors
from underthesea import sent_tokenize # Dùng underthesea cho tiếng Việt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Encapsulate the core summarization logic into a function
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
    
    # Lấy vector câu, chỉ số câu hợp lệ, câu đã xử lý (tokens)
    sentence_vectors, valid_indexes, _ = get_sentence_vectors(original_text)

    if not sentence_vectors:
        return None, sentences_raw, "Không có câu nào có từ trong mô hình Word2Vec. Vui lòng kiểm tra lại văn bản hoặc mô hình."
    
    # Đảm bảo số câu tóm tắt không vượt quá số câu hợp lệ
    if num_sentences_desired > len(sentence_vectors):
        return None, sentences_raw, f"Số câu tóm tắt ({num_sentences_desired}) không thể lớn hơn số câu hợp lệ trong văn bản ({len(sentence_vectors)})."

    X = np.array(sentence_vectors)

    # Khởi tạo và chạy KMeans
    # Thêm n_init=10 để tránh cảnh báo trên sklearn >= 1.2
    kmeans = KMeans(n_clusters=num_sentences_desired, random_state=42, n_init=10)
    kmeans.fit(X)

    # Tính trung bình chỉ số câu trong mỗi cụm để sắp xếp tóm tắt đúng thứ tự
    avg = []
    for j in range(num_sentences_desired):
        idx = np.where(kmeans.labels_ == j)[0]
        if len(idx) > 0: # Đảm bảo cụm không rỗng
            avg_index = np.mean(idx)
            avg.append(avg_index)
        else:
            avg.append(float('inf')) # Gán giá trị lớn để nó không ảnh hưởng đến thứ tự

    # Tìm câu gần tâm cụm nhất
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)

    # Sắp xếp cụm theo thứ tự xuất hiện của câu trong văn bản (để tóm tắt giữ đúng trật tự)
    valid_order_indices = [i for i, val in enumerate(avg) if val != float('inf')]
    ordering = sorted(valid_order_indices, key=lambda k: avg[k])

    # Tạo bản tóm tắt từ câu đại diện của mỗi cụm
    summary_sentences = [sentences_raw[valid_indexes[closest[idx]]] for idx in ordering]
    summary_text = ' '.join(summary_sentences)
    
    return summary_text, sentences_raw, None # Trả về bản tóm tắt, câu gốc, và không có lỗi


# Phần code này CHỈ chạy khi bạn chạy file summarizer.py trực tiếp (ví dụ: python src/summarizer.py)
# Nó sẽ KHÔNG chạy khi app.py import summarizer.py
if __name__ == "__main__":
    # Nhập văn bản tiếng Việt từ console
    user_input = input("Nhập văn bản tiếng Việt:\n")

    # Lấy vector câu, chỉ số câu hợp lệ, câu đã xử lý (tokens)
    _, valid_indexes, _ = get_sentence_vectors(user_input) # Gọi để kiểm tra số câu hợp lệ

    if not valid_indexes: # Kiểm tra valid_indexes thay vì sentence_vectors
        print("⚠️ Không có câu nào có từ trong mô hình Word2Vec. Vui lòng kiểm tra lại văn bản hoặc mô hình.")
        exit()

    # Nhập số câu tóm tắt từ người dùng, tối đa không quá số câu có thể tóm tắt
    while True:
        try:
            n_clusters = int(input(f"Nhập số câu tóm tắt (tối đa {len(valid_indexes)}): "))
            if 1 <= n_clusters <= len(valid_indexes):
                break
            else:
                print(f"Vui lòng nhập số hợp lệ trong khoảng từ 1 đến {len(valid_indexes)}.")
        except ValueError:
            print("Vui lòng nhập số nguyên hợp lệ.")
    
    # Gọi hàm tóm tắt chính
    summary, _, error = summarize_text_k_means(user_input, n_clusters)
    
    if error:
        print(f"Lỗi: {error}")
    else:
        print("\nKết quả tóm tắt:\n")
        print(summary)
