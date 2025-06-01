from word_embedding import get_sentence_vectors
<<<<<<< HEAD
<<<<<<< HEAD
from underthesea import sent_tokenize
=======
from underthesea import sent_tokenize # Dùng underthesea cho tiếng Việt
>>>>>>> 83c1b7c5599fe9ced85fdeb8dc711c50b6e71a2e
=======
from underthesea import sent_tokenize # Dùng underthesea cho tiếng Việt
>>>>>>> 83c1b7c5599fe9ced85fdeb8dc711c50b6e71a2e
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

<<<<<<< HEAD
user_input = input("Nhập văn bản tiếng Việt:\n")

sentences = sent_tokenize(user_input)

=======
# Nhập văn bản tiếng Việt
user_input = input("Nhập văn bản tiếng Việt:\n")

# Tách câu gốc tiếng Việt
sentences = sent_tokenize(user_input)

# Lấy vector câu, chỉ số câu hợp lệ, câu đã xử lý (tokens)
>>>>>>> 83c1b7c5599fe9ced85fdeb8dc711c50b6e71a2e
sentence_vectors, valid_indexes, _ = get_sentence_vectors(user_input)

if not sentence_vectors:
    print("⚠️ Không có câu nào có từ trong mô hình Word2Vec.")
    exit()

X = np.array(sentence_vectors)

<<<<<<< HEAD
=======
# Nhập số câu tóm tắt từ người dùng, tối đa không quá số câu có thể tóm tắt
>>>>>>> 83c1b7c5599fe9ced85fdeb8dc711c50b6e71a2e
while True:
    try:
       n_clusters = int(input(f"Nhập số câu tóm tắt (tối đa {len(sentence_vectors)}): "))
       if 1 <= n_clusters <= len(sentence_vectors):
        break
       else:
        print("Vui lòng nhập số hợp lệ trong khoảng từ 1 đến số câu hợp lệ.")
    except ValueError:
        print("Vui lòng nhập số nguyên hợp lệ.")

<<<<<<< HEAD
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

=======
# Khởi tạo và chạy KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)

# Tính trung bình chỉ số câu trong mỗi cụm để sắp xếp tóm tắt đúng thứ tự
>>>>>>> 83c1b7c5599fe9ced85fdeb8dc711c50b6e71a2e
avg = []
for j in range(n_clusters):
   idx = np.where(kmeans.labels_ == j)[0]
   avg_index = np.mean(idx)
   avg.append(avg_index)

<<<<<<< HEAD
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)

ordering = sorted(range(n_clusters), key=lambda k: avg[k])

=======
# Tìm câu gần tâm cụm nhất
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)

# Sắp xếp cụm theo thứ tự xuất hiện của câu trong văn bản (để tóm tắt giữ đúng trật tự)
ordering = sorted(range(n_clusters), key=lambda k: avg[k])

# Tạo bản tóm tắt từ câu đại diện của mỗi cụm
>>>>>>> 83c1b7c5599fe9ced85fdeb8dc711c50b6e71a2e
summary = ' '.join([sentences[valid_indexes[closest[idx]]] for idx in ordering])

print("\nKết quả tóm tắt:\n")
print(summary)