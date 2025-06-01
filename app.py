import os
from flask import Flask, render_template, request, g, redirect, url_for, flash
import sqlite3
import numpy as np # Import numpy vì summarizer.py có thể trả về numpy array

# Import hàm tóm tắt từ module summarizer của bạn
# Đảm bảo rằng thư mục src được thêm vào PYTHONPATH hoặc bạn chạy app.py từ thư mục gốc của dự án
# Nếu bạn chạy 'python -m app' từ thư mục gốc, import này sẽ hoạt động
from src.summarizer import get_sentence_vectors, sent_tokenize, KMeans, pairwise_distances_argmin_min

# Import cấu hình
import config

app = Flask(__name__)
app.config.from_object(config) # Tải cấu hình từ config.py

# --- Database functions ---
def get_db():
    """Kết nối đến database SQLite."""
    if 'db' not in g:
        g.db = sqlite3.connect(
            app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row # Cho phép truy cập cột bằng tên
    return g.db

def close_db(e=None):
    """Đóng kết nối database."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """Khởi tạo database từ schema.sql."""
    db = get_db()
    with app.open_resource('database/schema.sql') as f:
        db.executescript(f.read().decode('utf8'))
    print("Database initialized.")

# Đăng ký hàm đóng database khi ứng dụng kết thúc request
app.teardown_appcontext(close_db)

# --- Routes ---
@app.route('/', methods=('GET', 'POST'))
def index():
    summary_text = None
    original_text = ""
    error_message = None
    
    if request.method == 'POST':
        original_text = request.form['original_text']
        num_sentences = request.form['num_sentences']

        if not original_text:
            error_message = "Vui lòng nhập văn bản cần tóm tắt."
        elif not num_sentences.isdigit() or int(num_sentences) <= 0:
            error_message = "Số câu tóm tắt phải là số nguyên dương."
        else:
            num_sentences = int(num_sentences)
            
            try:
                # Tách câu gốc tiếng Việt
                sentences_raw = sent_tokenize(original_text)
                
                # Lấy vector câu, chỉ số câu hợp lệ, câu đã xử lý (tokens)
                # Hàm get_sentence_vectors trả về processed_sentences (tokens) làm tham số thứ 3
                sentence_vectors, valid_indexes, _ = get_sentence_vectors(original_text)

                if not sentence_vectors:
                    error_message = "Không có câu nào có từ trong mô hình Word2Vec. Vui lòng kiểm tra lại văn bản hoặc mô hình."
                elif num_sentences > len(sentence_vectors):
                    error_message = f"Số câu tóm tắt ({num_sentences}) không thể lớn hơn số câu hợp lệ trong văn bản ({len(sentence_vectors)})."
                else:
                    X = np.array(sentence_vectors)

                    # Khởi tạo và chạy KMeans
                    # Thêm n_init=10 để tránh cảnh báo trên sklearn >= 1.2
                    kmeans = KMeans(n_clusters=num_sentences, random_state=42, n_init=10) 
                    kmeans.fit(X)

                    # Tính trung bình chỉ số câu trong mỗi cụm để sắp xếp tóm tắt đúng thứ tự
                    avg = []
                    for j in range(num_sentences):
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

                    # Lưu vào database
                    db = get_db()
                    db.execute(
                        'INSERT INTO summaries (original_text, summary_text) VALUES (?, ?)',
                        (original_text, summary_text)
                    )
                    db.commit()
                    flash('Văn bản đã được tóm tắt và lưu thành công!', 'success')

            except Exception as e:
                error_message = f"Đã xảy ra lỗi trong quá trình tóm tắt: {e}"
                flash(f'Lỗi: {e}', 'danger')

    # Lấy 5 bản tóm tắt gần đây nhất từ database để hiển thị
    db = get_db()
    recent_summaries = db.execute(
        'SELECT original_text, summary_text, created_at FROM summaries ORDER BY created_at DESC LIMIT 5'
    ).fetchall()

    return render_template('index.html', 
                           summary=summary_text, 
                           original_text=original_text, 
                           error=error_message,
                           recent_summaries=recent_summaries)

# --- Command line utility for database initialization ---
@app.cli.command('init-db')
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    print('Initialized the database.')

if __name__ == '__main__':
    # Để chạy ứng dụng từ thư mục gốc, bạn có thể dùng lệnh: flask run
    # Hoặc, nếu bạn muốn chạy trực tiếp file này (chỉ để phát triển):
    # app.run(debug=True)
    pass # Để Flask CLI quản lý việc chạy app
