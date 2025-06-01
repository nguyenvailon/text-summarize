import os

# Lấy đường dẫn thư mục gốc của dự án
project_root = os.path.dirname(os.path.abspath(__file__))

# Cấu hình database SQLite
DATABASE = os.path.join(project_root, 'database', 'summaries.db')

# Khóa bí mật cho Flask sessions (thay đổi thành một chuỗi ngẫu nhiên và mạnh mẽ)
SECRET_KEY = 'your_super_secret_key_here_change_this' # Rất quan trọng: thay đổi khóa này trong môi trường production!

# Chế độ Debug (chỉ bật trong môi trường phát triển)
DEBUG = True

# Đường dẫn đến mô hình Word2Vec (nếu cần tải trực tiếp trong app.py, nhưng hiện tại đã tải trong word_embedding.py)
# W2V_MODEL_PATH = os.path.join(project_root, 'models', 'w2v.model')