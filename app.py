import os
from flask import Flask, render_template, request, g, redirect, url_for, flash
import sqlite3
import numpy as np 
from src.summarizer import summarize_text_k_means 
import config

app = Flask(__name__)
app.config.from_object(config) 

def get_db():
    """Kết nối đến database SQLite."""
    if 'db' not in g:
        g.db = sqlite3.connect(
            app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row 
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

app.teardown_appcontext(close_db)

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
                
                summary_text, _, error_message = summarize_text_k_means(original_text, num_sentences)

                if error_message:
                    flash(f'Lỗi: {error_message}', 'danger')
                else:
                    
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

    
    db = get_db()
    recent_summaries = db.execute(
        'SELECT id, original_text, summary_text, created_at FROM summaries ORDER BY created_at DESC LIMIT 5'
    ).fetchall()

    return render_template('index.html', 
                           summary=summary_text, 
                           original_text=original_text, 
                           error=error_message,
                           recent_summaries=recent_summaries)

@app.route('/delete/<int:id>', methods=('POST',))
def delete_summary(id):
    """Xóa một bản tóm tắt khỏi database."""
    db = get_db()
    cursor = db.execute('DELETE FROM summaries WHERE id = ?', (id,))
    db.commit()
    if cursor.rowcount == 0:
        flash('Không tìm thấy bản tóm tắt để xóa.', 'danger')
    else:
        flash('Bản tóm tắt đã được xóa thành công!', 'success')
    return redirect(url_for('index'))


@app.cli.command('init-db')
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    print('Initialized the database.')

if __name__ == '__main__':
    pass 
