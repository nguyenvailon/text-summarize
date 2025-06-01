from pyvi import ViTokenizer

def preprocess_user_text(content):
<<<<<<< HEAD
=======
    # Tiền xử lý
>>>>>>> 83c1b7c5599fe9ced85fdeb8dc711c50b6e71a2e
    contents_parsed = content.lower()
    contents_parsed = contents_parsed.replace('\n', ' ')
    contents_parsed = contents_parsed.strip()

<<<<<<< HEAD
    sentences = [s.strip() for s in contents_parsed.split('.') if s.strip()]

=======
    # Tách câu theo dấu chấm
    sentences = [s.strip() for s in contents_parsed.split('.') if s.strip()]

    # Tách từ trong từng câu bằng ViTokenizer
>>>>>>> 83c1b7c5599fe9ced85fdeb8dc711c50b6e71a2e
    tokenized_sentences = [ViTokenizer.tokenize(sen).split() for sen in sentences]

    return tokenized_sentences