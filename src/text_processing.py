from pyvi import ViTokenizer

def preprocess_user_text(content):
    # Tiền xử lý
    contents_parsed = content.lower()
    contents_parsed = contents_parsed.replace('\n', ' ')
    contents_parsed = contents_parsed.strip()

    # Tách câu theo dấu chấm
    sentences = [s.strip() for s in contents_parsed.split('.') if s.strip()]

    # Tách từ trong từng câu bằng ViTokenizer
    tokenized_sentences = [ViTokenizer.tokenize(sen).split() for sen in sentences]

    return tokenized_sentences