from pyvi import ViTokenizer

def preprocess_user_text(content):
    contents_parsed = content.lower()
    contents_parsed = contents_parsed.replace('\n', ' ')
    contents_parsed = contents_parsed.strip()

    # Tách câu theo dấu chấm
    sentences = [s.strip() for s in contents_parsed.split('.') if s.strip()]

    # Tách từ trong từng câu
    tokenized_sentences = [ViTokenizer.tokenize(sen).split() for sen in sentences]

    return tokenized_sentences

# Nếu chạy trực tiếp thì cho người dùng nhập
if __name__ == "__main__":
    content = input("Nhập vào đoạn văn bản tiếng Việt :")
    result = preprocess_user_text(content)
    print("\n Kết quả tách từ :")
    for sent in result:
        print(sent)
