from pyvi import ViTokenizer

content = "Đây là câu thứ nhất. Đây là câu thứ hai.\nĐây là câu thứ ba!"
# Tiền xử lý
contents_parsed = content.lower()
contents_parsed = contents_parsed.replace('\n', ' ')
contents_parsed = contents_parsed.strip()

# Tách câu đơn giản theo dấu chấm
sentences = [s.strip() for s in contents_parsed.split('.') if s.strip()]

# Tách từ trong từng câu bằng ViTokenizer

tokenized_sentences = [ViTokenizer.tokenize(sen).split() for sen in sentences]
for sent in tokenized_sentences:
    print(sent)
