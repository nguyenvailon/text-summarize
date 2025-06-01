from pyvi import ViTokenizer

def preprocess_user_text(content):
    contents_parsed = content.lower()
    contents_parsed = contents_parsed.replace('\n', ' ')
    contents_parsed = contents_parsed.strip()

    sentences = [s.strip() for s in contents_parsed.split('.') if s.strip()]

    tokenized_sentences = [ViTokenizer.tokenize(sen).split() for sen in sentences]

    return tokenized_sentences