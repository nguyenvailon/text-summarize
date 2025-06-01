from pyvi import ViTokenizer

def preprocess_user_text(text_input):
    # Tiền xử lý
    contents_parsed = text_input.lower()
    contents_parsed = contents_parsed.replace('\n', ' ')
    contents_parsed = contents_parsed.strip()

    # Tách câu đơn giản theo dấu chấm
    sentences_raw = contents_parsed.split('.')
    sentences = [s.strip() for s in sentences_raw if s.strip()] # Filter out empty strings resulting from split

    # Tách từ trong từng câu bằng ViTokenizer
    tokenized_sentences = []
    for sen in sentences:
        # ViTokenizer.tokenize returns a string with words separated by underscores.
        # .split() then splits this string into a list of tokens.
        tokens = ViTokenizer.tokenize(sen).split()
        if tokens: # Ensure we don't add empty lists if a sentence had no processable tokens
            tokenized_sentences.append(tokens)
            
    return tokenized_sentences

# This part is for testing the script directly (optional)
if __name__ == "__main__":
    content_example = "Đây là câu thứ nhất. Đây là câu thứ hai.\nĐây là câu thứ ba!"
    print(f"Original content:\n{content_example}\n")
    
    processed_example = preprocess_user_text(content_example)
    print("Tokenized sentences:")
    for sent_tokens in processed_example:
        print(sent_tokens)

    content_example_2 = "Chào bạn! Thế nào rồi?" # No period
    print(f"\nOriginal content 2:\n{content_example_2}\n")
    processed_example_2 = preprocess_user_text(content_example_2)
    print("Tokenized sentences (example 2):")
    for sent_tokens in processed_example_2:
        print(sent_tokens)