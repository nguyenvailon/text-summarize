import os
import gensim
from pyvi import ViTokenizer
from tqdm import tqdm
import pickle

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def get_data(folder):
    sentences = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding="utf-16") as f:
                lines = f.readlines()
                for line in lines:
                    sens = line.split('.')
                    for sen in sens:
                        if len(sen) > 10:
                            sen = gensim.utils.simple_preprocess(sen)
                            sen = ' '.join(sen)
                            sen = ViTokenizer.tokenize(sen)
                            sentences.append(sen)
    return sentences

# Cập nhật đường dẫn tới thư mục data/raw
train_paths = [
    os.path.join(dir_path, 'data', 'raw', '10Topics', 'Ver1.1', 'Train_Full'),
    os.path.join(dir_path, 'data', 'raw', '10Topics', 'Ver1.1', 'Test_Full'),
    os.path.join(dir_path, 'data', 'raw', '27Topics', 'Ver1.1', 'new test'),
    os.path.join(dir_path, 'data', 'raw', '27Topics', 'Ver1.1', 'new train')
]

sentences = []
for path in tqdm(train_paths):
    sens = get_data(path)
    sentences.extend(sens)

print(len(sentences))

# Cập nhật đường dẫn để lưu file pkl vào data/processed
pickle.dump(sentences, open(os.path.join(dir_path, 'data', 'processed', 'sentences.pkl'), 'wb'))