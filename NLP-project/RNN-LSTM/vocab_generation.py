import re
import os
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from collections import defaultdict


# Text segmentation rules
def text_split(content: str) -> List[str]:
    """

    Args:
        content: Text to split
    """
    # add a blank space before .!?
    content = re.sub(r"([.!?])", r" \1 ", content)
    # remove letters beyond a-z/A-Z/.!?
    content = re.sub(r"[^a-zA-Z.!?]+", r" ", content)
    token = [i.strip().lower() for i in content.split()]
    return token


class Vocabulary:
    # unknown character
    UNK_TAG = "UNK"
    # padding character
    PAD_TAG = "PAD"
    UNK = 0
    PAD = 1

    def __init__(self):
        self.inverse_vocab = None
        self.vocabulary = {self.UNK_TAG: self.UNK, self.PAD_TAG: self.PAD}
        # frequency of words
        self.count = defaultdict(int)

    def fit(self, sentence_: List[str]):
        for word in sentence_:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocabulary(self, min=0, max=None, max_size=None) -> Tuple[dict, dict]:
        # Word frequencies greater than or less than a certain value are discarded
        if min is not None:
            self.count = {word: value for word, value in self.count.items() if value > min}
        if max is not None:
            self.count = {word: value for word, value in self.count.items() if value < max}
        if max_size is not None:
            raw_size = len(self.count.items())
            vocab_size = max_size if raw_size > max_size else raw_size
            self.count = dict(sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:vocab_size])

        # Create a word list: token -> index
        for word in self.count:
            self.vocabulary[word] = len(self.vocabulary)
        # Word list reverse: index -> token
        self.inverse_vocab = dict(zip(self.vocabulary.values(), self.vocabulary.keys()))

        return self.vocabulary, self.inverse_vocab

    def __len__(self):
        return len(self.vocabulary)


def plot_word_frequency(word_count_dict, hist_size=100):
    words = list(word_count_dict.keys())[:hist_size]
    frequencies = list(word_count_dict.values())[:hist_size]
    plt.figure(figsize=(10, 6))
    plt.bar(words, frequencies)
    plt.title('Word Frequency')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=75)
    path_out = 'word_frequency.jpg'
    plt.savefig(path_out)
    print(f'save word frequency: {path_out}')


if __name__ == "__main__":
    max_size = 20000
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, r'aclImdb_v1\aclImdb\train')
    out_dir = os.path.join(BASE_DIR, 'word_list_result')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    vocab_path = os.path.join(out_dir, 'aclImdb_vocab.npy')
    vocab_inv_path = os.path.join(out_dir, 'aclImdb_inverse_vocab.npy')

    word_freq = Vocabulary()
    data_path_list = [os.path.join(path, "pos"), os.path.join(path, "neg")]
    for data_path in data_path_list:
        file_paths = [os.path.join(data_path, file_name) for file_name in os.listdir(data_path) if
                      file_name.endswith("txt")]
        for file_path in tqdm(file_paths):
            sentence = text_split(open(file_path, encoding='utf-8').read())
            word_freq.fit(sentence)
    # 2 is unk and pad
    vocab, inverse_vocab = word_freq.build_vocabulary(max_size=(max_size - 2))

    np.save(vocab_path, vocab)
    np.save(vocab_inv_path, inverse_vocab)

    # print(vocab, inverse_vocab)

    word_count = word_freq.count
    plot_word_frequency(word_count)