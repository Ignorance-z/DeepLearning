import sys
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(__file__)
PRJ_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PRJ_DIR)

from vocabulary import text_split


class WordToIdx(object):
    def __init__(self):
        self.PAD_TAG = "PAD"
        self.UNK = 0

    def encode(self, sentence, vocab_dict, max_len=None):
        if max_len is not None:
            if max_len > len(sentence):
                sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))
            if max_len < len(sentence):
                sentence = sentence[:max_len]
        return [vocab_dict.get(word, self.UNK) for word in sentence]

    @staticmethod
    def decode(ws_inverse, indices):
        return [ws_inverse.get(idx) for idx in indices]


class AclImdbDataset(Dataset):
    def __init__(self, root_dir, vocab_path, is_train=True, max_len=200, vocab_building_method='own'):
        sub_dir = "train" if is_train else "test"
        self.data_dir = os.path.join(root_dir, sub_dir)
        self.vocab_path = vocab_path
        self.max_len = max_len

        self.word2idx = WordToIdx()
        self._init_vocab_own() if vocab_building_method == 'own' else self._init_vocab_official()
        self._get_file_info()

    def __getitem__(self, item):
        file_path = self.total_file_path[item]
        label = 0 if os.path.basename(os.path.dirname(file_path)) == "neg" else 1

        # tokenize & encode to index
        token_list = text_split(open(file_path, encoding='utf-8').read())  # split
        token_idx_list = self.word2idx.encode(token_list, self.vocab, self.max_len)

        return np.array(token_idx_list), label

    def __len__(self):
        return len(self.total_file_path)

    def _get_file_info(self):
        self.data_dir_list = [os.path.join(self.data_dir, "pos"), os.path.join(self.data_dir, "neg")]
        self.total_file_path = []
        for dir in self.data_dir_list:
            self.file_name_list = os.listdir(dir)
            self.file_path_list = [os.path.join(dir, filename) for filename in self.file_name_list if
                                   filename.endswith("txt")]
            self.total_file_path.extend(self.file_path_list)

    def _init_vocab_own(self):
        self.vocab = np.load(self.vocab_path, allow_pickle=True).item()

    def _init_vocab_official(self):
        with open(self.vocab_path, 'r', encoding='utf-8') as vocab:
            lines = vocab.read().split('\n')
        self.vocab = {**{'UNK': 0, 'PAD': 1}, **{lines[i - 2]: i for i in range(2, len(lines) + 2)}}


if __name__ == "__main__":
    root_dir = os.path.join(BASE_DIR, '..', r'aclImdb_v1\aclImdb')
    # You can adjust the functions in AclImdbDataset class to use different vocab building methods.
    #
    # To use own vocab(_init_vocab_own), you can use this part
    vocab_path = os.path.join(BASE_DIR, "..", r'word_list_result\aclImdb_vocab.npy')
    # To use official vocab(_init_vocab_official), you can use this part
    # vocab_path = os.path.join(BASE_DIR, "..", r'aclImdb_v1\aclImdb\imdb.vocab')

    # If you are using your own vocab, select 'own' for the vocab_building_method parameter,
    # or 'official' for the official one.
    train_dataset = AclImdbDataset(root_dir, vocab_path, is_train=True, max_len=200, vocab_building_method='own')
    test_dataset = AclImdbDataset(root_dir, vocab_path, is_train=False, max_len=200, vocab_building_method='own')

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    for i, (inputs, targets) in enumerate(train_loader):
        print(i, inputs.shape, inputs, targets.shape, targets)
        break
