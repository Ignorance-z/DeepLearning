import os
import sys
import time
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import platform
from torch.utils.data import DataLoader
import torch.nn.functional as F

BASE_DIR = os.path.dirname(__file__)
PRJ_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PRJ_DIR)


class LSTMTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, hidden_size, labels, bidirectional, **kwargs):
        super(LSTMTextClassifier, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=bidirectional)
        # labels: number of categories
        self.decoder = nn.Linear(4 * hidden_size, labels) if bidirectional else nn.Linear(2 * hidden_size, labels)
