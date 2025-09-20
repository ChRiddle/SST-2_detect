import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch

def GetSST_2Data(path):
    if 'test' not in path:
        data = pd.read_csv(path, sep='\t')
        texts = data['sentence']
        labels = data['label']
    else:
        texts = []
        labels = []
        data = pd.read_csv(path, sep='\t')
        data = data['sentence']
        for text in data:
            texts.append(text[2:])
            labels.append(1 if text[0] == '1' else 0)
    return texts,labels

#构造输入数据
class GetDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # 将文本转换为索引序列
        tokens = (text.lower()).split()
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in tokens]
        # 填充或截断序列
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices = indices + [self.word2idx['<PAD>']] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)