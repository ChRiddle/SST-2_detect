import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import f1_score,accuracy_score
import sys
sys.path.append('..')
from My_SST2_Process import DataProcess as DP

#设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextDataset(Dataset):
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




#定义模型
class BiLSTM(nn.Module):
    def __init__(self, vocab_size,embedding_size ,hidden_size,bidirectional,num_layers, num_classes,dropout ,embedding_weight=None):
        super(BiLSTM, self).__init__()
        if embedding_weight is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_weight,freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size,embedding_size)
        # self.layers = num_layers
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers,bidirectional=bidirectional,batch_first=True)
        # 层归一化
        self.layer_norm = nn.LayerNorm(2*hidden_size)
        self.fc = nn.Linear(2*hidden_size, num_classes) # *2 因为双向
        self.dropout = nn.Dropout(dropout)
    def forward(self, X):
        X = self.embedding(X)
        out,(H,C) = self.lstm(X)
        # out = torch.cat((H[-1,:,:],H[-2,:,:]),dim=1)
        out = self.layer_norm(out)
        out = out.mean(dim=1)
        out = self.fc(self.dropout(out))
        return out



def evaluate_model(model, iterator):
    model.eval()
    pre_all = []
    lable_all = []
 
    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            text, labels = text.to(device), labels.to(device)
            predictions = model(text)
            preds = predictions.argmax(dim=1)

            # 统计所有预估和真实标签
            pre_all += preds.tolist()
            lable_all += labels.tolist()
    # 计算F1 scores、acc
    f1 = f1_score(pre_all, lable_all,average='binary')
    acc = accuracy_score(lable_all,pre_all)
    return f1, acc







#加载测试数据集
lines_test,lines_test_labels = DP.GetSST_2Data('/home/user_home/qiankunjie/project/First-exam/SST-2/test.tsv')


lines_train,lines_train_labels = DP.GetSST_2Data('/home/user_home/qiankunjie/project/First-exam/SST-2/train.tsv')
lines_dev,lines_dev_labels = DP.GetSST_2Data('/home/user_home/qiankunjie/project/First-exam/SST-2/dev.tsv')

word_sequence_train = " ".join(lines_train).split()
word_sequence_dev = " ".join(lines_dev).split()
    # word_sequence_test = " ".join(lines_test).split()
word_sequence = word_sequence_dev + word_sequence_train
vocab = ['<PAD>', '<UNK>'] + sorted(set(word_sequence))
vocab_size = len(vocab)
    # 构建词汇映射表
word2idx = {w: i for i, w in enumerate(vocab)}





test_dataset = TextDataset(lines_test, lines_test_labels, word2idx, 55)
test_loader = DataLoader(test_dataset, batch_size=32,shuffle=False)

#导入模型
para = torch.load('/home/user_home/qiankunjie/project/First-exam/BiLSTM/best_model.pt')
model = BiLSTM(vocab_size,300,512,True,2,2,0.1).to(device)
model.load_state_dict(para)
#测试结果
f1, acc= evaluate_model(model, test_loader)
#输出结果
print("BiLSTM")
print(f"测试集数据量:{len(lines_test)}")
print(f"f1_score: {f1:.3}\taccuracy: {100*acc:.3}%")
