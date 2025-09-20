import gensim.models
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
import json
import sys
sys.path.append('..')
from My_SST2_Process import DataProcess as DP

#设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# 3. 准备预训练词向量
def load_pretrained_embeddings(word2idx, embedding_dim=300):
    # 创建权重矩阵
    embedding_matrix = np.random.randn(len(word2idx), embedding_dim)

    word2vec = gensim.models.KeyedVectors.load_word2vec_format('/home/user_home/qiankunjie/project/First-exam/GoogleNews-vectors-negative300.bin',binary=True)

    for word, i in word2idx.items():
        if word in word2vec:
            embedding_matrix[i] = word2vec[word]
        elif word.lower() in word2vec:
            embedding_matrix[i] = word2vec[word.lower()]

    return torch.FloatTensor(embedding_matrix)



# 4. 训练函数
def train_model(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    pre_all = []
    lable_all = []
    for batch in iterator:
        optimizer.zero_grad()

        text, labels = batch
        text, labels = text.to(device), labels.to(device)

        predictions = model(text).squeeze(1)
        loss = criterion(predictions, labels)
        preds = predictions.argmax(dim=1)

        #统计所有预估和真实标签
        pre_all += preds.tolist()
        lable_all += labels.tolist()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    # 计算F1 scores
    f1 = f1_score(pre_all, lable_all)
    acc = accuracy_score(pre_all,lable_all)
    return f1, acc, epoch_loss / len(iterator)


# 5. 评估函数
def evaluate_model(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    pre_all = []
    lable_all = []
    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            text, labels = text.to(device), labels.to(device)

            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels)

            preds = predictions.argmax(dim=1)
            correct = (preds == labels)
            acc = correct.sum() / len(correct)

            # 统计所有预估和真实标签
            pre_all += preds.tolist()
            lable_all += labels.tolist()
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    # 计算F1 scores
    f1 = f1_score(pre_all, lable_all)
    return f1, epoch_acc / len(iterator), epoch_loss / len(iterator)


#参数设置
config = {
    "embedding_dim": 300,
    "hidden_dim": 512,
    "output_dim": 2,
    "n_layers": 2,
    "bidirectional": True,
    "dropout": 0.1,
    "max_len":55, #train_data最长句子52个word，dev_data最长句子47个word
    "batchsize" : 32,
    "n_epochs": 20,
    "lr": 1e-3,
    "momentum": 0.9,
    "save_model_path": "./best_model.pt",#模型保存路径
    "training_result": "./result.json",#保存训练过程中产生的结果
    "save_time": 10,#每隔多少轮保存一次训练结果
    "best_dev_acc": 0.5,
    "f1_dev": 0
}


def main():
    # 读取数据
    lines_train,lines_train_labels = DP.GetSST_2Data('/home/user_home/qiankunjie/project/First-exam/SST-2/train.tsv')
    lines_dev,lines_dev_labels = DP.GetSST_2Data('/home/user_home/qiankunjie/project/First-exam/SST-2/dev.tsv')

    # 构建词汇表
    word_sequence_train = " ".join(lines_train).split()
    word_sequence_dev = " ".join(lines_dev).split()
    word_sequence = word_sequence_dev + word_sequence_train
    vocab = ['<PAD>', '<UNK>'] + sorted(set(word_sequence))
    vocab_size = len(vocab)
    # 构建词汇映射表
    word2idx = {w: i for i, w in enumerate(vocab)}

    # 加载预训练的词向量
    embedding_matric = load_pretrained_embeddings(word2idx)



    train_dataset = DP.GetDataset(lines_train, lines_train_labels, word2idx, config["max_len"])
    dev_dataset = DP.GetDataset(lines_dev, lines_dev_labels, word2idx, config["max_len"])


    train_loader = DataLoader(train_dataset, batch_size=config["batchsize"])
    dev_loader = DataLoader(dev_dataset, batch_size=config["batchsize"])




    #初始化模型
    net = BiLSTM(vocab_size, config["embedding_dim"], config["hidden_dim"],config["bidirectional"], config["n_layers"], config["output_dim"], config["dropout"],
                 embedding_matric).to(device)
    # 定义优化器和损失函数
    optimizer = torch.optim.SGD(net.parameters(),lr=config["lr"],momentum=config["momentum"])
    criterion = nn.CrossEntropyLoss().to(device)

    temp_list = []
    #训练
    for epoch in range(config["n_epochs"]):
        train_f1, train_acc, train_loss = train_model(net, train_loader, optimizer, criterion)
        dev_f1, dev_acc, dev_loss = evaluate_model(net, dev_loader, criterion)
        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain f1: {train_f1:.3f} | Train Acc: {train_acc * 100:.2f}% | Train Loss: {train_loss:.3f}')
        print(f'\tdev f1: {dev_f1:.3f} | dev Acc: {dev_acc * 100:.2f}% | dev Loss: {dev_loss:.3f}')
        
        if ((1 + epoch) % config["save_time"]) == 0 or epoch == config["n_epochs"] - 1:
            temp_list.append({"epoch:":epoch + 1, "eval_accuracy:":dev_acc, "eval_f1_score:":dev_f1})
        # 保存最佳模型
        if dev_acc > config["best_dev_acc"]:
            config["best_dev_acc"] = dev_acc
            config["f1_dev"] = dev_f1
            torch.save(net.state_dict(), config["save_model_path"])        
    with open(config["training_result"],"a+") as f:
                json.dump(temp_list,f, indent=4)
    print(f'最佳验证准确率: {config["best_dev_acc"] * 100:.2f}%','\n',f'f1分数：{config["f1_dev"]:.3f}')
if __name__ == "__main__":
    main()