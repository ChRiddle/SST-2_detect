# from transformers import pipeline
# classifier = pipeline('text-classification', model='bert-base-uncased')
# pre = classifier('We are very happy to introduce pipeline to the transformers repository.')
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import AutoTokenizer
from transformers import Trainer
import numpy as np
import torch
import pandas as pd
# from torch.utils.data import Dataset
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import sys
sys.path.append('..')
from My_SST2_Process import DataProcess as DP

#设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sentence = ["hello, world!"]
# print(model)


# #构造输入数据
# class TextDataset(Dataset):
#     def __init__(self, texts, labels, word2idx, max_len):
#         self.texts = texts
#         self.labels = labels
#         self.word2idx = word2idx
#         self.max_len = max_len
#     def __len__(self):
#         return len(self.texts)
#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         label = self.labels[idx]
#         # 将文本转换为索引序列
#         tokens = (text.lower()).split()
#         indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in tokens]
#         # 填充或截断序列
#         if len(indices) > self.max_len:
#             indices = indices[:self.max_len]
#         else:
#             indices = indices + [self.word2idx['<PAD>']] * (self.max_len - len(indices))
#         return torch.tensor(indices, dtype=torch.long).to(device), torch.tensor(label, dtype=torch.long)



#加载bert模型
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path='/home/user_home/qiankunjie/HF_Model_Hub/bert-base-uncased',
                                                            num_labels=2,output_attentions=False,output_hidden_states=False).to(device)

#加载分词器
tokenizer = AutoTokenizer.from_pretrained('/home/user_home/qiankunjie/HF_Model_Hub/bert-base-uncased')
#分词处理
def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)




def main():
    # 读取数据
    lines_train,lines_train_labels = DP.GetSST_2Data('/home/user_home/qiankunjie/project/First-exam/SST-2/train.tsv')
    lines_dev,lines_dev_labels = DP.GetSST_2Data('/home/user_home/qiankunjie/project/First-exam/SST-2/dev.tsv')

    
    
    dataset_train = Dataset.from_dict({"text":lines_train,"labels":lines_train_labels})
    dataset_dev = Dataset.from_dict({"text":lines_dev,"labels":lines_dev_labels})
    

    # 应用分词函数到整个数据集
    tokenized_datasets_train = dataset_train.map(tokenize_function, batched=True)
    tokenized_datasets_dev = dataset_dev.map(tokenize_function, batched=True)
    
    # 设置数据集格式以返回 PyTorch 张量
    tokenized_datasets_train.set_format('torch', columns=['input_ids', 'labels'])#必须为input_ids和lables
    tokenized_datasets_dev.set_format('torch', columns=['input_ids', 'labels'])


    #定义训练参数
    training_args = TrainingArguments(
        output_dir='./results2',          # 输出目录
        num_train_epochs=3,              # 训练轮数
        per_device_train_batch_size=8,   # 每个设备的训练批次大小
        per_device_eval_batch_size=16,   # 每个设备的评估批次大小
        #啥叫预热步数
        warmup_steps=500,                # 预热步数
        weight_decay=0.01,               # 权重衰减
        logging_dir='./logs',            # 日志目录
        logging_steps=10,                # 日志记录频率
        eval_strategy='epoch',     # 评估策略：每个 epoch 后评估
        save_strategy='epoch',           # 保存策略：每个 epoch 后保存
        load_best_model_at_end=True,     # 训练结束后加载最佳模型
        metric_for_best_model='accuracy', # 用于选择最佳模型的指标
        learning_rate=2e-5,              # 学习率，通常设置较小:cite[1]
    )
    #加载评估指标
    # metric = evaluate.load('F1_score')

    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred
    #     predictions = np.argmax(logits, axis=-1)
    #     return metric.compute(predictions=predictions, references=labels)

    def compute_metrics(eval_pred):

        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1) # 将logits转换为预测标签
        
        # 计算各项指标
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro' 
        )
    
        # 返回指标字典
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }


    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets_train,
        eval_dataset=tokenized_datasets_dev,
        #compute_metrics传入函数方法
        compute_metrics=compute_metrics,
    )

    # 开始训练！
    trainer.train()

    # 在验证集上评估最终模型性能
    final_eval_results = trainer.evaluate()
    print(f"Final evaluation results: {final_eval_results}")


if __name__ == "__main__":
    main()