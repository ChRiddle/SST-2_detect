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
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import sys
sys.path.append('..')
from My_SST2_Process import DataProcess as DP

#设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#加载bert模型
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path='/home/user_home/qiankunjie/project/First-exam/BERT/results/checkpoint-1299').to(device)

#加载分词器
tokenizer = AutoTokenizer.from_pretrained('/home/user_home/qiankunjie/HF_Model_Hub/bert-base-uncased')
#分词处理
def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)




def main():
    #加载测试数据集
    lines_test,lines_test_labels = DP.GetSST_2Data('/home/user_home/qiankunjie/project/First-exam/SST-2/test.tsv')
    
    
    dataset_test = Dataset.from_dict({"text":lines_test,"labels":lines_test_labels})


    # 应用分词函数到整个数据集
    tokenized_datasets_test = dataset_test.map(tokenize_function, batched=True)

    
    # 设置数据集格式以返回 PyTorch 张量
    tokenized_datasets_test.set_format('torch', columns=['input_ids', 'labels'])#必须为input_ids和lables



    # 定义训练参数
    training_args = TrainingArguments(
        output_dir='./test',          # 输出目录
        per_device_eval_batch_size=16,   # 每个设备的评估批次大小
    )

    def compute_metrics(eval_pred):

        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1) # 将logits转换为预测标签
        
        # 计算各项指标
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary' 
        )
    
        # 返回指标字典
        return {
            'accuracy': accuracy,
            'f1': f1,
        }


    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        # eval_dataset=tokenized_datasets_test,
        #compute_metrics传入函数方法
        compute_metrics=compute_metrics,
    )


    # 在验证集上评估最终模型性能
    # final_eval_results = trainer.evaluate()
    # print(f"Final evaluation results: {final_eval_results}")
    final_test_results = trainer.predict(tokenized_datasets_test,metric_key_prefix="predict")
    print("BERT")
    print(f"测试集数据量：{len(lines_test)}")
    print(f"Final testion results: {final_test_results.metrics}")


if __name__ == "__main__":
    main()