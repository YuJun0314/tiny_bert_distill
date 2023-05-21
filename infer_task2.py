from transformers import BertTokenizerFast,BertModel,BertForSequenceClassification
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import torch
import os
from  tqdm import tqdm
import time
def read_data(file):
    all_data = pd.read_csv(file)
    text_a = all_data['text_a'].tolist()
    text_b = all_data['text_b'].tolist()
    label = all_data['label'].tolist()

    return text_a,text_b,label

class myDataset(Dataset):
    def __init__(self,all_text_a,all_text_b,all_label,max_len):
        self.all_text_a = all_text_a
        self.all_text_b = all_text_b
        self.all_label = all_label
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.all_text_a[index][-16:] + tokenizer.cls_token + self.all_text_b[index][:17]
        label = self.all_label[index]

        item = tokenizer(text=text, return_tensors='pt', max_length=self.max_len,truncation=True,padding="max_length")
        item = item.data
        item['input_ids'] = item['input_ids'].squeeze(0)
        item['token_type_ids'] = item['token_type_ids'].squeeze(0)
        item['attention_mask'] = item['attention_mask'].squeeze(0)
        item['labels'] = label
        return item

    def __len__(self):
        return len(self.all_label)

if __name__ == "__main__":
    path = ".\\tiny_bert_data"
    max_len = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BertForSequenceClassification.from_pretrained(path,num_labels=4)
    model.load_state_dict(torch.load("tiny_bert_output\\gs1000.pkl",map_location=device))
    model = model.to(device)

    test_text_a, test_text_b, test_label = read_data(".\\dataset\\ants\\dev.csv")
    test_dataset = myDataset(test_text_a, test_text_b, test_label,max_len)
    test_dataloader = DataLoader(test_dataset, 40, shuffle=False)

    tokenizer = BertTokenizerFast.from_pretrained(path)

    start_time = time.time()
    right_num = 0
    for batch_data in tqdm(test_dataloader):
        input_ids = batch_data["input_ids"].to(device)
        attention_mask = batch_data["attention_mask"].to(device)
        label = batch_data["labels"].to(device)

        output = model.forward(input_ids, attention_mask=attention_mask, labels=label)

        pre = torch.argmax(output.logits, dim=-1)
        right_num += int(torch.sum(pre == label))
    acc = right_num / len(test_dataset)

    end_time = time.time()

    cost_time = end_time-start_time

    print(f"模型参数量：{sum(p.numel() for p in model.parameters())/1000/1000/1000}B")  # chat-glm 6B
    print(f"准确率：{acc}")
    print(f"推理测试集时间：{cost_time}s")