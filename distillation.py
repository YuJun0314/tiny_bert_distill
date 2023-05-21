from transformers import BertTokenizerFast,BertModel,BertForSequenceClassification
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import softmax
import time

def read_data(file):
    all_data = pd.read_csv(file)
    all_text = all_data["text"].tolist()
    all_label = all_data["label"].tolist()
    return all_text,all_label

class BaseDataset(Dataset):
    def __init__(self,all_text,all_label,max_len):
        self.all_text = all_text
        self.all_label = all_label
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.all_text[index]
        label = self.all_label[index]

        item = tokenizer(text=text, return_tensors='pt', max_length=self.max_len,truncation=True,padding="max_length")
        item = item.data
        item['label'] = label
        return item

    def __len__(self):
        return len(self.all_text)

class TextCLS(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(21128, 100)
        self.lstm = nn.LSTM(100,hidden_size=256,batch_first=True,num_layers=2)
        self.dense1 = nn.Linear(256, 100)
        self.dense2 = nn.Linear(100, 4)
    def forward(self,x):
        x = self.embedding(x)
        _,h = self.lstm(x)
        x = h[0][1]
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def CE(pred,label,t=1): # t是蒸馏温度,缩小权重值之间的差距，让他们接近一点
    pred = softmax(pred/t,dim=-1)
    label = softmax(label / t, dim=1)

    loss = - torch.sum(label * torch.log(pred))

    return loss

if __name__=="__main__":
    train_text,train_label = read_data("dataset/tnews_public/train.csv")
    dev_text, dev_label = read_data("dataset/tnews_public/dev.csv")
    tokenizer = BertTokenizerFast.from_pretrained("./roberta_data")

    epoch = 20
    batch_size = 35
    max_len = 32
    lr = 0.5e-3
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dataset = BaseDataset(train_text,train_label,max_len)
    train_dataloader = DataLoader(train_dataset,batch_size,shuffle=False)

    dev_dataset = BaseDataset(dev_text, dev_label, max_len)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=False)
#------------------------学生模型------------------------------------------
    student = TextCLS()
    student.load_state_dict(torch.load("model_weight\\student_without_distill.bin",map_location=device))
    student = student.to(device)
#------------------------教师模型-------------------------------------------
    teacher = BertForSequenceClassification.from_pretrained("./roberta_data",num_labels=4)
    teacher.load_state_dict(torch.load("model_weight\\roberta.bin",map_location=device))
    teacher = teacher.to(device)
#-------------------------------------------------------------------------
    opt = torch.optim.Adam(student.parameters(),lr)
    loss_func = nn.CrossEntropyLoss()

    best_acc = -1
    # best_acc:0.789
    for e in range(epoch):
        for batch_data in tqdm(train_dataloader, desc="training"):
            input_ids = batch_data["input_ids"].squeeze(1).to(device)
            attention_mask = batch_data["attention_mask"].squeeze(1).to(device)
            label = batch_data["label"].to(device)

            student_pre = student.forward(input_ids)
            with torch.no_grad():
                teacher_out = teacher.forward(input_ids,attention_mask=attention_mask,labels=label)
            teacher_pre = teacher_out.logits
            loss1 = loss_func(student_pre, label)
            loss2 = CE(student_pre,teacher_pre)
            loss = loss1*0.3+loss2*0.7
            loss.backward()
            opt.step()
            opt.zero_grad()

        right_num = 0
        for batch_data in dev_dataloader:
            input_ids = batch_data["input_ids"].squeeze(1).to(device)
            attention_mask = batch_data["attention_mask"].squeeze(1).to(device)
            label = batch_data["label"].to(device)
            output = student.forward(input_ids)
            pre = torch.argmax(output, dim=-1)
            right_num += int(torch.sum(pre == label))

        acc = right_num / len(dev_dataset)

        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), "model_weight/student_distilled.bin")
        print(f"acc:{acc:.3f}, best_acc:{best_acc:.3f}")