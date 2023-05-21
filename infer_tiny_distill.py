from transformers import BertTokenizerFast,BertModel,BertForSequenceClassification
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import torch
import torch.nn as nn
from tqdm import tqdm
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
        item['labels'] = label
        return item

    def __len__(self):
        return len(self.all_text)


if __name__=="__main__":
    student_path = ".\\tiny_bert_data"
    test_text,test_label = read_data("dataset/tnews_public/test.csv")
    tokenizer = BertTokenizerFast.from_pretrained(student_path)

    batch_size = 35
    max_len = 32
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    student = BertForSequenceClassification.from_pretrained(student_path,num_labels=4)
    student.load_state_dict(torch.load("model_weight\\tiny_bert\\best.pkl"))  # time:2.206s, acc:0.823
    #student.load_state_dict(torch.load("model_weight\\student_without_distill.bin")) #time:0.663s, acc:0.758
    #student.load_state_dict(torch.load("model_weight\\student_distilled.bin")) #time:0.576s, acc:0.780
    student = student.to(device)
    testdataset = BaseDataset(test_text,test_label,max_len)
    dataloader = DataLoader(testdataset,batch_size,shuffle=False)

    start_time = time.time()
    right_num = 0
    for batch_data in tqdm(dataloader, desc="testing"):
        input_ids = batch_data["input_ids"].squeeze(1).to(device)
        attention_mask = batch_data["attention_mask"].squeeze(1).to(device)
        label = batch_data["labels"].to(device)

        output = student.forward(input_ids, attention_mask=attention_mask, labels=label)
        pre = torch.argmax(output.logits,dim=-1)
        right_num += int(torch.sum(pre == label))


    acc = right_num/len(testdataset)
    print(f"time:{time.time()-start_time:.3f}s, acc:{acc:.3f}")