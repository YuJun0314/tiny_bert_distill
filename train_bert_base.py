from transformers import BertTokenizerFast,BertModel,BertForSequenceClassification
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import torch
import os
from  tqdm import tqdm
from textbrewer import DistillationConfig,TrainingConfig,GeneralDistiller,MultiTeacherDistiller,MultiTaskDistiller
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

if __name__=="__main__":
    path = os.path.join("dataset","ants","train.csv")
    train_text_a, train_text_b, train_label = read_data(path)
    dev_text_a, dev_text_b, dev_label = read_data(os.path.join("dataset","ants","dev.csv"))
    tokenizer = BertTokenizerFast.from_pretrained(".//roberta_data")
    max_len = 32
    batch_size = 35
    epoch = 4
    lr = 1e-6
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dataset = myDataset(train_text_a,train_text_b,train_label,max_len)
    train_dataloader = DataLoader(train_dataset,batch_size,shuffle=True)

    dev_dataset = myDataset(dev_text_a, dev_text_b, dev_label,max_len)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=False)

    model = BertForSequenceClassification.from_pretrained("./bert_base_chinese", num_labels=4)
    model.load_state_dict(torch.load("model_weight/bert.bin",map_location=device))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr)

    best_acc = 0.7059

    for e in range(epoch):
        for batch_data in tqdm(train_dataloader,desc="train..."):
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            label = batch_data["labels"].to(device)

            output = model(input_ids,attention_mask=attention_mask,labels=label)
            loss = output.loss
            loss.backward()
            opt.step()
            opt.zero_grad()

        right_num = 0
        for batch_data in tqdm(dev_dataloader):
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            label = batch_data["labels"].to(device)

            output = model.forward(input_ids, attention_mask=attention_mask, labels=label)

            pre = torch.argmax(output.logits, dim=-1)
            right_num += int(torch.sum(pre == label))

        acc = right_num / len(dev_dataset)

        if acc > best_acc:
            torch.save(model.state_dict(), "model_weight/bert.bin")
        print(f"acc:{acc:.4f}")

