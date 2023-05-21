from transformers import BertTokenizerFast,BertModel,BertForSequenceClassification
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import torch
from tqdm import tqdm

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

if __name__=="__main__":
    train_text,train_label = read_data("dataset/tnews_public/train.csv")
    dev_text, dev_label = read_data("dataset/tnews_public/dev.csv")
    tokenizer = BertTokenizerFast.from_pretrained("./roberta_data")

    epoch = 6
    batch_size = 35
    max_len = 32
    lr = 0.8e-5
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dataset = BaseDataset(train_text,train_label,max_len)
    train_dataloader = DataLoader(train_dataset,batch_size,shuffle=False)

    dev_dataset = BaseDataset(dev_text, dev_label, max_len)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=False)

    model = BertForSequenceClassification.from_pretrained("./roberta_data",num_labels=4)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(),lr)

    best_acc = -1
    for e in range(epoch):
        for batch_data in tqdm(train_dataloader,desc="training"):
            input_ids = batch_data["input_ids"].squeeze(1).to(device)
            attention_mask = batch_data["attention_mask"].squeeze(1).to(device)
            label = batch_data["label"].to(device)

            output = model.forward(input_ids, attention_mask=attention_mask,labels=label)
            loss = output.loss
            loss.backward()
            opt.step()
            opt.zero_grad()

        right_num = 0
        for batch_data in dev_dataloader:
            input_ids = batch_data["input_ids"].squeeze(1).to(device)
            attention_mask = batch_data["attention_mask"].squeeze(1).to(device)
            label = batch_data["label"].to(device)
            output = model.forward(input_ids, attention_mask=attention_mask, labels=label)
            pre = torch.argmax(output.logits, dim=-1)
            right_num += int(torch.sum(pre == label))

        acc = right_num/len(dev_dataset)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(),"model_weight/roberta.bin")
        print(f"acc:{acc:.3f}, best_acc:{best_acc:.3f}")
