from transformers import BertTokenizer,BertModel,BertForSequenceClassification
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import torch
from  tqdm import tqdm
import time

def read_data(file):
    all_data = pd.read_csv(file)
    all_text = all_data["text"].tolist()
    all_label = all_data["label"].tolist()

    return all_text,all_label

class BaseDataset(Dataset):
    def __init__(self,all_text,all_label):
        self.all_text = all_text
        self.all_label = all_label

    def __getitem__(self, index):
        text = self.all_text[index]
        label = self.all_label[index]

        item = tokenizer(text=text,return_tensors="pt",truncation=True,padding="max_length",max_length=max_len)
        item = item.data

        item["label"] = label

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

    test_text, test_label = read_data(".\\dataset\\tnews_public\\test.csv")
    test_dataset = BaseDataset(test_text, test_label)
    test_dataloader = DataLoader(test_dataset, 40, shuffle=False)

    tokenizer = BertTokenizer.from_pretrained(path)

    start_time = time.time()
    right_num = 0
    for batch_data in tqdm(test_dataloader):
        input_ids = batch_data["input_ids"].squeeze(1).to(device)
        attention_mask = batch_data["attention_mask"].squeeze(1).to(device)
        label = batch_data["label"].to(device)

        output = model.forward(input_ids, attention_mask=attention_mask, labels=label)

        pre = torch.argmax(output.logits, dim=-1)
        right_num += int(torch.sum(pre == label))
    acc = right_num / len(test_dataset)

    end_time = time.time()

    cost_time = end_time-start_time

    print(f"模型参数量：{sum(p.numel() for p in model.parameters())/1000/1000/1000}B")  # chat-glm 6B
    print(f"准确率：{acc}")
    print(f"推理测试集时间：{cost_time}s")
    """模型参数量：0.011420572B
    准确率：0.7835150737652341
    推理测试集时间：2.5356063842773438s"""