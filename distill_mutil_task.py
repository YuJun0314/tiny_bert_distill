from transformers import BertTokenizerFast,BertModel,BertForSequenceClassification
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import torch
from  tqdm import tqdm
from textbrewer import DistillationConfig,TrainingConfig,GeneralDistiller,MultiTeacherDistiller,MultiTaskDistiller
import os

def simple_adaptor(batch,model_outputs):
    return {
        "losses":model_outputs[0],
        "logits":model_outputs[1],
        "hidden":model_outputs[2],
        "attention":model_outputs[3]
    }


def read_data1(file):
    all_data = pd.read_csv(file)
    all_text = all_data["text"].tolist()
    all_label = all_data["label"].tolist()
    return all_text,all_label

def read_data2(file):
    all_data = pd.read_csv(file)
    all_text_a = all_data["text_a"].tolist()
    all_text_b = all_data["text_b"].tolist()
    all_label = all_data["label"].tolist()
    return all_text_a,all_text_b,all_label

class BaseDataset(Dataset):
    def __init__(self,all_text,all_label):
        self.all_text = all_text
        self.all_label = all_label

    def __getitem__(self, index):
        text = self.all_text[index]
        label = self.all_label[index]

        item = tokenizer(text=text,return_tensors="pt",truncation=True,padding="max_length",max_length=max_len)
        item = item.data

        item["input_ids"] = item["input_ids"].squeeze(0)
        item["token_type_ids"] = item["token_type_ids"].squeeze(0)
        item["attention_mask"] = item["attention_mask"].squeeze(0)

        item["labels"] = label

        return item

    def __len__(self):
        return len(self.all_label)

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
    max_len = 32
    batch_size = 35
    epoch = 4
    lr = 1e-4
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizerFast.from_pretrained(".//roberta_data")

    train_text_a, train_text_b, train_label = read_data2(os.path.join("dataset","ants","train.csv"))
    train_dataset2 = myDataset(train_text_a, train_text_b, train_label, max_len)
    train_dataloader2 = DataLoader(train_dataset2, batch_size, shuffle=True)

    train_text1, train_label1 = read_data1("./dataset/tnews_public/train.csv")
    train_dataset1 = BaseDataset(train_text1, train_label1)
    train_dataloader1 = DataLoader(train_dataset1, batch_size, shuffle=False)
    #--------------------加载robertta模型------------------------------------------
    teacher_model_roberta = BertForSequenceClassification.from_pretrained(".\\roberta_data", num_labels=4,
                                                                          return_dict=False, output_hidden_states=True,
                                                                          output_attentions=True)
    teacher_model_roberta.load_state_dict(torch.load("./model_weight/roberta.bin", map_location=device))
    teacher_model_roberta = teacher_model_roberta.to(device)
    #------------------加载bert-base模型-------------------------------------------
    teacher_model_bert_base = BertForSequenceClassification.from_pretrained(".\\bert_base_chinese", num_labels=4,
                                                                            return_dict=False,
                                                                            output_hidden_states=True,
                                                                            output_attentions=True)
    teacher_model_bert_base.load_state_dict(torch.load("./model_weight/bert.bin", map_location=device))
    teacher_model_bert_base = teacher_model_bert_base.to(device)
    #----------------学生模型------------------------------------------------------
    student = BertForSequenceClassification.from_pretrained(".\\tiny_bert_data", num_labels=4, return_dict=False,output_hidden_states=True, output_attentions=True)
    student_model = student.to(device)
    opt = torch.optim.Adam(student_model.parameters(), lr=lr)

    distill_config = DistillationConfig(
        temperature=4,
        hard_label_weight=1,
        kd_loss_type='ce',
        kd_loss_weight=1.3,
        intermediate_matches=[
            {"layer_T": 0, "layer_S": 0, "feature": "hidden", "loss": "hidden_mse", "weight": 1,
             "proj": ["linear", 312, 768]},  # embedding 映射
            {"layer_T": 3, "layer_S": 1, "feature": "hidden", "loss": "hidden_mse", "weight": 0.6,
             "proj": ["linear", 312, 768]},  # encoder 映射
            {"layer_T": 6, "layer_S": 2, "feature": "hidden", "loss": "hidden_mse", "weight": 0.7,
             "proj": ["linear", 312, 768]},  # encoder 映射
            {"layer_T": 9, "layer_S": 3, "feature": "hidden", "loss": "hidden_mse", "weight": 0.8,
             "proj": ["linear", 312, 768]},  # encoder 映射
            {"layer_T": 12, "layer_S": 4, "feature": "hidden", "loss": "hidden_mse", "weight": 1,
             "proj": ["linear", 312, 768]},  # encoder 映射
            {"layer_T": 2, "layer_S": 0, "feature": "attention", "loss": "attention_mse", "weight": 0.7},
            {"layer_T": 5, "layer_S": 1, "feature": "attention", "loss": "attention_mse", "weight": 0.8},
            {"layer_T": 8, "layer_S": 2, "feature": "attention", "loss": "attention_mse", "weight": 0.9},
            {"layer_T": 11, "layer_S": 3, "feature": "attention", "loss": "attention_mse", "weight": 1}
        ]
    )
    train_config = TrainingConfig(
        output_dir="./tiny_bert_output",
        ckpt_steps=200
    )

    distiller = MultiTaskDistiller(
        train_config=train_config,
        distill_config=distill_config,
        model_S=student,
        model_T={"cls":teacher_model_roberta,"match":teacher_model_bert_base},
        adaptor_T={"cls":simple_adaptor,"match":simple_adaptor},
        adaptor_S={"cls":simple_adaptor,"match":simple_adaptor}
    )
    with distiller:
        print("开始蒸馏")
        distiller.train(optimizer=opt,
                        dataloaders={"cls":train_dataloader1,"match":train_dataloader2},
                        num_steps=1000)