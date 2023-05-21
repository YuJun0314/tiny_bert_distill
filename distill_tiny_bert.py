from transformers import BertTokenizerFast,BertModel,BertForSequenceClassification
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import torch
from  tqdm import tqdm
from textbrewer import DistillationConfig,TrainingConfig,GeneralDistiller

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
        item = tokenizer(text=text,return_tensors="pt",truncation=True,padding="max_length",max_length=self.max_len)
        item = item.data
        item["input_ids"] = item["input_ids"].squeeze(0)
        item["token_type_ids"] = item["token_type_ids"].squeeze(0)
        item["attention_mask"] = item["attention_mask"].squeeze(0)
        item["labels"] = label
        return item

    def __len__(self):
        return len(self.all_label)

def simple_adaptor_T(batch, model_outputs):
    return{
        "losses":model_outputs[0],
        "logits":model_outputs[1],
        "hidden":model_outputs[2],
        "attention":model_outputs[3]
    }

def simple_adaptor_S(batch, model_outputs):
    return{
        "losses":model_outputs.loss,
        "logits":model_outputs.logits,
        "hidden":model_outputs.hidden_states,
        "attention":model_outputs.attentions
    }

if __name__ == "__main__":
    teacher_path = ".\\roberta_data"
    student_path = ".\\tiny_bert_data"

    train_text, train_label = read_data("dataset/tnews_public/train.csv")
    dev_text, dev_label = read_data("dataset/tnews_public/dev.csv")
    tokenizer = BertTokenizerFast.from_pretrained(teacher_path)

    epoch = 6
    batch_size = 35
    max_len = 32
    lr = 1e-5
    device = "cuda:0" if torch.cuda.is_available() else "cup"
    train_dataset = BaseDataset(train_text, train_label, max_len)
    train_dataloader = DataLoader(train_dataset, batch_size,shuffle=False)
    dev_dataset = BaseDataset(dev_text,dev_label,max_len)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=False)
    teacher = BertForSequenceClassification.from_pretrained(teacher_path, num_labels=4,return_dict=False,
                                                            output_hidden_states=True,output_attentions=True)
    teacher.load_state_dict(torch.load("model_weight/roberta.bin",map_location=device))
    teacher = teacher.to(device)
    student = BertForSequenceClassification.from_pretrained(student_path, num_labels=4, return_dict=True,
                                                            output_hidden_states=True, output_attentions=True)

    student.load_state_dict(torch.load("model_weight\\tiny_bert\\best.pkl",map_location=device))
    student = student.to(device)

    opt = torch.optim.Adam(student.parameters(), lr=lr)

    distill_config = DistillationConfig(
        temperature=4,
        hard_label_weight=0.2,
        kd_loss_type='ce',
        kd_loss_weight=0.8,
        intermediate_matches=[
            {'layer_T': 0, 'layer_S': 0, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1,
             'proj': ['linear', 312, 768]},     # embedding 映射
            {'layer_T': 3, 'layer_S': 1, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 0.6,
             'proj': ['linear', 312, 768]},
            {'layer_T': 6, 'layer_S': 2, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 0.7,
             'proj': ['linear', 312, 768]},
            {'layer_T': 9, 'layer_S': 3, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 0.8,
             'proj': ['linear', 312, 768]},
            {'layer_T': 12, 'layer_S': 4, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1,
             'proj': ['linear', 312, 768]},

            {'layer_T': 2, 'layer_S': 0, 'feature': 'attention', 'loss': 'attention_mse', 'weight': 0.7},
            {'layer_T': 5, 'layer_S': 1, 'feature': 'attention', 'loss': 'attention_mse', 'weight': 0.8},
            {'layer_T': 8, 'layer_S': 2, 'feature': 'attention', 'loss': 'attention_mse', 'weight': 0.9},
            {'layer_T': 11, 'layer_S': 3, 'feature': 'attention', 'loss': 'attention_mse', 'weight': 1}
        ]
    )
    train_config = TrainingConfig(
        output_dir="model_weight/tiny_bert",
        ckpt_epoch_frequency=2,
        device=device,
    )

    distiller = GeneralDistiller(
        model_T = teacher,
        model_S = student,
        distill_config=distill_config,
        train_config=train_config,
        adaptor_T=simple_adaptor_T,
        adaptor_S=simple_adaptor_S
    )
    print("开始蒸馏....")
    distiller.train(opt,train_dataloader,epoch)