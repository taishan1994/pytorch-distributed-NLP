import torch
import json
import random
import numpy as np

from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig


def get_data():
    with open("data/train.json", "r", encoding="utf-8") as fp:
        data = fp.read()
    data = json.loads(data)
    return data


def load_data():
    data = get_data()
    return_data = []
    # [(文本， 标签id)]
    for d in data:
        text = d[0]
        label = d[1]
        return_data.append(("".join(text.split(" ")).strip(), label))
    return return_data


class Collate:
    def __init__(self,
                 tokenizer,
                 max_seq_len,
                 ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def collate_fn(self, batch):
        input_ids_all = []
        token_type_ids_all = []
        attention_mask_all = []
        label_all = []
        for data in batch:
            text = data[0]
            label = data[1]
            inputs = self.tokenizer.encode_plus(text=text,
                                                max_length=self.max_seq_len,
                                                padding="max_length",
                                                truncation="longest_first",
                                                return_attention_mask=True,
                                                return_token_type_ids=True)
            input_ids = inputs["input_ids"]
            token_type_ids = inputs["token_type_ids"]
            attention_mask = inputs["attention_mask"]
            input_ids_all.append(input_ids)
            token_type_ids_all.append(token_type_ids)
            attention_mask_all.append(attention_mask)
            label_all.append(label)

        input_ids_all = torch.tensor(input_ids_all, dtype=torch.long)
        token_type_ids_all = torch.tensor(token_type_ids_all, dtype=torch.long)
        attention_mask_all = torch.tensor(attention_mask_all, dtype=torch.long)
        label_all = torch.tensor(label_all, dtype=torch.long)
        return_data = {
            "input_ids": input_ids_all,
            "attention_mask": attention_mask_all,
            "token_type_ids": token_type_ids_all,
            "labels": label_all
        }
        return return_data

class Args:
    model_path = "model_hub/chinese-bert-wwm-ext"
    single_gpu_ckpt_path = "output/single-gpu-cls.pt"
    max_seq_len = 128
    ratio = 0.92
    epochs = 1
    eval_step = 50
    dev = False
    local_rank = None
    train_batch_size = 32
    dev_batch_size = 32
    weight_decay = 0.01
    learning_rate=3e-5

models = {
    "multi_gpu_dataparallel_ckpt_path" : "output/multi-gpu-dataparallel-cls.pt",
    "multi_gpu_distributed_ckpt_path" : "output/multi-gpu-distributed-cls.pt",
    "multi_gpu_distributed_mp_ckpt_path" : "output/multi-gpu-distributed-mp-cls.pt",
    "multi_gpu_distributed_mp_amp_ckpt_path" : "output/multi-gpu-distributed-mp-amp-cls.pt",
    "multi_gpu_horovod_ckpt_path" : "output/multi-gpu-horovod-cls.pt",
    "multi_gpu_deepspeed_ckpt_path" : "output/deepspeed/pytorch_model.bin",
    "multi_gpu_accelerate_ckpt_path" : "output/accelerate/multi-gpu-accelerate-cls.pt",
    "multi_gpu_transformers_ckpt_path" : "output/transformers/checkpoint-100/pytorch_model.bin",
}

def mapping(checkpoint):
    old_state = torch.load(checkpoint)
    new_state = {}
    for k,v in old_state.items():
        new_state[k.replace("module.", "")] = v
    return new_state


def tokenize_input(args, tokenizer, text):
    inputs = tokenizer.encode_plus(text,
                                   max_length=args.max_seq_len,
                                   truncation=True,
                                   padding="max_length",
                                   return_tensors="pt",
                                   return_attention_mask=True,
                                   return_token_type_ids=True)
    input_ids = inputs["input_ids"].cuda()
    token_type_ids = inputs["token_type_ids"].cuda()
    attention_mask = inputs["attention_mask"].cuda()
    output = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
    return output


def predict(args, model, id2label, tokenizer, text):
    inputs = tokenize_input(args, tokenizer, text)
    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        output = model(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               )
    logits = output.logits
    logits = logits.detach().cpu().numpy()
    pred = np.argmax(logits, axis=1).flatten().tolist()
    print("预测：", id2label[pred[0]])


if __name__ == "__main__":
    # =======================================
    # 定义相关参数
    args = Args()
    label2id = {
        "其他": 0,
        "喜好": 1,
        "悲伤": 2,
        "厌恶": 3,
        "愤怒": 4,
        "高兴": 5,
    }
    id2label = {v:k for k,v in label2id.items()}

    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    data = load_data()
    do_label = None
    while do_label != 3:
        d = random.choice(data)
        do_label = d[1]
    text, label = d
    # =======================================
    for model_name, model_checkpoint in models.items():
        print("="*100)


        config = BertConfig.from_pretrained(args.model_path, num_labels=6)
        model = BertForSequenceClassification.from_pretrained(args.model_path, config=config)
        model.load_state_dict(mapping(model_checkpoint))
        model.cuda()

        print(model_name)
        print("文本：", text)
        print("真实：", id2label[label])
        predict(args, model, id2label, tokenizer, text)
        print("=" * 100)