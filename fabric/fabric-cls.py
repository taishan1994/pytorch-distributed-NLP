import json
import time
import random
import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from collections import Counter
from transformers import BertForMaskedLM, BertTokenizer, BertForSequenceClassification, BertConfig, AdamW
from lightning.fabric import Fabric


def set_seed(seed=123):
    """
    设置随机数种子，保证实验可重现
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_data():
    with open("../data/train.json", "r", encoding="utf-8") as fp:
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
            "label": label_all
        }
        return return_data


def build_optimizer(model, args):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    return optimizer


class Trainer:
    def __init__(self,
                 args,
                 config,
                 model,
                 criterion,
                 optimizer,
                 scheduler=None):
        self.args = args
        self.device = args.device
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def on_step(self, batch_data):
        label = batch_data["label"]
        input_ids = batch_data["input_ids"]
        token_type_ids = batch_data["token_type_ids"]
        attention_mask = batch_data["attention_mask"]
        output = self.model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            labels=label)
        logits = output[1]
        return logits, label

    def on_test_step(self, batch_data):
        label = batch_data["label"].to(self.device)
        input_ids = batch_data["input_ids"].to(self.device)
        token_type_ids = batch_data["token_type_ids"].to(self.device)
        attention_mask = batch_data["attention_mask"].to(self.device)
        output = self.model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            labels=label)
        logits = output[1]
        return logits, label

    def train(self, train_loader):
        gloabl_step = 1
        best_acc = 0.
        start = time.time()
        for epoch in range(1, self.args.epochs + 1):
            for step, batch_data in enumerate(train_loader):
                self.model.train()
                logits, label = self.on_step(batch_data)
                if self.args.use_grad_accumulation:
                    loss = self.criterion(logits, label) / self.args.grad_accumulation
                else:
                    loss = self.criterion(logits, label)
                # loss.backward()
                fabric.backward(loss)
                if self.args.use_grad_accumulation:
                    if (step + 1) % self.args.grad_accumulation == 0:
                        self.optimizer.step()
                        if self.scheduler:
                            self.scheduler.step()
                        self.optimizer.zero_grad()

                        print("【train】 epoch：{}/{} step：{}/{} loss：{:.6f}".format(
                            epoch, self.args.epochs, gloabl_step, self.args.total_step, loss.item()
                        ))
                else:
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    print("【train】 epoch：{}/{} step：{}/{} loss：{:.6f}".format(
                        epoch, self.args.epochs, gloabl_step, self.args.total_step, loss.item()
                    ))
                gloabl_step += 1

        end = time.time()
        print("耗时：{}分钟".format((end - start) / 60))

        torch.save(self.model.state_dict(), self.args.ckpt_path)

    def test(self, model, test_loader, labels):
        self.model = model
        self.model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for step, batch_data in enumerate(test_loader):
                logits, label = self.on_test_step(batch_data)
                label = label.view(-1).detach().cpu().numpy().tolist()
                logits = logits.detach().cpu().numpy()
                pred = np.argmax(logits, axis=1).flatten().tolist()
                trues.extend(label)
                preds.extend(pred)
        # print(trues, preds, labels)
        print(np.array(trues).shape, np.array(preds).shape)
        report = classification_report(trues, preds, target_names=labels)
        return report


class Args:
    model_path = "../model_hub/chinese-bert-wwm-ext"
    ckpt_path = "../output/single-gpu-cls.pt"
    max_seq_len = 128
    ratio = 0.92
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    train_batch_size = 32
    dev_batch_size = 32
    weight_decay = 0.01
    epochs = 1
    learning_rate = 3e-5
    eval_step = 100

    use_fp16 = True
    use_bf16 = False
    use_grad_accumulation = True
    grad_accumulation = 4
    optim = "sgd"
    use_fabrac_init = True


if __name__ == "__main__":
    # =======================================
    # 定义相关参数
    set_seed()
    label2id = {
        "其他": 0,
        "喜好": 1,
        "悲伤": 2,
        "厌恶": 3,
        "愤怒": 4,
        "高兴": 5,
    }
    args = Args()
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    # =======================================

    # =======================================
    # 加载数据集
    data = load_data()
    # 取1万条数据出来
    data = data[:10000]
    random.shuffle(data)
    train_num = int(len(data) * args.ratio)
    train_data = data[:train_num]
    dev_data = data[train_num:]

    collate = Collate(tokenizer, args.max_seq_len)
    train_loader = DataLoader(train_data,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collate.collate_fn)
    total_step = len(train_loader) * args.epochs
    args.total_step = total_step
    test_loader = DataLoader(dev_data,
                             batch_size=args.dev_batch_size,
                             shuffle=False,
                             num_workers=2,
                             collate_fn=collate.collate_fn)
    # =======================================

    # =======================================
    # 定义模型、优化器、损失函数
    config = BertConfig.from_pretrained(args.model_path, num_labels=6)
    if args.use_fp16:
        fabric = Fabric(accelerator="cuda", devices=1, precision="16-mixed")
    elif args.use_bf16:
        import torch

        print(torch.cuda.is_bf16_supported())
        fabric = Fabric(accelerator="cuda", devices=1, precision="bf16-mixed")

    if args.use_fabrac_init:
        with fabric.init_module():
            model = BertForSequenceClassification.from_pretrained(args.model_path, config=config)
    else:
        model = BertForSequenceClassification.from_pretrained(args.model_path, config=config)
    criterion = torch.nn.CrossEntropyLoss()
    if args.optim == "adamw":
        optimizer = build_optimizer(model, args)
    elif args.optim == "sgd":
        num_steps = args.epochs * len(train_loader)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps)

    else:
        fabric = Fabric(accelerator="cuda", devices=1)
    fabric.launch()

    model, optimizer = fabric.setup(model, optimizer)
    train_loader = fabric.setup_dataloaders(train_loader)
    # =======================================
    # 定义训练器
    trainer = Trainer(args,
                      config,
                      model,
                      criterion,
                      optimizer,
                      scheduler)

    # 训练和验证
    trainer.train(train_loader)

    # 测试
    labels = list(label2id.keys())
    config = BertConfig.from_pretrained(args.model_path, num_labels=6)
    model = BertForSequenceClassification.from_pretrained(args.model_path, config=config)
    model.load_state_dict(torch.load(args.ckpt_path))
    model.to(args.device)
    report = trainer.test(model, test_loader, labels)
    print(report)
    # =======================================


