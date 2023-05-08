import json
import time
import random
import torch
import deepspeed
import torch.nn as nn
import numpy as np
import torch.distributed as dist

from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from collections import Counter
from transformers import BertForMaskedLM, BertTokenizer, BertForSequenceClassification, BertConfig, AdamW


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
            "label": label_all
        }
        return return_data


class Trainer:
    def __init__(self,
                 args,
                 config,
                 model_engine,
                 criterion,
                 optimizer):
        self.args = args
        self.config = config
        self.model_engine = model_engine
        self.criterion =criterion
        self.optimizer = optimizer


    def on_step(self, batch_data):
        label = batch_data["label"].cuda()
        input_ids = batch_data["input_ids"].cuda()
        token_type_ids = batch_data["token_type_ids"].cuda()
        attention_mask = batch_data["attention_mask"].cuda()
        output = self.model_engine.forward(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            labels=label)
        logits = output[1]
        return logits, label

    def loss_reduce(self, loss):
        rt = loss.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= torch.cuda.device_count()
        return rt

    def output_reduce(self, outputs, targets):
        output_gather_list = [torch.zeros_like(outputs) for _ in range(torch.cuda.device_count())]
        # 把每一个GPU的输出聚合起来
        dist.all_gather(output_gather_list, outputs)

        outputs = torch.cat(output_gather_list, dim=0)
        target_gather_list = [torch.zeros_like(targets) for _ in range(torch.cuda.device_count())]
        # 把每一个GPU的输出聚合起来
        dist.all_gather(target_gather_list, targets)
        targets = torch.cat(target_gather_list, dim=0)
        return outputs, targets

    def train(self, train_loader, dev_loader=None):
        gloabl_step = 1
        best_acc = 0.
        if self.args.local_rank == 0:
            start = time.time()
        for epoch in range(1, self.args.epochs + 1):
            for step, batch_data in enumerate(train_loader):
                self.model_engine.train()
                logits, label = self.on_step(batch_data)
                loss = self.criterion(logits, label)
                self.model_engine.backward(loss)
                self.model_engine.step()
                loss = self.loss_reduce(loss)
                if self.args.local_rank == 0:
                    print("【train】 epoch：{}/{} step：{}/{} loss：{:.6f}".format(
                        epoch, self.args.epochs, gloabl_step, self.args.total_step, loss
                    ))
                gloabl_step += 1
                if self.args.dev:
                    if gloabl_step % self.args.eval_step == 0:
                        loss, accuracy = self.dev(dev_loader)
                        if self.args.local_rank == 0:
                            print("【dev】 loss：{:.6f} accuracy：{:.4f}".format(loss, accuracy))
                            if accuracy > best_acc:
                                best_acc = accuracy
                                print("【best accuracy】 {:.4f}".format(best_acc))
                                torch.save(self.model_engine.state_dict(), self.args.ckpt_path)
                                # self.model_engine.save_checkpoint("./")
        if self.args.local_rank == 0:
            end = time.time()
            print("耗时：{}分钟".format((end - start) / 60))
        if not self.args.dev and self.args.local_rank == 0:
            torch.save(self.model_engine.state_dict(), self.args.ckpt_path)
            # self.model_engine.save_checkpoint("./")

    def dev(self, dev_loader):
        self.model_engine.eval()
        correct_total = 0
        num_total = 0
        loss_total = 0.
        with torch.no_grad():
            for step, batch_data in enumerate(dev_loader):
                logits, label = self.on_step(batch_data)
                loss = self.criterion(logits, label)
                loss = self.loss_reduce(loss)
                logits, label = self.output_reduce(logits, label)
                loss_total += loss
                logits = logits.detach().cpu().numpy()
                label = label.view(-1).detach().cpu().numpy()
                num_total += len(label)
                preds = np.argmax(logits, axis=1).flatten()
                correct_num = (preds == label).sum()
                correct_total += correct_num

        return loss_total, correct_total / num_total

    def test(self, model, test_loader, labels):
        self.model_engine = model
        self.model_engine.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for step, batch_data in enumerate(test_loader):
                logits, label = self.on_step(batch_data)
                logits, label = self.output_reduce(logits, label)
                label = label.view(-1).detach().cpu().numpy().tolist()
                logits = logits.detach().cpu().numpy()
                pred = np.argmax(logits, axis=1).flatten().tolist()
                trues.extend(label)
                preds.extend(pred)
        # print(trues, preds, labels)
        report = classification_report(trues, preds, target_names=labels)
        return report


class Args:
    model_path = "model_hub/chinese-bert-wwm-ext"
    ckpt_path = "output/single-gpu-cls.pt"
    max_seq_len = 128
    ratio = 0.92
    epochs = 1
    eval_step = 50
    dev = False
    local_rank = None

deepspeed_config = {
  "train_micro_batch_size_per_gpu": 32,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-5
    }
  },
  "fp16": {
    "enabled": True
  },
  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": True,
    "allgather_bucket_size": 2e8,
    "overlap_comm": True,
    "reduce_scatter": True,
    "reduce_bucket_size": 2e8
  },
  "activation_checkpointing": {
    "partition_activations": True,
    "cpu_checkpointing": True,
    "contiguous_memory_optimization": True
  },
  "wall_clock_breakdown": True,
  "log_dist": False
}

def main():
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
                              batch_size=deepspeed_config["train_micro_batch_size_per_gpu"],
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collate.collate_fn)
    total_step = len(train_loader) * args.epochs
    args.total_step = total_step
    dev_loader = DataLoader(dev_data,
                            batch_size=deepspeed_config["train_micro_batch_size_per_gpu"],
                            shuffle=False,
                            num_workers=2,
                            collate_fn=collate.collate_fn)
    test_loader = dev_loader
    # =======================================

    # =======================================
    # 定义模型、优化器、损失函数
    config = BertConfig.from_pretrained(args.model_path, num_labels=6)
    model = BertForSequenceClassification.from_pretrained(args.model_path, config=config)
    model.cuda()
    criterion = torch.nn.CrossEntropyLoss()

    model_engine, optimizer, _, _ = deepspeed.initialize(
      config=deepspeed_config,
      model=model,
      model_parameters=model.parameters())

    args.local_rank = model_engine.local_rank
    # =======================================
    # 定义训练器
    trainer = Trainer(args,
              config,
              model_engine,
              criterion,
              optimizer)

    # 训练和验证
    trainer.train(train_loader, dev_loader)

    # 测试
    labels = list(label2id.keys())
    config = BertConfig.from_pretrained(args.model_path, num_labels=6)
    model = BertForSequenceClassification.from_pretrained(args.model_path, config=config)
    model.cuda()
    model_engine, optimizer, _, _ = deepspeed.initialize_from_checkpoint(
        config=deepspeed_config,
        model=model,
        model_parameters=model.parameters())
    model_engine.load_state_dict(torch.load(args.ckpt_path))
    report = trainer.test(model_engine, test_loader, labels)
    if args.local_rank == 0:
        print(report)
    # =======================================


if __name__ == '__main__':
    main()

