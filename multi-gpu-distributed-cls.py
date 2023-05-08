import os
import time
import json
import random
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist

from collections import Counter
from tqdm import tqdm
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
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


class ClsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


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
                 optimizer):
        self.args = args
        self.config = config,
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def on_step(self, batch_data):
        # 第六步：根据local_rank将数据分发给指定的GPU
        label = batch_data["label"].cuda()
        input_ids = batch_data["input_ids"].cuda()
        token_type_ids = batch_data["token_type_ids"].cuda()
        attention_mask = batch_data["attention_mask"].cuda()
        output = self.model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            labels=label)
        logits = output[1]
        return logits, label

    def loss_reduce(self, loss):
        rt = loss.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.local_world_size
        return rt

    def output_reduce(self, outputs, targets):
        output_gather_list = [torch.zeros_like(outputs) for _ in range(self.args.local_world_size)]
        # 把每一个GPU的输出聚合起来
        dist.all_gather(output_gather_list, outputs)

        outputs = torch.cat(output_gather_list, dim=0)
        target_gather_list = [torch.zeros_like(targets) for _ in range(self.args.local_world_size)]
        # 把每一个GPU的输出聚合起来
        dist.all_gather(target_gather_list, targets)
        targets = torch.cat(target_gather_list, dim=0)
        return outputs, targets

    def train(self, train_loader, dev_loader=None, train_sampler=None):
        gloabl_step = 1
        best_acc = 0.
        if self.args.local_rank == 0:
            start = time.time()
        for epoch in range(1, self.args.epochs + 1):
            # 第四步：训练时每一个epoch打乱数据
            train_sampler.set_epoch(epoch)
            for step, batch_data in enumerate(train_loader):
                self.model.train()
                logits, label = self.on_step(batch_data)
                # 实际上这里计算的loss和output[0]是一样的
                loss = self.criterion(logits, label)
                # 第五步：等待所有GPU
                torch.distributed.barrier()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # 第七步：reduce计算损失
                loss = self.loss_reduce(loss)
                # 第八步：在主rank打印指标
                if self.args.local_rank == 0:
                    print("【train】 epoch：{}/{} step：{}/{} loss：{:.6f}".format(
                        epoch, self.args.epochs, gloabl_step, self.args.total_step, loss
                    ))
                gloabl_step += 1
                if self.args.dev:
                    # 第九步：在主rank保存模型
                    if gloabl_step % self.args.eval_step == 0:
                        loss, accuracy = self.dev(dev_loader)
                        if self.args.local_rank == 0:
                            print("【dev】 loss：{:.6f} accuracy：{:.4f}".format(loss, accuracy))
                            if accuracy > best_acc:
                                best_acc = accuracy
                                print("【best accuracy】 {:.4f}".format(best_acc))
                                torch.save(self.model.state_dict(), self.args.ckpt_path)
        if self.args.local_rank == 0:
            end = time.time()
            print("耗时：{}分钟".format((end - start) / 60))
        if not self.args.dev and self.args.local_rank == 0:
            torch.save(self.model.state_dict(), self.args.ckpt_path)

    def dev(self, dev_loader):
        self.model.eval()
        correct_total = 0
        num_total = 0
        loss_total = 0.
        with torch.no_grad():
            for step, batch_data in tqdm(enumerate(dev_loader)):
                logits, label = self.on_step(batch_data)
                loss = self.criterion(logits, label)
                torch.distributed.barrier()
                loss = self.loss_reduce(loss)
                loss_total += loss
                # 第十步：reduce得到结果
                logits, label = self.output_reduce(logits, label)
                logits = logits.detach().cpu().numpy()
                label = label.view(-1).detach().cpu().numpy()
                num_total += len(label)
                preds = np.argmax(logits, axis=1).flatten()
                correct_num = (preds == label).sum()
                correct_total += correct_num

        return loss_total, correct_total / num_total

    def test(self, model, test_loader, labels):
        self.model = model
        self.model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for step, batch_data in enumerate(test_loader):
                logits, label = self.on_step(batch_data)
                torch.distributed.barrier()
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
    ckpt_path = "output/multi-gpu-distributed-cls.pt"
    max_seq_len = 128
    ratio = 0.92
    train_batch_size = 32
    dev_batch_size = 32
    weight_decay = 0.01
    epochs = 1
    learning_rate = 3e-5
    eval_step = 50
    local_rank = None
    local_world_size = None
    device_ids = None
    rank = None
    dev = False


def main(local_world_size, local_rank):
    # =======================================
    # 设置相关参数
    # 从环境变量中获取参数
    set_seed()

    label2id = {
        "其他": 0,
        "喜好": 1,
        "悲伤": 2,
        "厌恶": 3,
        "愤怒": 4,
        "高兴": 5,
    }

    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK")
    }
    if local_rank == 0:
        for k, v in env_dict.items():
            print(k, v)
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    # 第一步：初始化
    dist.init_process_group(backend="nccl")
    n = torch.cuda.device_count() // local_world_size
    device_ids = [dist.get_rank()]
    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "
        + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids} \n", end=''
    )

    torch.cuda.set_device(int(env_dict["LOCAL_RANK"]))

    args = Args()
    args.local_world_size = local_world_size
    args.local_rank = int(env_dict["LOCAL_RANK"])
    args.device_ids = device_ids
    args.rank = dist.get_rank()
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
    # 第二步：DistributedSampler
    train_dataset = ClsDataset(train_data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_batch_size,
                              num_workers=2,
                              collate_fn=collate.collate_fn,
                              sampler=train_sampler)
    total_step = len(train_loader) * args.epochs
    args.total_step = total_step
    dev_dataset = ClsDataset(dev_data)
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_dataset)
    dev_loader = DataLoader(dev_dataset,
                            batch_size=args.dev_batch_size,
                            shuffle=False,
                            num_workers=2,
                            collate_fn=collate.collate_fn,
                            sampler=dev_sampler)
    test_loader = dev_loader
    # =======================================

    # =======================================
    # 定义模型、优化器、损失函数
    config = BertConfig.from_pretrained(args.model_path, num_labels=6)
    model = BertForSequenceClassification.from_pretrained(args.model_path,
                                                            config=config)
    # 第三步：封装模型
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.device_ids)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args)
    # =======================================

    # =======================================
    # 定义训练器，进行训练、验证和测试
    trainer = Trainer(args,
                      config,
                      model,
                      criterion,
                      optimizer)

    trainer.train(train_loader, dev_loader, train_sampler)

    labels = list(label2id.keys())
    config = BertConfig.from_pretrained(args.model_path, num_labels=6)
    model = BertForSequenceClassification.from_pretrained(args.model_path, config=config)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.device_ids)
    model.load_state_dict(torch.load(args.ckpt_path))
    report = trainer.test(model, test_loader, labels)
    if args.local_rank == 0:
        print(report)
    # =======================================

    # =======================================
    # 第十一步
    dist.destroy_process_group()
    # =======================================


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # 第零步：需要定义一个参数local-rank
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)
    p_args = parser.parse_args()
    main(p_args.local_world_size, p_args.local_rank)
