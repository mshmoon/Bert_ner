import argparse
import glob
import logging
import os
import json
import time
import numpy as np
import torch
from sklearn.metrics import classification_report
from models.transformers import WEIGHTS_NAME,BertConfig
from transformers import  BertForTokenClassification
from transformers import BertTokenizer
from models.bert_for_ner import BertSoftmaxForNer

from transformers import BertTokenizer

from torch.utils.data import DataLoader,TensorDataset,RandomSampler, SequentialSampler

from processors.ner_seq import convert_examples_to_features
from processors.ner_seq import ner_processors as processors
from processors.ner_seq import collate_fn

from torch.optim import Adam

MODEL_CLASSES = {'bert': (BertConfig, BertSoftmaxForNer)}

BERT_NAME = 'bert-base-chinese'

def load_and_cache_examples( task, tokenizer, data_type='train'):
    processor = processors[task]()
    # Load data features from cache or dataset file
    label_list = processor.get_labels()
    if data_type == 'train':
        examples = processor.get_train_examples("/home/mash/EasyBert/NER/datasets/cluener")

    if data_type == 'test':
        examples = processor.get_train_examples("/home/mash/EasyBert/NER/datasets/cluener")

    features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=label_list,
                                                max_seq_length=128,
                                                cls_token_at_end=0,
                                                pad_on_left=0,
                                                cls_token = tokenizer.cls_token,
                                                cls_token_segment_id=0,
                                                sep_token=tokenizer.sep_token,
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=0,
                                                )
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens,all_label_ids)
    return dataset

def train( train_dataset, model, tokenizer):
    """ Train the model """
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=64, collate_fn=collate_fn)
    t_total = 10000
    warmup_proportion = 0.1
    warmup_steps = int(t_total * warmup_proportion)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.0001,},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    #optimizer = AdamW(optimizer_grouped_parameters, lr= 3e-5)

    model = model.cuda()
    model.train()

    optimizer = Adam(model.parameters(),lr = 3e-5)
    for _ in range(int(20)):
        if _ >= 15:
            model.eval()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to("cuda") for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            outputs,_ = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            print(step,loss)

    return model

def test( test_dataset, model, tokenizer):
    """ Train the model """
    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=64, collate_fn=collate_fn)

    pred_list = []
    true_list = []
    for step, batch in enumerate(test_dataloader):

        batch = tuple(t.to("cuda") for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None}
        _,logits = model(**inputs)
        pred = torch.argmax(logits,dim = 2).cpu().numpy().tolist()
        true_label =  batch[3].cpu().numpy().tolist()
        for item in zip(true_label,pred):
            true_list.extend(item[0])
            pred_list.extend(item[1])

        print(len(pred_list),len(true_list))

    print(classification_report(true_list,pred_list))

def main():
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertSoftmaxForNer()  

    train_dataset = load_and_cache_examples("cluener",tokenizer, data_type='train')
    model = train(train_dataset,model,tokenizer)

    test_dataset = load_and_cache_examples("cluener",tokenizer, data_type='test')
    model = test(test_dataset,model,tokenizer)
if __name__ == "__main__":
    main()
