import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model
from transformers.optimization import  Adafactor 
from tqdm import tqdm
from operator import ne
import os
import torch
import argparse
import random
import numpy as np

from preprocess import preprocess

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--data_json_path', default = "../data/data.json")
parser.add_argument('--test_num', default = 100, type = int)
parser.add_argument('--batch_size', default = 4, type = int)
parser.add_argument('--epochs', default = 8, type = int)
parser.add_argument('--key_threshold', default = 0.5, type = float)
parser.add_argument('--device', default = "cuda", type = str)
parser.add_argument('--gradient_accumulation_step', default = 8, type = int)
args = parser.parse_args()

if __name__ == '__main__':
    train_dataloader, test_dataloader = preprocess(args)

    tokenizer = T5Tokenizer.from_pretrained("../mengzi-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("../mengzi-t5-base").to(args.device)   

    optimizer = Adafactor(
    model.parameters(),
    lr=1e-3,
    eps=(1e-30, 1e-3),
    clip_threshold=1.0,
    decay_rate=-0.8,
    beta1=None,
    weight_decay=0.0,
    relative_step=False,
    scale_parameter=False,
    warmup_init=False
    )

    
    model.train()

    for epoch_id in range(1,args.epochs + 1):
        
        pbar = tqdm(train_dataloader)
        pbar.set_description('Training Epoch {}'.format(epoch_id))
        step_id = 0
        for inputbatch, labelbatch in pbar:
            step_id += 1
            inputbatch=tokenizer.batch_encode_plus(inputbatch,padding=True,truncation=True,max_length=1000,return_tensors='pt')["input_ids"]
            labelbatch=tokenizer.batch_encode_plus(labelbatch,padding=True,truncation=True,max_length=1000,return_tensors="pt") ["input_ids"]
            inputbatch=inputbatch.to(args.device)
            labelbatch=labelbatch.to(args.device)

            # Forward propogation
            outputs = model(input_ids = inputbatch, labels = labelbatch)
            loss = outputs.loss
            loss_num=loss.item()
            logits = outputs.logits
            pbar.set_postfix(loss = loss.item())

            # calculating the gradients
            loss /= args.gradient_accumulation_step
            loss.backward()

            if step_id % args.gradient_accumulation_step == 0:
                #updating the params
                optimizer.step()
                # clear out the gradients of all Variables 
                optimizer.zero_grad()
        torch.save(model.state_dict(), 'save'+str(epoch_id)+'.pt')


        


