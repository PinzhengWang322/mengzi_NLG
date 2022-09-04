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
parser.add_argument('--test_num', default = 50, type = int)
parser.add_argument('--batch_size', default = 1, type = int)
parser.add_argument('--epochs', default = 4, type = int)
parser.add_argument('--key_threshold', default = 0.5, type = float)
parser.add_argument('--device', default = "cuda", type = str)
parser.add_argument('--gradient_accumulation_step', default = 4, type = int)
parser.add_argument('--test_samples', default = 10, type = int)
parser.add_argument('--model_path', default = "./save4.pt")
args = parser.parse_args()

if __name__ == '__main__':
    train_dataloader, test_dataloader = preprocess(args)

    tokenizer = T5Tokenizer.from_pretrained("../mengzi-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("../mengzi-t5-base").to(args.device)
    model.load_state_dict(torch.load(args.model_path))


    # 测试集测试
    for inputbatch, labelbatch in test_dataloader:
        if args.test_samples:
            args.test_samples -= 1
        else:
            break
        inputs = inputbatch[0]
        print("inputs:", inputs)
        input_ids = tokenizer(inputs, return_tensors="pt").input_ids.to(args.device)
        outputs = model.generate(input_ids, max_length = 500, num_beams = 5, no_repeat_ngram_size=7)
        print("outputs:", tokenizer.decode(outputs[0], skip_special_tokens=True))
        print("-"*40)

    # 手动测试

    # inputs = "2014成都车展：长安悦翔V7正式亮相|关键词：1.6L发动机, 成都车展, 长安悦翔V7, 新车尾部造型, 液晶显示屏, 倒梯形设计, 进气格栅, 上下进气口, 大灯, 铬装饰进行点缀|"
    # input_ids = tokenizer(inputs, return_tensors="pt").input_ids.to(args.device)
    # outputs = model.generate(input_ids, max_length = 500, num_beams = 5, no_repeat_ngram_size=10)
    # print("outputs:", tokenizer.decode(outputs[0], skip_special_tokens=True))

    # inputs = "标题：售22.98-32.38万奥德赛锐·混动上市|关键词：特别版车型, 车身科技感, 日版车辆, 魔术感应门, 电调收音机, |<厂商,广汽本田>, <级别,中型MPV>, <能源类型,油电混合>, <最大功率,138马力>, <变速箱,E-CVT无级变速>, <长宽高, 4866/1832/1465mm>, <0-100km/h加速时间, 5.9秒>, <百公里油耗, 7.8L/100km>"
    # input_ids = tokenizer(inputs, return_tensors="pt").input_ids.to(args.device)
    # outputs = model.generate(input_ids, max_length = 500, num_beams = 5, no_repeat_ngram_size=10)
    # print("outputs:", tokenizer.decode(outputs[0], skip_special_tokens=True))
