import json
from torch.utils.data import DataLoader
import re

def filter(text):
    return re.sub(" ",'',text)


# Input:
# "title|keyword1, keyword2, keyword3|<entityA, relationX, entityB>, <entityC, relationY, entityD>"
# Output: 
# "body of the text"
def make_input(data, args):
    s = ""
    s += ("标题：" + re.sub("\n",'',filter(data['title'])) + "|")
    s += "关键词："
    for word, score in data['key_pharses']:
        
        if score < args.key_threshold: 
            s += "|"
            break
        if s[-1] != "：":
            s += ", "
        s += word
        
    
    for key in data['args']:
        s += "<" + key + "," + data['args'][key] + ">, "

    return s

def preprocess(args):
    f = open(args.data_json_path)
    data_lst = json.loads(f.read())
    train_data, test_data = data_lst[ : - args.test_num], data_lst[- args.test_num : ]
    
    train_dataset = [(make_input(data, args), "生成：" + data['content']) for data in train_data] 
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True)

    test_dataset = [(make_input(data, args), "生成：" + data['content']) for data in test_data] 
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False)

    return train_dataloader, test_dataloader