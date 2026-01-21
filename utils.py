import pandas as pd
import torch.utils.data as data
from config import *
from transformers import BertTokenizer
import torch
import random

def get_ent_pos(lst):
    items = []
    for i in range(len(lst)):
        # B-ASP 开头
        if lst[i] == 1:
            item = [i]
            while True:
                i += 1
                # 到 I-ASP 结束
                if i >= len(lst) or lst[i] != 2:
                    items.append(item)
                    break
                else:
                    item.append(i)
        i += 1
    return items

# [CLS]这个手机外观时尚，美中不足的是拍照像素低。[SEP]
# 0 0 0 0 0 1 2 0 0 0 0 0 0 0 0 0 1 2 2 2 0 0 0
print(get_ent_pos([0,0,0,0,0,1,2,0,0,0,0,0,0,0,0,0,1,2,2,2,0,0,0]))
# [[5, 6], [16, 17, 18, 19]]

def get_ent_weight(max_len, ent_pos):
    cdm = []
    cdw = []

    for i in range(max_len):
        dst = min(abs(i - ent_pos[0]), abs(i - ent_pos[-1]))
        if dst <= SRD:
            cdm.append(1)
            cdw.append(1)
        else:
            cdm.append(0)
            cdw.append(1 / (dst - SRD + 1))
    return cdm, cdw

# print(get_ent_weight(23, [5,6]))
# exit()

class Dataset(data.Dataset):
    def __init__(self, type='train'):
        super().__init__()
        file_path = TRAIN_FILE_PATH if type == 'train' else TEST_FILE_PATH
        self.df = pd.read_csv(file_path)
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    def __len__(self):
        return len(self.df) - 1

    def __getitem__(self, index):
        # 相邻两个句子拼接
        text1, bio1, pola1 = self.df.loc[index]
        text2, bio2, pola2 = self.df.loc[index+1]
        text = text1 + ' ; ' + text2
        bio = bio1 + ' O ' + bio2
        pola = pola1 + ' -1 ' + pola2

        # 按自己的规则分词
        tokens = ['[CLS]'] + text.split(' ') + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # BIO标签转id
        bio_arr = ['O'] + bio.split(' ') + ['O']
        bio_label = [BIO_MAP[l] for l in bio_arr]
        
        # 情感值转数字
        pola_arr = ['-1'] + pola.split(' ') + ['-1']
        pola_label = list(map(int, pola_arr))
        return input_ids, bio_label, pola_label
    

if __name__ == '__main__':
    dataset = Dataset()
    loader = data.DataLoader(dataset)
    # loader = data.DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    print(next(iter(loader)))