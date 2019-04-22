import os
import pandas as pd
import csv

import numpy as np
from PIL import Image
import _pickle as pickle

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms


test_path = '../../Datasets/ChineseDetection/test2'
model_path = 'test.pkl'
answer_path = 'answer.csv'

LABELS = ['且', '世', '东', '九', '亭', '今', '从', '令', '作', '使', '侯', '元', 
          '光', '利', '印', '去', '受', '右', '司', '合', '名', '周', '命', '和', 
          '唯', '堂', '士', '多', '夜', '奉', '女', '好', '始', '字', '孝', '守',
          '宗', '官', '定', '宜', '室', '家', '寒', '左', '常', '建', '徐', '御', 
          '必', '思', '意', '我', '敬', '新', '易', '春', '更', '朝', '李', '来', 
          '林', '正', '武', '氏', '永', '流', '海', '深', '清', '游', '父', '物',
          '玉', '用', '申', '白', '皇', '益', '福', '秋', '立', '老', '臣', '良',
          '莫', '虎', '衣', '西', '起', '足', '身', '通', '遂', '重', '陵', '雨',
          '章', '高', '黄', '鼎']
LABELDIC, ANTILABELDICT = {}, {}
for i, label in enumerate(LABELS):
    LABELDIC[label] = i
    ANTILABELDICT[i] = label

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5), (1.0))
])


def default_loader(path):
    return Image.open(path)


class testImageFolder(data.Dataset):
    def __init__(self, rootdir, transform=None, target_transform=None, loader=default_loader):
        images, files, labels = [], [], []
        path_list = os.listdir(rootdir)
        for filename in path_list:
            image_path = rootdir + '/' + filename
            images.append(image_path)
            files.append(filename)

        self.rootdir = rootdir
        self.images = images
        self.files = files
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    
    def __getitem__(self, index):
        sample = self.loader(str(self.images[index]))
        if self.transform is not None:
            sample = self.transform(sample)
        sample = sample.expand(3, 256, 256)
        return sample
    
    def __len__(self):
        return len(self.images)

test_set = testImageFolder(test_path, transform)

model = torch.load(model_path)
if torch.cuda.is_available():
    print('GPU model')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)


model.eval()
test_loss, correct = 0, 0
with torch.no_grad():
    answer_list = []
    for i in range(len(test_set.images)):
        image = test_set[i]
        image = image.expand([1, 3, 256, 256])
        image = image.to(device)
        # image = Variable(image)
        output = model(image).cpu()
        _, pred = output.data.topk(5, dim=1)
        chinese_answer = ''
        for j in pred.numpy().tolist()[0]:
            chinese_answer += ANTILABELDICT[j]
        answer_list.append([test_set.files[i], chinese_answer])
        if i % 100 == 0 and i > 0:
            print('{}/{}'.format(i, len(test_set.images)))

name = ['filename', 'label']
df = pd.DataFrame(columns=name, data=answer_list)
df.to_csv(answer_path, encoding='utf8')
