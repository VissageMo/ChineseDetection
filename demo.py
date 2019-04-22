import os
import pandas as pd
import csv

import numpy as np
from PIL import Image
import _pickle as pickle
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

is_cuda = torch.cuda.is_available()
num_gpu = 1
iters = 10
batch_size = 64
load_model = False
model_path = 'test.pkl'
train_path = '../../Datasets/ChineseDetection/train'
test_path = '../../Datasets/ChineseDetection/test1'
# test_labels = csv.reader(open('ChineseDetection/label-test1-fake.csv'))
test_labels = pd.read_csv('../../Datasets/ChineseDetection/label-test1-fake.csv')

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


class trainImageFolder(data.Dataset):
    def __init__(self, rootdir, transform=None, target_transform=None, loader=default_loader):
        images, labels = [], []
        for i, label in enumerate(LABELS):
            cur_path = '%s/%s' % (rootdir, label)
            path_list = os.listdir(cur_path)
            label_id = LABELDIC[label]
            for filename in path_list:
                image_path = cur_path + '/' + filename
                images.append(image_path)
                labels.append(label_id)

        self.rootdir = rootdir
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        sample, target = self.loader(str(self.images[index])), self.labels[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        sample = sample.expand(3, 256, 256)
        return sample, target

    def __len__(self):
        return len(self.images)


class testImageFolder(data.Dataset):
    def __init__(self, rootdir, transform=None, target_transform=None, loader=default_loader):
        images, labels = [], []
        for i in range(test_labels.shape[0]):
            image_name, label = test_labels.loc[i]
            image_path = rootdir + '/' + image_name
            first_label = LABELDIC[label[0]]
            # images.append(tuple([image_path, first_label]))
            images.append(image_path)
            labels.append(first_label)

        self.rootdir = rootdir
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    
    def __getitem__(self, index):
        sample, target = self.loader(str(self.images[index])), self.labels[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        sample = sample.expand(3, 256, 256)
        return sample, target
    
    def __len__(self):
        return len(self.images)


train_set = trainImageFolder(train_path, transform)
test_set = testImageFolder(test_path, transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

#*** Model ***#
if load_model:
    model = torch.load(model_path, map_location='cpu')
else:
    model = torchvision.models.resnet18(pretrained=True)
    # model = torchvision.models.vgg11_bn(num_classes=100)
    for param in model.parameters():
        param.requires_grad = True
    fc_features = model.fc.in_features
    model.fc = nn.Linear(2048, 100)
if is_cuda:
    model.cuda()
    print('Use GPU')
writer = SummaryWriter()

optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

for epoch in range(iters):
    model.train()
    for batch_id, (data, target) in enumerate(train_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        writer.add_scalar('Train/Loss', loss.item(), epoch)

        if (batch_id) % 50 == 0:
            _, pred = output.data.max(1)
            total = target.size(0)
            correct = (pred == target).sum().item()
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Correct：[{}/{}]'
                   .format(epoch+1, iters, (batch_id)*batch_size, 40000, loss.item(), correct, total))

    torch.save(model, model_path)


