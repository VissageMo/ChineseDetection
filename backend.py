import os

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

is_cuda = torch.cuda.is_available()
num_gpu = 1
train_path = '../../Datasets/ChineseDetection/train'
test_path = '../../Datasets/ChineseDetection/test1' 
Labels = ['且', '世', '东', '九', '亭', '今', '从', '令', '作', '使', '侯', '元', 
          '光', '利', '印', '去', '受', '右', '司', '合', '名', '周', '命', '和', 
          '唯', '堂', '士', '多', '夜', '奉', '女', '好', '始', '字', '孝', '守',
          '宗', '官', '定', '宜', '室', '家', '寒', '左', '常', '建', '徐', '御', 
          '必', '思', '意', '我', '敬', '新', '易', '春', '更', '朝', '李', '来', 
          '林', '正', '武', '氏', '永', '流', '海', '深', '清', '游', '父', '物',
          '玉', '用', '申', '白', '皇', '益', '福', '秋', '立', '老', '臣', '良',
          '莫', '虎', '衣', '西', '起', '足', '身', '通', '遂', '重', '陵', '雨',
          '章', '高', '黄', '鼎']



def default_loader(path):
    return Image.open(path)

class imageFloder(data.Dataset):
    def __init__(self, rootdir, transform=None, target_transform=None, loader=default_loader):
        images, labels = [], []
        for i in range(len(Labels)):
            cur_path = '%s/%s' % (rootdir, Labels[i])
            path_list = os.listdir(cur_path)
            for filename in path_list:
                with open('%s/%s' % (cur_path, filename), 'rb') as fo:
                    # image = Image.open(fo)
                    # image = np.array(image)
                    images.append(tuple([('%s/%s' % (cur_path, filename)), i]))
                    # images.append(image)
                    labels.append(i)
        # images = np.array(images, dtype='float32')
        # labels = np.array(labels, dtype='int')
        self.rootdir = rootdir
        self.images = images
        self.labels = labels
        self.target_transform = target_transform
        self.loader = loader
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    def __len__(self):
        return len(self.images)


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5), (1.0))
])

trainset = torchvision.datasets.ImageFolder(train_path, transform)
imageLoader = torch.utils.data.DataLoader(trainset, batch_size=16)

#*** Model ***#
model = torchvision.models.resnet18(pretrained=False)
for param in model.parameters():
    param.requires_grad = True
model.fc = nn.Linear(2048, 100)
if is_cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

for epoch in range(10):
    model.train()
    for batch_id, (data, target) in enumerate(imageLoader):
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_id % 20 == 0:
            print(loss.item())

