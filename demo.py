import os
import pandas as pd

import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter
from data import train_test_split, fullImageFolder
from model import resnet
from utils import adjust_learning_rate

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms


is_cuda = torch.cuda.is_available()
num_gpu = 1
iters = 10
batch_size = 64
init_learning_rate = 0.1
decay_point = [0.5, 0.8, 0.9, 0.95]
load_model = False
learning_rate_decay = False
model_path = 'test.pkl'
train_path = '../../Datasets/ChineseDetection/train'
test_path = '../../Datasets/ChineseDetection/test1'

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
    transforms.Resize((196, 196)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5), (1.0))
])

full_set = fullImageFolder(train_path, transform)
train_set, test_set = train_test_split(full_set, 0.8)
train_num, test_num, batch_num = len(train_set), len(test_set), train_num / batch_size
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

#*** Model ***#
if load_model:
    model = torch.load(model_path, map_location='cpu')
else:
    model = resnet(num_classes=100, input_channel=1)
if is_cuda:
    model.cuda()
    print('Use GPU')

#*** Init writer and optimizer ***#
writer = SummaryWriter()
optimizer = optim.SGD(model.parameters(), lr=init_learning_rate, momentum=0.9)
best_accuracy = 0

for epoch in range(iters):
    model.train()
    for batch_id, (data, target) in enumerate(train_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        writer.add_scalar('Train/Loss', loss.item(), epoch * batch_num + batch_id)
 
        if (batch_id) % 50 == 0 and batch_id > 0:
            _, pred = output.data.max(1)
            total = target.size(0)
            correct = (pred == target).sum().item()
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, iters, (batch_id)*batch_size, train_num, loss.item()))
    
    model.eval()
    with torch.no_grad():
        correct = 0
        for batch_id, (data, target) in enumerate(test_loader):
            if is_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            _, pred = output.data.max(1)
            total = target.size(0)
            correct += (pred == target).sum().item()
        accuracy = correct / test_num
        writer.add_scalar('Train/Accuracy', correct / test_num, epoch)
        print('################################################')
        print('Epoch [{}/{}] finished, Correct: [{}/{}]'.format(epoch+1, iters, correct, test_num))
        print('################################################')
    
    # change learning rate
    if learning_rate_decay:
        learning_rate = adjust_learning_rate(init_learning_rate, decay_point, accuracy=accuracy)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # save model if better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model, model_path)
        print('Save model successd as {}, accuracy: {}'.format(model_path, best_accuracy))


