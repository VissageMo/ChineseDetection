import os

from PIL import Image
import torch
import torch.utils.data as data


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


def default_loader(path):
    return Image.open(path)


class fullImageFolder(data.Dataset):
    def __init__(self, rootdir, transform=None, target_transform=None, loader=default_loader):
        images, labels = [], []
        for _, label in enumerate(LABELS):
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
        # sample = sample.expand(3, 256, 256)
        return sample, target

    def __len__(self):
        return len(self.images)


def train_test_split(full_dataset, train_percent):
    train_size = int(len(full_dataset) * train_percent)
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = data.random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset