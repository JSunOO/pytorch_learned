import torch
import os
import shutil
import math

torch.cuda.is_available()
print(torch.__version__)
print('hello')

dataset_dir = './tomato'
classes_list = os.listdir(dataset_dir)
print('dataset_lists ', classes_list)
train_dir = './tomato/train'
train_classes_list = os.listdir(train_dir)
print('train_lists ', train_classes_list)
train_3000_dir = os.path.join(dataset_dir, 'train_3000')
#train_3000_dir = './tomato/train_3000'
os.mkdir(train_3000_dir)


for clss in train_classes_list:
    os.mkdir(os.path.join(train_3000_dir, clss))
    path = os.path.join(train_dir, clss)
    fnames = os.listdir(path)
    train_size = math.floor(len(fnames) * 0.3)
    
    train_fnames = fnames[:train_size]
    print('train size(', clss,'): ',len(train_fnames))
    for fname in train_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(os.path.join(train_3000_dir, clss), fname)
        shutil.copyfile(src, dst)
