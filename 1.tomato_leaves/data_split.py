import torch
import os
import shutil
import math

torch.cuda.is_available()
print(torch.__version__)
print('hello')

dataset_dir = './tomato_2'
classes_list = os.listdir(dataset_dir)
print('dataset_lists ', classes_list)
origin_dir = './tomato_2/all_tomato'
origin_classes_list = os.listdir(origin_dir)
print('dataset_classes_lists ', origin_classes_list)

train_dir = os.path.join(dataset_dir, 'train')
os.mkdir(train_dir)
test_dir = os.path.join(dataset_dir, 'test')
os.mkdir(test_dir)
val_dir = os.path.join(dataset_dir, 'val')
os.mkdir(val_dir)

for clss in origin_classes_list:
    os.mkdir(os.path.join(train_dir, clss))
    os.mkdir(os.path.join(val_dir, clss))
    os.mkdir(os.path.join(test_dir, clss))
    
# train test val 비율 맞추기
for clss in origin_classes_list:
    path = os.path.join(origin_dir, clss)
    fnames = os.listdir(path)
    
    train_size = math.floor(len(fnames) * 0.6)
    test_size = math.floor(len(fnames) * 0.2)
    val_size = math.floor(len(fnames) * 0.2)
    
    train_fnames = fnames[:train_size]
    print('train size(', clss,'): ',len(train_fnames))
    for fname in train_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(os.path.join(train_dir, clss), fname)
        shutil.copyfile(src, dst)

    val_fnames = fnames[train_size:(val_size + train_size)]
    print('val size(', clss,'): ',len(val_fnames))
    for fname in val_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(os.path.join(val_dir, clss), fname)
        shutil.copyfile(src, dst)

    test_fnames = fnames[(val_size + train_size):(val_size + train_size + test_size)]
    print('test size(', clss,'): ',len(test_fnames))
    for fname in test_fnames:
        src = os.path.join(path, fname)
        dst = os.path.join(os.path.join(test_dir, clss), fname)
        shutil.copyfile(src, dst)
