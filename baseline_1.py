import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import multiprocessing as mp
from multiprocessing import freeze_support

"""
baseline 모델을 설계 및 학습하기 위한 준비
"""

is_cuda = torch.cuda.is_available()
DEVICE = torch.device('cuda' if is_cuda else 'cpu')

print('Current cuda device is', DEVICE)

Batch_Size = 256
Epoch = 30

transform_base = transforms.Compose([transforms.Resize((64,64)),
                                                       transforms.ToTensor()])

train_dataset = ImageFolder(root='./tomato/train_3000', transform=transform_base)

val_dataset = ImageFolder(root='./tomato/val', transform=transform_base)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=Batch_Size, shuffle=True,
                                           num_workers=4)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=Batch_Size, shuffle=True,
                                           num_workers=4)

"""
baseline 모델을 본격적으로 설계
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool =  nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 33)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout(x, p=0.25, training=self.training)
        
        x = x.view(-1, 4096)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
    
model_base = Net().to(DEVICE) #정의한 cnn모댈 Net()의 새로운 객체를 생성 후 to(DEVICE)를 통해 모델을 현재 사용중인 장비에 할당함
optimizer = optim.Adam(model_base.parameters(), lr=0.001)

"""
   모델 학습을 위한 함수
"""
def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
"""
   모델 평가를 위한 함수
"""
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            
            test_loss = F.cross_entropy(output,
                        target, reduction='sum').item()
            
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, test_accuracy #측정한 정확도와 손실을 반환

"""
    모델 학습 실행하기
"""
import time
import copy
import os

def train_baseline(model, train_loader, 
                   val_loader, optimizer, num_epoch = 30):
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(1, num_epoch + 1):
        since = time.time()
        train(model, train_loader, optimizer)
        train_loss, train_acc = evaluate(model, train_loader)
        val_loss, val_acc = evaluate(model, val_loader)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
        time_elapsed = time.time() - since
        print('------------epoch{}-------------'.format(epoch))
        print('train loss: {: .4f}, accuracy: {: .2f}%'.format(train_loss, train_acc))
        print('val loss: {: .4f}, accuracy: {: .2f}%'.format(val_loss, val_acc))
        print('Completed in {: .0f}m {: .0f}s'.format(time_elapsed//60, time_elapsed%60))
        
    model.load_state_dict(best_model_wts)
    return model

base = train_baseline(model_base, train_loader, 
                      val_loader, optimizer, Epoch)

torch.save(base, 'baseline.pt')

if __name__ == '__main__':
    freeze_support()

"""
   transfer learning을 위한 준비
"""
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize([64,64]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(52),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406,
                              0.229, 0.224, 0.225])
    ]), 
    'val': transforms.Compose([
        transforms.Resize([64,64]),
        transforms.RandomCrop(52),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406,
                              0.229, 0.224, 0.225])
    ])
}

data_dir = './tomato'
image_datasets = {x: ImageFolder(root=os.path.join(data_dir, x), 
            transform=data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.Dataloader(image_datasets[x],
            batch_size=Batch_Size, shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

"""
   pre-trained model 불러오기
"""
from torchvision import models

resnet = models.resnet50(pretrained=True)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 33)
resnet = resnet.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, 
                        resnet.parameters()), lr=0.001)

from torch.optim import lr_scheduler

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,
                                       step_size=7, gamma=0.1)

"""
   pre_trained model의 일부 layer freeze하기
"""
ct = 0
for child in resnet.children():
    ct += 1
    if ct < 6:
        for param in child.parameters():
            param.requires_grad = False
            
"""
   transfer learning 모델 학습과 검증을 위한 함수
"""
def train_baseline(model, criterion, optimizer, scheduler, num_epoch = 25):
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epoch):
        print('------------epoch{}-------------'.format(epoch + 1))
        since = time.time()
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
                l_r = [x['lr'] for x in optimizer_ft.param_groups]
                print('learning rate: ', l_r)
                
            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_corrects.double()/dataset_sizes[phase]
            
            print('{} loss: {: .4f}, acc: {: .4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
        time_elapsed = time.time() - since
        print('Completed in {: .0f}m {: .0f}s'.format(time_elapsed//60, time_elapsed%60))
        
    print('Best val acc: {: 4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    
    return model    

"""
   모델 학습 실행하기
"""
model_resnet50 = train_dataset(resnet, criterion, optimizer_ft, 
                      exp_lr_scheduler, num_epochs=Epoch)

torch.save(model_resnet50, 'resnet_50.pt')