from vit_pytorch import ViT
from dataset import cifar100_dataset
import torch.optim as optim
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
EPOCH = 30
pre_epoch = 0
BATCH_SIZE = 128
LR = 0.01
CIFAR_PATH = "data"
model_name = "VIT"
net = ViT(
    image_size = 32,
    patch_size = 16,
    num_classes = 100,
    dim = 512,
    depth = 4,
    heads = 16,
    mlp_dim = 1024,
    dropout = 0,
    emb_dropout = 0.1
).to("cuda:0")
train_log_dir ="VIT-train"
train_writer = SummaryWriter(log_dir=train_log_dir)
test_log_dir = "VIT-test"
test_writer =  SummaryWriter(log_dir=test_log_dir)
trainloader, testloader = cifar100_dataset(CIFAR_PATH, train_batch_size = BATCH_SIZE,test_batch_size = BATCH_SIZE,num_workers=0)
length = len(trainloader)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
History = {"train_loss":[],"train_acc":[],"test_acc":[]}
for epoch in range(pre_epoch, EPOCH):
    print('\nEpoch: %d' % (epoch + 1))
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        sum_loss += loss.cpu().item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' 
              % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
    train_writer.add_scalar('Accuracy', float(100. * correct / total), epoch + 1)
    train_writer.add_scalar('Loss', sum_loss / (i + 1), epoch +1 )
    print('Waiting Test...')
    with torch.no_grad():
        correct = 0
        total = 0
        sum_loss = 0
        for i,data in enumerate(testloader,0):
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            sum_loss += loss.cpu().item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().sum()
        test_acc = float(100 * correct / total)
        test_loss= sum_loss / (i + 1)
        print('Test\'s ac is: %.3f%%' % test_acc,"loss is: %.03f" % test_loss)
        test_writer.add_scalar('Accuracy',test_acc, epoch + 1)
        test_writer.add_scalar('Loss',test_loss, epoch + 1)
    model_to_save = net.module if hasattr(net, 'module') else net  # Only save the model it-self
    torch.save(model_to_save.state_dict(), "model\%s_epoch_%d.pth" % (model_name,(epoch+1)))
print('Train has finished, total epoch is %d' % EPOCH)
train_writer.close()
test_writer.close()