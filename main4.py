import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import sys
from torchvision import transforms
from torch.multiprocessing import Process, set_start_method, Manager
try:
     set_start_method('spawn')
except RuntimeError:
    pass

from loss import NTXentLoss
from network import LeNet2
from dataset import Augmented_MNIST, Labeled_MNIST
from trainer4 import train, label

if __name__ == '__main__':

    device = torch.device("cuda:0")

    model = LeNet2().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    test_loader = DataLoader(
                datasets.MNIST(
                        "./data/mnist",
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                            transforms.Resize(32),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                            ])
                        ),
                batch_size=32, shuffle=False)
    
    self_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.ToTensor(),
        ])
    self_set = Augmented_MNIST("/data1/ymh/asynch/", self_transform)
    #self_set = Augmented_MNIST("C://유민형//개인 연구//Asynchronous Interactive Classification//", self_transform)

    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    aug_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(32),
        #transforms.RandomAffine(0, shear=[-15, 15, -15, 15]),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])

    cls_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),
        #transforms.RandomResizedCrop(32),
        #transforms.RandomAffine(0, shear=[-15, 15, -15, 15]),
        #transforms.RandomApply([color_jitter], p=0.8),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
    cls_set = Labeled_MNIST(cls_transform)

    self_size = 64

    self_loss = NTXentLoss(device, self_size)
    cls_loss = nn.CrossEntropyLoss()

    model.share_memory()

    dataset1_manager = Manager()
    dataset1_list = dataset1_manager.list()
    dataset1_list.append(self_set)
    
    dataset2_manager = Manager()
    dataset2_list = dataset2_manager.list()
    dataset2_list.append(cls_set)
    
    print("Appended")
    
    procs = []
    
    fn = sys.stdin.fileno()
    proc2 = Process(target=label, args=(fn, model, dataset1_list, dataset2_list, device, test_loader))
    proc2.start()
    procs.append(proc2)
    
    proc1 = Process(target=train, args=(model, dataset1_list, dataset2_list, self_loss, cls_loss, optimizer, device, aug_transform))
    proc1.start()
    procs.append(proc1)
    
    for proc in procs:
        proc.join()




