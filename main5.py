import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import sys
from torchlars import LARS
from torchvision import transforms
from torch.multiprocessing import Process, set_start_method, Manager
try:
     set_start_method('spawn')
except RuntimeError:
    pass

from loss import NTXentLoss, topKsim
from network import LeNet_rep, MLP
from dataset import Unlabeled_MNIST, Labeled_MNIST, Pseudo_Labeled_MNIST
from trainer5 import train, label

if __name__ == '__main__':

    device = torch.device("cuda:0")

    feature = LeNet_rep().to(device)
    represent = MLP(32).to(device)
    classifier = MLP().to(device)
    
    model = list(feature.parameters()) + list(represent.parameters())
    
    base_optimizer = torch.optim.SGD(model, lr=0.1)
    optimizer1 = LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
    optimizer2 = torch.optim.Adam(classifier.parameters(), lr=0.1)
    
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
    self_set = Unlabeled_MNIST("/home/compu/ymh/asynch/", self_transform)
    #self_set = Unlabeled_MNIST("/data1/ymh/asynch/", self_transform)
    #self_set = Augmented_MNIST("C://유민형//개인 연구//Asynchronous Interactive Classification//", self_transform)

    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    aug_transform = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.RandomResizedCrop(32),
        transforms.RandomAffine(0, shear=[-15, 15, -15, 15]),
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
    
    capacity = 300
    pseudo_set = Pseudo_Labeled_MNIST(capacity, cls_transform)
    
    self_size = 1024
    topK = 11

    self_loss = NTXentLoss(device, self_size)
    cls_loss = nn.CrossEntropyLoss()
    measure = topKsim(topK)

    feature.share_memory()
    represent.share_memory()
    classifier.share_memory()

    dataset1_manager = Manager()
    dataset1_list = dataset1_manager.list()
    dataset1_list.append(self_set)
    
    dataset2_manager = Manager()
    dataset2_list = dataset2_manager.list()
    dataset2_list.append(cls_set)
    
    dataset3_manager = Manager()
    dataset3_list = dataset3_manager.list()
    dataset3_list.append(pseudo_set)
    
    print("Appended")
    
    procs = []
    
    fn = sys.stdin.fileno()
    proc2 = Process(target=label, args=(fn, feature, represent, classifier, dataset1_list, dataset2_list, dataset3_list, measure, device, test_loader))
    proc2.start()
    procs.append(proc2)
    
    proc1 = Process(target=train, args=(feature, represent, classifier, dataset1_list, dataset2_list, dataset3_list, self_loss, cls_loss, optimizer1, optimizer2, device, aug_transform))
    proc1.start()
    procs.append(proc1)
    
    for proc in procs:
        proc.join()




