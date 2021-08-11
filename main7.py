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
from dataset3 import Augmented_MNIST, Index_MNIST
from trainer7 import cont_train, cls_train, label

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
    
    normal_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
    
    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    aug_transform = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.RandomResizedCrop(32),
        transforms.Resize(32),
        transforms.RandomAffine(0, shear=[-15, 15, -15, 15]),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
    
    aug_set = Augmented_MNIST("/home/compu/ymh/asynch/", aug_transform)    
    index_set = Index_MNIST("/home/compu/ymh/asynch/", aug_transform, normal_transform)
        
    self_size = 2048
    topK = 15

    self_loss = NTXentLoss(device, self_size)
    cls_loss = nn.CrossEntropyLoss()
    measure = topKsim(topK)

    feature.share_memory()
    represent.share_memory()
    classifier.share_memory()
    
    dataset2_manager = Manager()
    dataset2_list = dataset2_manager.list()
    dataset2_list.append(index_set)
    
    index_manager = Manager()
    index_list = index_manager.list()
    index_list.append([i for i in range(60000)]) # unlabeled index
    index_list.append([]) # labeled index
    index_list.append([]) # pseudo labeled index
    
    print("Appended.")   
    
    procs = []
    
    fn = sys.stdin.fileno()
    proc1 = Process(target=label, args=(fn, feature, represent, classifier, dataset2_list, index_list, measure, device, test_loader))
    proc1.start()
    procs.append(proc1)
    
    proc2 = Process(target=cont_train, args=(feature, represent, aug_set, index_list, self_loss, optimizer1, device))
    proc2.start()
    procs.append(proc2)
    
    proc3 = Process(target=cls_train, args=(feature, classifier, dataset2_list, index_list, cls_loss, optimizer2, device))
    proc3.start()
    procs.append(proc3)
    
    for proc in procs:
        proc.join()




