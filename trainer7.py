import torch
import numpy as np
import sys
import os
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import time

def cont_train(feature, represent, dataset1, index_list, con_loss, optimizer1, device):
    print("Contrastive training start.")
    iteration = 0
    self_size = 2048
    
    self_loader = DataLoader(dataset1, batch_size=self_size, shuffle=True, drop_last=True)
    
    unlabeled_index = index_list[0] # Unlabeled
    while len(unlabeled_index):
        unlabeled_index = index_list[0] # Unlabeled
        for x1, x2 in self_loader:
            feature.train()
            represent.train()
            optimizer1.zero_grad()
            
            x1 = x1.to(device)
            x2 = x2.to(device)
            
            rep1 = represent(feature(x1))
            rep2 = represent(feature(x2))
    
            contrastive_loss = con_loss(rep1, rep2)
            for_print_con = contrastive_loss.item()
                    
            contrastive_loss.backward()
            optimizer1.step()
                
            print("[Cont_Iter:%d][Cont:%f]" % (iteration, for_print_con))        
            iteration += 1

def cls_train(feature, classifier, dataset2_list, index_list, cls_loss, optimizer2, device):
    print("Classifying training start.")
    epoch = 0
    
    while True:
        dataset2 = dataset2_list[0] # label 때문에 가져와야한다.
        dataset2.aug_default = True
        
        unlabeled_index = index_list[0] # Unlabeled
        labeled_index = index_list[1] # Labeled
        pseudo_index = index_list[2] # Pseudo
                
        if len(labeled_index) > 0:
            labeled_sampler = SubsetRandomSampler(labeled_index)
            cls_size = min(len(labeled_index), 64)
            cls_loader = DataLoader(dataset2, batch_size=cls_size, sampler=labeled_sampler)
            
            for x1, y1 in cls_loader:
                feature.eval()
                classifier.train()
                optimizer2.zero_grad()
                
                x1 = x1.to(device)
                y1 = y1.long().to(device)
            
                out = classifier(feature(x1))
                classification_loss = cls_loss(out, y1)
                for_print_cls = classification_loss.item()
                
                if len(pseudo_index) > 0:
                    pseudo_sampler = SubsetRandomSampler(pseudo_index)
                    pseudo_size = min(len(pseudo_index), 64)
                    pseudo_loader = DataLoader(dataset2, batch_size=pseudo_size, drop_last=True, sampler=pseudo_sampler)
                
                    x2, y2 = pseudo_loader.__iter__().next()
                    x2 = x2.to(device)
                    y2 = y2.long().to(device)
                    
                    out = classifier(feature(x2))
                    pseudo_classification_loss = cls_loss(out, y2)
                    for_print_pseudo = pseudo_classification_loss.item()
                
                else:
                    pseudo_classification_loss = 0
                    for_print_pseudo = 0
                
                loss = classification_loss + pseudo_classification_loss
                loss.backward()
                optimizer2.step()
                
            print("[Cls_epoch:%d][Cls:%f][Pseudo:%f][N_cls:%d][N_pseudo:%d][N_ul:%d]" % (epoch, for_print_cls, for_print_pseudo, len(labeled_index), len(pseudo_index), len(unlabeled_index)))
            epoch += 1
       
        
def label(fileno, feature, represent, classifier, dataset2_list, index_list, measure, device, test_loader):
    sys.stdin = os.fdopen(fileno)
    print("labeling is initialized.")
        
    unlabeled_index = index_list[0] # Unlabeled
    
    topK = 10
    time.sleep(120)
    print("labeling start")
    
    while len(unlabeled_index) > 1:
        dataset2 = dataset2_list[0]
        dataset2.aug_default = False
        
        unlabeled_index = index_list[0] # Unlabeled
        labeled_index = index_list[1] # Labeled
        pseudo_index = index_list[2] # Pseudo
        
        #unlabeled_sampler = SubsetRandomSampler(unlabeled_index)
        #self_size = min(len(unlabeled_index), 2048)
        #self_loader = DataLoader(dataset2, batch_size=self_size, sampler=unlabeled_sampler)
        self_loader = DataLoader(dataset2, batch_size=2048, shuffle=True)
        
        x, y = self_loader.__iter__().next()
        x = x.to(device)
        y = y.to(device)
        
        ## ---- labeling ---- ##
        time.sleep(1)
        #index = y[0].item()
        index = np.random.choice(unlabeled_index, 1)[0]
        dataset2.show(index)
        
        label = input()
        label = int(label)
        dataset2.labeling(index, label)
        unlabeled_index.remove(index)
        labeled_index.append(index)
        
        ## ---- pseudo labeling ---- ##
        feature.eval()
        represent.eval()
        classifier.eval()
        with torch.no_grad():
            fea = feature(x)
            rep = represent(fea)
        
        #a = rep[:1]
        #b = rep[1:]
        
        a, _ = dataset2.__getitem__(index)
        a = a.unsqueeze(0)
        a = a.to(device)
        with torch.no_grad():
            a = feature(a)
            a = represent(a)
        
        b = rep
        
        _, similar_index = measure.which(a,b)
        #similar_index = similar_index + 1
                
        with torch.no_grad():
            fea = feature(x[similar_index])
            output = classifier(fea)
        
        y = y[similar_index]
        filtered_index = torch.topk(output.var(dim=1), topK)[1]
        filtered_index = y[filtered_index]
        
        for i in range(topK):
            index = filtered_index[i].item()
            if index in unlabeled_index:
                dataset2.show(index)
                dataset2.labeling(index,label)
                unlabeled_index.remove(index)
                pseudo_index.append(index)
        
        if len(pseudo_index) > 300:
            howmany = len(pseudo_index) - 300
            for i in range(howmany):
                index = pseudo_index.pop(0)
                unlabeled_index.append(index)
        
        dataset2_list[0] = dataset2
        
        index_list[0] = unlabeled_index
        index_list[1] = labeled_index
        index_list[2] = pseudo_index
        
        print("image is saved.")
        
        feature.eval()
        classifier.eval()
        accuracy = 0
        with torch.no_grad():
            correct = 0
            for x, y in test_loader:
                output = classifier(feature(x.float().to(device)))
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()
                    
        accuracy = correct / len(test_loader.dataset)
        print("*"*50)
        print(" ")
        print(" ")
        print("[Accuracy:%f]" % accuracy)
        print(" ")
        print(" ")
        print("*"*50)
    
    ## ---- labeling ---- ##
    dataset2 = dataset2_list[0]
    dataset2.aug_default = False
        
    unlabeled_index = index_list[0] # Unlabeled
    labeled_index = index_list[1] # Labeled
    
    #time.sleep(1)
    index = unlabeled_index[0]
    dataset2.show(index)
        
    label = input()
    label = int(label)
    dataset2.labeling(index, label)
    unlabeled_index.remove(index)
    labeled_index.append(index)
    
    dataset2_list[0] = dataset2
        
    index_list[0] = unlabeled_index
    index_list[1] = labeled_index
    
    while len(pseudo_index) > 1:
        dataset2 = dataset2_list[0]
        dataset2.aug_default = False
        
        labeled_index = index_list[1] # Labeled
        pseudo_index = index_list[2] # Pseudo
        
        pseudo_sampler = SubsetRandomSampler(pseudo_index)
        self_size = min(len(pseudo_index), 300)
        self_loader = DataLoader(dataset2, batch_size=self_size, sampler=pseudo_sampler)
        
        x, y = self_loader.__iter__().next()
        x = x.to(device)
        y = y.to(device)
        
        ## ---- labeling ---- ##
        #time.sleep(1)
        index = y[0].item()
        dataset2.show(index)
        
        label = input()
        label = int(label)
        dataset2.labeling(index, label)
        pseudo_index.remove(index)
        labeled_index.append(index)
        
        ## ---- pseudo label correcting ---- ##
        feature.eval()
        represent.eval()
        classifier.eval()
        with torch.no_grad():
            fea = feature(x)
            rep = represent(fea)
        
        a = rep[:1]
        b = rep[1:]
        
        _, similar_index = measure.which(a,b)
        similar_index = similar_index + 1
                
        with torch.no_grad():
            fea = feature(x[similar_index])
            output = classifier(fea)
        
        y = y[similar_index]
        filtered_index = torch.topk(output.var(dim=1), topK)[1]
        filtered_index = y[filtered_index]
        
        for i in range(topK): # correcting
            index = filtered_index[i].item()
            dataset2.labeling(index,label)
        
        dataset2_list[0] = dataset2
        
        index_list[1] = labeled_index
        index_list[2] = pseudo_index
        
        print("image is saved.")
        
        feature.eval()
        classifier.eval()
        accuracy = 0
        with torch.no_grad():
            correct = 0
            for x, y in test_loader:
                output = classifier(feature(x.float().to(device)))
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()
                    
        accuracy = correct / len(test_loader.dataset)
        print("*"*50)
        print(" ")
        print(" ")
        print("[Accuracy:%f]" % accuracy)
        print(" ")
        print(" ")
        print("*"*50)
    
    dataset2 = dataset2_list[0]
    dataset2.aug_default = False
        
    labeled_index = index_list[1] # Labeled
    pseudo_index = index_list[2] # Pseudo
    
    ## ---- labeling ---- ##
    #time.sleep(1)
    index = pseudo_index[0]
    dataset2.show(index)
        
    label = input()
    label = int(label)
    dataset2.labeling(index, label)
    pseudo_index.remove(index)
    labeled_index.append(index)    
    
    return
