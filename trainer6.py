import torch
import numpy as np
import sys
import os
from torch.utils.data import DataLoader
import time

def cont_train(feature, represent, dataset1_list, con_loss, optimizer1, device):
    print("Contrastive training start.")
    iteration = 0
    self_size = 2048
    
    dataset1 = dataset1_list[0] # Augmented set
    self_loader = DataLoader(dataset1, batch_size=self_size, shuffle=True, drop_last=True)
    
    while True:
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

def cls_train(feature, classifier, dataset3_list, dataset4_list, cls_loss, optimizer2, device):
    print("Classifying training start.")
    epoch = 0
    
    while True:
        dataset3 = dataset3_list[0] # Labeled
        dataset4 = dataset4_list[0] # Pseudo
        
        if dataset3.len > 0:
            cls_size = min(dataset3.len, 64)
            cls_loader = DataLoader(dataset3, batch_size=cls_size, shuffle=True)
            
            for x1, y1 in cls_loader:
                feature.eval()
                classifier.train()
                optimizer2.zero_grad()
                
                x1 = x1.to(device)
                y1 = y1.long().to(device)
            
                out = classifier(feature(x1))
                classification_loss = cls_loss(out, y1)
                for_print_cls = classification_loss.item()
                
                if dataset4.len > 0:
                    pseudo_size = min(dataset4.len, 64)
                    pseudo_loader = DataLoader(dataset4, batch_size=pseudo_size, shuffle=True, drop_last=True)
                
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
                
            print("[Cls_epoch:%d][Cls:%f][Pseudo:%f][N_cls:%d][N_pseudo:%d]" % (epoch, for_print_cls, for_print_pseudo, dataset3.len, dataset4.len))        
            epoch += 1
       
        
def label(fileno, feature, represent, classifier, dataset2_list, dataset3_list, dataset4_list, measure, device, test_loader):
    sys.stdin = os.fdopen(fileno)
    print("labeling is initialized.")
    dataset2 = dataset2_list[0]
    topK = 10
    time.sleep(120)
    print("labeling start")
    while dataset2.len > 0:
        dataset2 = dataset2_list[0] # Unlabeled set
        dataset3 = dataset3_list[0] # Labeled set
        dataset4 = dataset4_list[0] # Pseudo labeled set
        
        self_size = min(dataset2.len, 2048)
        self_loader = DataLoader(dataset2, batch_size=self_size, shuffle=True)
        
        x, y = self_loader.__iter__().next()
        x = x.to(device)
        y = y.to(device)
        
        feature.eval()
        represent.eval()
        classifier.eval()
        with torch.no_grad():
            fea = feature(x)
            rep = represent(fea)
            #vec = classifier(fea)
        '''
        ## ---- sampling method ---- ##
        classifier.eval()        
        with torch.no_grad():        
            out = classifier(fea)
        b = 1/(out.var(dim=1) + 1)
        c = b/b.sum()
        d = np.random.choice(1024, 1, p = c.detach().cpu().numpy())[0]
        index = y[d]
        
        a = rep[d].unsqueeze(0)
        pseudo_index = measure.which(a,rep)[1:]
        
        
        '''
        '''
        c = vec[:1]
        d = vec[1:]
        
        _, cls_index = measure.which(c,d)
        cls_index = cls_index + 1
        cls_index = y[cls_index]
        '''
        ## ---- sampling method ---- ##
        a = rep[:1]
        b = rep[1:]
        
        similarity, pseudo_index = measure.which(a,b)
        pseudo_index = pseudo_index + 1
        
        index = y[0]
        pseudo_index = y[pseudo_index]
        
        dataset2.show(index, 1)
        
        img = dataset2.take(index)
        label = input()
        label = int(label)
        dataset3.labeling(img,label)
        #dataset4.check(index)
        
        for i in range(topK):
            index = pseudo_index[i].item()
            #dataset2.show(index, similarity[i].item())
            img = dataset2.copy(index)
            dataset4.labeling(img,label,index)
        
        dataset2_list[0] = dataset2
        dataset3_list[0] = dataset3
        dataset4_list[0] = dataset4
        
        '''
        #time.sleep(1)
        pseudo_index = pseudo_index.cpu().numpy().tolist()
        cls_index = cls_index.cpu().numpy().tolist()
        intersection = list(set(pseudo_index) & set(cls_index))
        for i in range(len(intersection)):
            index = intersection[i]
            dataset2.show(index, 2)
        '''
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
    return
