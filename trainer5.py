import torch
import numpy as np
import sys
import os
from torch.utils.data import DataLoader
import time


def train(feature, represent, classifier, dataset1_list, dataset2_list, dataset3_list, con_loss, cls_loss, optimizer1, optimizer2, device, transform):
    print("train start")
    iteration = 0
    self_size = 1024
    while True:
        dataset1 = dataset1_list[0] # Unlabeled
        dataset2 = dataset2_list[0] # Labeled
        dataset3 = dataset3_list[0] # Pseudo
        
        if dataset1.len >= self_size:
            feature.train()
            represent.train()
            
            optimizer1.zero_grad()
            self_loader = DataLoader(dataset1, batch_size=self_size, shuffle=True)
            
            x, _ = self_loader.__iter__().next()
            x1 = torch.randn_like(x)
            x2 = torch.randn_like(x)
            for i in range(self_size):
                x1[i] = transform(x[i])
                x2[i] = transform(x[i])
            x1 = x1.to(device)
            x2 = x2.to(device)
        
            rep1 = represent(feature(x1))
            rep2 = represent(feature(x2))
        
            contrastive_loss = con_loss(rep1, rep2)
            for_print_con = contrastive_loss.item()
                    
            contrastive_loss.backward()
            optimizer1.step()    
        else:
            for_print_con = 0
        
        if dataset2.len > 0:
            feature.eval()
            classifier.train()
            
            optimizer2.zero_grad()
            cls_size = min(dataset2.len, 64)
            cls_loader = DataLoader(dataset2, batch_size=cls_size, shuffle=True)
            
            x1, y1 = cls_loader.__iter__().next()
            x1 = x1.to(device)
            y1 = y1.long().to(device)
            
            out = classifier(feature(x1))
            classification_loss = cls_loss(out, y1)
            for_print_cls = classification_loss.item()
            
            if dataset3.len > 0:
                pseudo_size = min(dataset3.len, 64)
                pseudo_loader = DataLoader(dataset3, batch_size=pseudo_size, shuffle=True)
                
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
        
        else:
            for_print_cls = 0
            for_print_pseudo = 0
                            
        print("[Iter:%d][Cont:%f][Cls:%f][Pseudo:%f][N_cont:%d][N_cls:%d][N_pseudo:%d]" % (iteration, for_print_con, for_print_cls, for_print_pseudo, dataset1.len, dataset2.len, dataset3.len))        
        iteration += 1
        
        
def label(fileno, feature, represent, classifier, dataset1_list, dataset2_list, dataset3_list, measure, device, test_loader):
    sys.stdin = os.fdopen(fileno)
    print("labeling initialize")
    dataset1 = dataset1_list[0]
    topK = 10
    time.sleep(120)
    print("labeling start")
    while dataset1.len > 0:
        dataset1 = dataset1_list[0]
        dataset2 = dataset2_list[0]
        dataset3 = dataset3_list[0]
        
        self_size = min(dataset1.len, 1024)
        self_loader = DataLoader(dataset1, batch_size=self_size, shuffle=True)
            
        x, y = self_loader.__iter__().next()
        x = x.to(device)
        y = y.to(device)
        
        feature.eval()
        represent.eval()
        classifier.eval()        
        
        fea = feature(x)
        out = classifier(fea)
        b = 1/(out.var(dim=1) + 1)
        c = b/b.sum()
        d = np.random.choice(1024, 1, p = c.detach().cpu().numpy())[0]
        index = y[d]
        
        rep = represent(fea)
        
        #a = rep[:1]
        #b = rep[1:]
        a = rep[d].unsqueeze(0)
        
        #pseudo_index = measure.which(a,b) + 1
        pseudo_index = measure.which(a,rep)[1:]
        
        output = classifier(feature(x[pseudo_index]))
        pred = output.argmax(1)
        print(pred)
        
        #index = y[0]
        pseudo_index = y[pseudo_index]
        
        dataset1.show(index)
        
        img = dataset1.take(index)
        label = input()
        label = int(label)
        dataset2.labeling(img,label)
        
        for i in range(topK):
            index = pseudo_index[i].item()
            dataset1.show(index)
            img = dataset1.copy(index)
            dataset3.labeling(img,label)
        
        dataset1_list[0] = dataset1
        dataset2_list[0] = dataset2
        dataset3_list[0] = dataset3
        
        print("image is saved.")
        
        classifier.eval()
        accuracy = 0
        with torch.no_grad():
            correct = 0
            for x, y in test_loader:
                output = classifier(feature(x.float().to(device)))
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()
                    
        accuracy = correct / len(test_loader.dataset)
        print("[Accuracy:%f]" % accuracy)
        
    return
