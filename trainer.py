import torch
import numpy as np
import sys
import os
from torch.utils.data import DataLoader

def train(network, dataset1_list, dataset2_list, con_loss, cls_loss, optimizer, device, transform):
    print("train start")
    iteration = 0
    while True:
        dataset1 = dataset1_list[0]
        dataset2 = dataset2_list[0]
        
        network.train()
        optimizer.zero_grad()
        
        if dataset1.len > 0:
            self_size = min(dataset1.len, 32)
            self_loader = DataLoader(dataset1, batch_size=self_size, shuffle=True)
            
            x, _ = self_loader.__iter__().next()
            x1 = torch.randn_like(x)
            x2 = torch.randn_like(x)
            for i in range(self_size):
                x1[i] = transform(x[i])
                x2[i] = transform(x[i])
            x1 = x1.to(device)
            x2 = x2.to(device)
        
            rep1 = network(x1)
            rep2 = network(x2)
        
            contrastive_loss = con_loss(rep1, rep2)
            for_print_con = contrastive_loss.item()
        else:
            contrastive_loss = 0
            for_print_con = 0
        
        if dataset2.len > 0:
            cls_size = min(dataset2.len, 32)
            cls_loader = DataLoader(dataset2, batch_size=cls_size, shuffle=True)

            x, y = cls_loader.__iter__().next()
            x = x.to(device)
            y = y.long().to(device)
        
            out = network(x)
            
            classification_loss = cls_loss(out, y)
            for_print_cls = classification_loss.item()
        else:
            classification_loss = 0
            for_print_cls = 0
        
        loss = contrastive_loss + classification_loss
        loss.backward()
        optimizer.step()
        print("[Iteration:%d] [Cont:%f] [Cls:%f] [B_cont:%d] [B_cls:%d]" % (iteration, for_print_con, for_print_cls, dataset1.len, dataset2.len))
        
        iteration += 1
    
def label(fileno, network, dataset1_list, dataset2_list, device, test_loader):
    sys.stdin = os.fdopen(fileno)
    print("labeling start")
    dataset1 = dataset1_list[0]
    while dataset1.len > 0:
        dataset1 = dataset1_list[0]
        dataset2 = dataset2_list[0]
        
        self_size = min(dataset1.len, 32)
        self_loader = DataLoader(dataset1, batch_size=self_size, shuffle=True)
            
        x, y = self_loader.__iter__().next()
        x = x.to(device)
        y = y.to(device)
        
        network.eval()
        out = network(x)
        #index = y[out.var(dim=1).argmin().item()]
        index = y[np.random.choice(self_size, 1)[0]]
        
        dataset1.show(index)
        
        img = dataset1.take(index)
        print("image is saved.")
        label = input()
        label = int(label)
        dataset2.labeling(img,label)
        
        dataset1_list[0] = dataset1
        dataset2_list[0] = dataset2
        
        accuracy = 0
        with torch.no_grad():
            correct = 0
            for x, y in test_loader:
                output = network(x.float().to(device))
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()
                    
        accuracy = correct / len(test_loader.dataset)
        print("[Accuracy:%f]" % accuracy)
        
    return
