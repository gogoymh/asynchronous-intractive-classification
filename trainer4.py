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
        
        if dataset2.len > 0:
            cls_size = min(dataset2.len, 32)
            cls_loader = DataLoader(dataset2, batch_size=cls_size, shuffle=True)
            '''
            if dataset2.len % 32 == 0 or dataset2.len % 32 >= 10:
                cls_loader = DataLoader(dataset2, batch_size=cls_size, shuffle=True)
            else:
                cls_loader = DataLoader(dataset2, batch_size=cls_size, shuffle=True, drop_last=True)
            '''
            for x, y in cls_loader:
                optimizer.zero_grad()
                x = x.to(device)
                y = y.long().to(device)
        
                out, _ = network(x)
            
                classification_loss = cls_loss(out, y)
                for_print_cls = classification_loss.item()
                
                if dataset1.len > 0:
                    self_size = min(dataset1.len, 64)
                    self_loader = DataLoader(dataset1, batch_size=self_size, shuffle=True)
            
                    x, _ = self_loader.__iter__().next()
                    x1 = torch.randn_like(x)
                    x2 = torch.randn_like(x)
                    for i in range(self_size):
                        x1[i] = transform(x[i])
                        x2[i] = transform(x[i])
                    x1 = x1.to(device)
                    x2 = x2.to(device)
        
                    _, rep1 = network(x1)
                    _, rep2 = network(x2)
        
                    contrastive_loss = con_loss(rep1, rep2)
                    for_print_con = contrastive_loss.item()
                    
                else:
                    classification_loss = 0
                    for_print_con = 0
                    
                loss = classification_loss + contrastive_loss
                loss.backward()
                optimizer.step()
                
                print("[Iteration:%d] [Cont:%f] [Cls:%f] [B_cont:%d] [B_cls:%d]" % (iteration, for_print_con, for_print_cls, dataset1.len, dataset2.len))        
                iteration += 1
        
        else:
            optimizer.zero_grad()
            self_size = min(dataset1.len, 64)
            self_loader = DataLoader(dataset1, batch_size=self_size, shuffle=True)
            
            x, _ = self_loader.__iter__().next()
            x1 = torch.randn_like(x)
            x2 = torch.randn_like(x)
            for i in range(self_size):
                x1[i] = transform(x[i])
                x2[i] = transform(x[i])
            x1 = x1.to(device)
            x2 = x2.to(device)
        
            _, rep1 = network(x1)
            _, rep2 = network(x2)
        
            contrastive_loss = con_loss(rep1, rep2)
            for_print_con = contrastive_loss.item()
                    
            contrastive_loss.backward()
            optimizer.step()
            
            for_print_cls = 0
            
            print("[Iteration:%d] [Cont:%f] [Cls:%f] [B_cont:%d] [B_cls:%d]" % (iteration, for_print_con, for_print_cls, dataset1.len, dataset2.len))        
            iteration += 1
        
        
def label(fileno, network, dataset1_list, dataset2_list, device, test_loader):
    sys.stdin = os.fdopen(fileno)
    print("labeling start")
    dataset1 = dataset1_list[0]
    while dataset1.len > 0:
        dataset1 = dataset1_list[0]
        dataset2 = dataset2_list[0]
        
        self_size = min(dataset1.len, 64)
        self_loader = DataLoader(dataset1, batch_size=self_size, shuffle=True)
            
        x, y = self_loader.__iter__().next()
        x = x.to(device)
        y = y.to(device)
        
        network.eval()

                
        out, _ = network(x)
        b = 1/(out.var(dim=1) + 1)
        c = b/b.sum()
        d = np.random.choice(64, 1, p = c.detach().cpu().numpy())[0]
        index = y[d]
        
        
        '''
        if dataset2.len > 10:
            out, _ = network(x)
            b = 1/(out.var(dim=1) + 0.01)
            c = b/b.sum()
            d = np.random.choice(64, 1, p = c.detach().cpu().numpy())[0]
            index = y[d]
        else:
            index = y[np.random.choice(self_size, 1)[0]]
        '''
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
                output, _ = network(x.float().to(device))
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()
                    
        accuracy = correct / len(test_loader.dataset)
        print("[Accuracy:%f]" % accuracy)
        
    return
