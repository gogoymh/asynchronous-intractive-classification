import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
import os
import matplotlib.pyplot as plt


class Pseudo_Labeled_MNIST(Dataset):
    def __init__(self, capacity, transform):
        super().__init__()

        self.transform = transform
        self.capacity = capacity
        
        self.mnist = np.empty((self.capacity,28,28,1))
        self.mnist = self.mnist.astype('uint8')
        self.label = torch.empty((self.capacity))
        
        self.indicies = []
        self.len = len(self.indicies)
        self.index = 0
        
    def __getitem__(self, index):
        
        img = self.mnist[self.indicies[index]]
        img = self.transform(img)
        
        label = self.label[self.indicies[index]]
        
        return img, label
    
    def __len__(self):
        return self.len
        
    def labeling(self, img, label):
        self.indicies.append(self.index)
        self.mnist[self.index] = img
        self.label[self.index] = label
        self.len = len(self.indicies)
        self.index += 1
        if self.index >= self.capacity:
            self.index = 0 # reset: first in, first out
        #print("Pseudo Labeled set length is %d." % self.len)
        
    def show(self, index):
        img = self.mnist[self.indicies[index]]
        img = img.squeeze()
        plt.imshow(img, cmap='gray')
        plt.show()
        plt.close()
        #print("This is index %d." % index)

class Labeled_MNIST(Dataset):
    def __init__(self, transform):
        super().__init__()

        self.transform = transform
        
        self.mnist = np.empty((60000,28,28,1))
        self.mnist = self.mnist.astype('uint8')
        self.label = torch.empty((60000))
        
        self.indicies = []
        self.len = len(self.indicies)
        self.index = 0
        
    def __getitem__(self, index):
        
        img = self.mnist[self.indicies[index]]
        img = self.transform(img)
        
        label = self.label[self.indicies[index]]
        
        return img, label
    
    def __len__(self):
        return self.len
        
    def labeling(self, img, label):
        self.indicies.append(self.index)
        self.mnist[self.index] = img
        self.label[self.index] = label
        self.len = len(self.indicies)
        self.index += 1
        #print("Labeled set length is %d." % self.len)
        
    def show(self, index):
        img = self.mnist[self.indicies[index]]
        img = img.squeeze()
        plt.imshow(img, cmap='gray')
        plt.show()
        plt.close()
        #print("This is index %d." % index)

class Unlabeled_MNIST(Dataset):
    def __init__(self, root, transform):
        super().__init__()

        self.transform = transform
        
        save_file = os.path.join(root, 'Augmented_MNIST.npy')
        
        if os.path.isfile(save_file):
            self.mnist = np.load(save_file)
            print("File is loaded.")
        
        else:
            train_loader = DataLoader(
                datasets.MNIST(
                        "./data/mnist",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor()])
                        ),
                batch_size=1, shuffle=False)
            
            self.mnist = np.empty((60000,28,28,1))
            
            for idx, (x, _) in enumerate(train_loader):
                x = x*255
                x = x.numpy().reshape(28,28,1)
                self.mnist[idx] = x
                print("[%d/60000]" % (idx+1))
                
            save_file = os.path.join(root, 'Augmented_MNIST')
            np.save(save_file, self.mnist)
        
        self.indicies = [i for i in range(60000)]
        self.len = len(self.indicies)
        self.mnist = self.mnist.astype('uint8')
        
    def __getitem__(self, rel_index):
        
        img = self.mnist[self.indicies[rel_index]]
        img = self.transform(img)
        
        return img, self.indicies[rel_index]
    
    def __len__(self):
        return self.len
    
    def remove(self, abs_index):
        self.indicies.remove(abs_index)
        self.len = len(self.indicies)
        #print("Index %d is removed. Unlabeled set length is %d." % (index, self.len))
    
    def show(self, abs_index):
        img = self.mnist[abs_index]
        img = img.squeeze()
        plt.imshow(img, cmap='gray')
        plt.savefig("./images/index_%07d.png" % abs_index)
        plt.show()
        plt.close()
        #print("This is index %d." % abs_index)
    
    def take(self, abs_index):
        img = self.mnist[abs_index]
        self.remove(abs_index)
        return img
    
    def copy(self, abs_index):
        img = self.mnist[abs_index]
        return img

if __name__ == "__main__":    
    transform = transforms.Compose([
         transforms.ToPILImage(),
         transforms.Resize(32),
         transforms.ToTensor()
     ])
    
    #a = Augmented_MNIST("/data1/ymh/asynch/", transform)
    a = Unlabeled_MNIST("C://유민형//개인 연구//Asynchronous Interactive Classification//", transform)
    b = Labeled_MNIST(transform)
    c = Pseudo_Labeled_MNIST(transform)
    
    a.show(0) # 5
    a.show(1) # 0
    
    img = a.take(0)
    label = 5
    b.labeling(img,label)
    a.show(0) # 0
    b.show(0) # 5
    
    img = a.copy(0)
    label = 0
    c.labeling(img,label,1)
    a.show(0) # 0
    c.show(0) # 0
    
    c.labeling(img,label,1)
    c.show(0) # 0
    
    '''
    a.show(0) # 5
    a.show(1) # 0

    b = Labeled_MNIST(transform)
    
    img = a.take(0)
    label = 5
    b.labeling(img,label)
    a.show(0) # 0
    b.show(0) # 5
    
    c1, c2 = b.__getitem__(0)
    
    '''
    '''
    d = DataLoader(a, batch_size=32, shuffle=True)
    
    x, _ = d.__iter__().next()
    #print(e1.shape, e2.shape)
    
    img = x[0].squeeze().numpy()  
    plt.imshow(img, cmap="gray")
    plt.show()
    plt.close()
    
    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    aug_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(32),
        #transforms.RandomAffine(0, shear=[-15, 15, -15, 15]),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
    
    x1 = torch.randn_like(x)
    x2 = torch.randn_like(x)
    for i in range(32):
        x1[i] = aug_transform(x[i])
        x2[i] = aug_transform(x[i])
    
    img = x1[0].squeeze().numpy()  
    plt.imshow(img, cmap="gray")
    plt.show()
    plt.close()
    '''
    
    '''
    d = DataLoader(a, batch_size=8, shuffle=True)
    
    e1, e2 = d.__iter__().next()
    print(e1.shape, e2.shape)
    
    img = a.take(0)
    label = 0
    b.labeling(img,label)
    a.show(0) # 4
    b.show(1) # 0
    
    d = DataLoader(a, batch_size=12, shuffle=True)
    
    e1, e2 = d.__iter__().next()
    print(e1.shape, e2.shape)
    '''