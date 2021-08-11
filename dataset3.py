import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
import os
import matplotlib.pyplot as plt


class Augmented_MNIST(Dataset):
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
        
        self.len = 60000
        self.mnist = self.mnist.astype('uint8')
        
    def __getitem__(self, index):
        
        img = self.mnist[index]
        x1 = self.transform(img)
        x2 = self.transform(img)
        
        return x1, x2
    
    def __len__(self):
        return self.len
    

class Index_MNIST(Dataset):
    def __init__(self, root, aug_transform, normal_transform, aug_default=True):
        super().__init__()
        
        self.aug_transform = aug_transform
        self.normal_transform = normal_transform
        
        self.aug_default = aug_default
        
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
        
        self.mnist = self.mnist.astype('uint8')
        self.label = torch.empty((60000))
        
        self.len = 60000
        
    def __getitem__(self, index):
        
        img = self.mnist[index]
        
        if self.aug_default:
            img = self.aug_transform(img)
            label = self.label[index]
            return img, label
            
        else:
            img = self.normal_transform(img)
            return img, index
    
    def __len__(self):
        return self.len
        
    def labeling(self, index, label):
        self.label[index] = label
            
    def show(self, index):
        img = self.mnist[index]
        img = img.squeeze()
        plt.imshow(img, cmap='gray')
        plt.savefig("./images/index_%07d.png" % index)
        plt.show()
        plt.close()
        #print("This is index %d." % index)



if __name__ == "__main__":
    from torch.utils.data.sampler import SubsetRandomSampler
    
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
    
    #a = Index_MNIST("/data1/ymh/asynch/", transform)
    a = Index_MNIST("C://유민형//개인 연구//Asynchronous Interactive Classification//", aug_transform, normal_transform)
    #a.show(1)
    
    train_idx = [1,20,33,14]
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = DataLoader(dataset=a, batch_size=2, sampler=train_sampler)
    
    b, c = train_loader.__iter__().next()
    print(b.shape, c)
    
    a.aug_default = False
    train_idx = [1,20,33,14]
    train_sampler = SubsetRandomSampler(train_idx)
    train_loader = DataLoader(dataset=a, batch_size=2, sampler=train_sampler)
    
    b, c = train_loader.__iter__().next()
    print(b.shape, c)
    
    