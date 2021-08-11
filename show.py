from torchvision import datasets
import matplotlib.pyplot as plt

a = datasets.MNIST("./data/mnist",
                        train=True,
                        download=True,
                        )
b, c = a.__getitem__(37038)

plt.imshow(b)
print(c)