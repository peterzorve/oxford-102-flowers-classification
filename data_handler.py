import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 

def data_loader():
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(400),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])])

    test_transform = transforms.Compose([transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5],
                                                           [0.5, 0.5, 0.5])])

    train_set=datasets.Flowers102('flower_data/', download=True, train=True, transform=train_transform)
    trainset=DataLoader(train_set, batch_size=32, shuffle=True)

    test_set=datasets.Flowers102('flower_data', download=True, train=False, transform=test_transform)
    testset=DataLoader(test_set, batch_size=32, shuffle=False)

    return trainset, testset
    
trainset, testset= data_loader()
print(trainset.shape)
print(testset.shape)

    