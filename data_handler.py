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

    test_transform = transforms.Compose([transforms.CenterCrop(400),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5],
                                                           [0.5, 0.5, 0.5])])

    data_dir= 'C:/Users/Abubakr/Documents/Datasets/oxford-102-flower-pytorch/flower_data'

    train_set=datasets.ImageFolder(data_dir+'/train',transform=train_transform)
    trainset=DataLoader(train_set, batch_size=102, shuffle=True)

    test_set=datasets.ImageFolder(data_dir+'/valid',transform=test_transform)
    testset=DataLoader(test_set, batch_size=102, shuffle=False)

    return trainset, testset
    
trainset, testset= data_loader()


    