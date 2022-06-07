import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader 
torch.manual_seed(0)
def data_loader():
    train_transform = transforms.Compose([transforms.Resize(350),
                                       transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(300),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])])

    test_transform = transforms.Compose([transforms.Resize(350),
                                      transforms.CenterCrop(300),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5],
                                                           [0.5, 0.5, 0.5])])

    data_dir= 'C:/Users/Abubakr/Documents/Datasets/oxford-102-flower-pytorch/flower_data'

    train_set=datasets.ImageFolder(data_dir+'/train',transform=train_transform)
    trainset=DataLoader(train_set, batch_size=32, shuffle=True)

    test_set=datasets.ImageFolder(data_dir+'/valid',transform=test_transform)
    testset=DataLoader(test_set, batch_size=32, shuffle=False)

    return trainset, testset


import matplotlib.pyplot as plt
import numpy as np
import data_handler as dh

def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.cpu().numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

import numpy as np
def view_classify_general(img, ps, class_list):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.cpu().data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    imshow(img, ax=ax1, normalize=True)
    ax1.axis('off')
    ax2.barh(np.arange(len(class_list)), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(len(class_list)))
    ax2.set_yticklabels([x for x in class_list], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()