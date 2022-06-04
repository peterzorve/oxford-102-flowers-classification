
from pickletools import optimize
from numpy import average
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 
from torchvision import transforms, datasets 
from model import Flowers_Classifier
import torch.optim as optim 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 


transforms_train = transforms.Compose([transforms.Grayscale(), transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize(mean=[0.5,], std=[0.5,])])
transforms_test  = transforms.Compose([transforms.Grayscale(), transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize(mean=[0.5,], std=[0.5,])])

root_path = "C:/Users/Omistaja/Desktop/epicode/deep_learning/flower_data/"

datasets_train = datasets.ImageFolder(root= root_path + 'train', transform=transforms_train)
datasets_test  = datasets.ImageFolder(root= root_path + 'test',  transform=transforms_test)

dataloader_train = DataLoader(dataset=datasets_train, batch_size=32, shuffle=True)
dataloader_test  = DataLoader(dataset=datasets_test,  batch_size=32, shuffle=True)

model = Flowers_Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

epochs, all_train_losses, all_test_losses, all_accuracies = 5, [], [], []

for epoch in range(epochs):
     train_losses, test_losses, test_accuracies = 0, 0, 0 
     for idx_train, (feature_train, target_train) in enumerate(iter(dataloader_train)):

          optimizer.zero_grad()

          prediction_train = model.forward(feature_train)
          loss_train = criterion(prediction_train, target_train)
          loss_train.backward()
          optimizer.step()

          train_losses += loss_train.item()

     average_train_losses = train_losses/len(dataloader_train)
     all_train_losses.append(average_train_losses)

     torch.save({"model_state": model.state_dict(), "input_size": 28*28}, 'saved_training_model')

     model.eval()
     with torch.no_grad():
          for idx_test, (feature_test, target_test) in enumerate(iter(dataloader_test)): 

               prediction_test = model.forward(feature_test)
               loss_test = criterion(prediction_test, target_test)

               test_losses += loss_test.item() 

               prediction_class = torch.argmax(prediction_test, dim=1)
               test_accuracies += accuracy_score(prediction_class, target_test)

          average_test_losses = test_losses/len(dataloader_test)
          all_test_losses.append(average_test_losses)

          average_test_accuracies = test_accuracies/len(dataloader_test)
          all_accuracies.append(average_test_accuracies)

     torch.save({"model_state": model.state_dict(), "input_size": 28*28}, 'saved_testing_model')

     model.train()

     print(f'{epoch+1:3}/{epochs} :  Train Loss : {average_train_losses:.6f}  |  Test Loss : {average_test_losses:.6f}  | Accuracy : {average_test_accuracies:.6f}')


torch.save({"model_state": model.state_dict(), "input_size": 28*28}, 'saved_final_model')

plt.plot(all_train_losses, label='Training Losses')
plt.plot(all_test_losses,  label='Testing Lossess')
plt.plot(all_accuracies,   label='Accuracy')
plt.show()

























