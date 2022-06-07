import torch
from model import Net
import data_handler as dh
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
torch.manual_seed(0)

net=Net()
criterion=nn.CrossEntropyLoss()
device= 'cuda' if torch.cuda.is_available() else "cpu"
print(device)
net.to(device)
trainloader, testloader=dh.data_loader()

def torch_fit(trainloader,criterion, lr, num_epochs, model):
    optimizer = optim.SGD(model.parameters(), lr)

    train_losses=[]
    test_losses=[]
    accuracy=[]
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch+1}/{num_epochs}')
        
        train_epoch_loss=[]
        for i,(images,labels) in enumerate(iter(trainloader)):
            images=images.to(device)
            labels=labels.to(device)
            #print(images.shape)
            #images.resize_(images.size()[0], images.size()[2],images.size()[3]) tried also (images.size()[0], 255*255)
            optimizer.zero_grad()
            output=model.forward(images)
            loss=criterion(output, labels)
            train_epoch_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        
        torch.save({'model_state' : model.state_dict(), 'input_size' : 128*128}, 'checkpoint_training')
        
        model.eval()
        with torch.no_grad():
            
            accuracy_on_epoch=[]
            test_epoch_loss=[]
            for j,(imagest,labelst) in enumerate(iter(testloader)):
                imagest=imagest.to(device)
                labelst=labelst.to(device)
                outputt=F.softmax(model(imagest), dim=1)
                losst=criterion(outputt,labelst)
                test_epoch_loss.append(losst.item())

                pred=outputt.argmax(dim=1)
                acc=(pred == labelst).sum()/len(labelst)
                accuracy_on_epoch.append(acc.item()) 

            torch.save({'model_state' : model.state_dict(), 'input_size' : 128*128}, 'checkpoint_testing')  

        mean_acc=sum(accuracy_on_epoch)/len(accuracy_on_epoch)
        accuracy.append(mean_acc)

        test_loss_mean=sum(test_epoch_loss)/len(test_epoch_loss)
        test_losses.append(test_loss_mean)


        train_loss_mean=sum(train_epoch_loss)/len(train_epoch_loss)
        train_losses.append(train_loss_mean)
        

        

        
        print(f'Mean epoch loss for train: {train_loss_mean}')
        print(f'Mean epoch loss for test: {test_loss_mean}')
        print(f'accuracy on epoch: {mean_acc}')
    
    torch.save({'model_state' : model.state_dict(), 'input_size' : 128*128}, 'checkpoint_finalModel') 

    x_axis_acc=list(range(num_epochs))
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train loss')
    plt.plot(test_losses, label='Test loss')
    
    plt.subplot(1,2,2)
    plt.plot(x_axis_acc,accuracy, label='accuracy')
    plt.legend()
    plt.show()

ans=torch_fit(trainloader=trainloader,criterion=criterion,lr=0.001,num_epochs=100,model=net)