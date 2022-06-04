
from ast import arg
from tkinter import font
from PIL import Image
from torchvision import transforms 
import torch  
from model import MNIST_Classifier
import matplotlib.pyplot as plt 


train_model = torch.load('new_trained_model')
model_state = train_model['model_state']

model = MNIST_Classifier()
model.load_state_dict(model_state)

def make_predict(image_path, model):
     preprocessor = transforms.Compose([ transforms.Grayscale()   , transforms.Resize((28, 28)),  transforms.ToTensor(),  transforms.Normalize([0.5], [0.5])  ])
     classes_dict = {0:'Zero',  1:'One',   2:'Two',  3:'Three',  4:'Four',  5:'Five',   6:'Six',  7:'Seven',   8:'Eight',   9:'Nine' }

     image = Image.open(image_path)
     processed_image = preprocessor(image)
     processed_image = processed_image.view(1, *processed_image.shape)

     model.eval()
     with torch.no_grad():
          prediction = model.forward(processed_image)
          _, prediction_class = torch.max(prediction, dim=1)

     plt.imshow(image, cmap='gray')
     plt.title(f'Prediction digit:  {prediction_class.item()}    ({classes_dict[prediction_class.item()]})', fontsize=15)
     plt.show()

     return

#######################################################################################################################################
########################################                  CLI                 #########################################################
#######################################################################################################################################

# import argparse 
# parser = argparse.ArgumentParser(description='Image path')
# parser.add_argument('first', type=str, help='Input the path of the image')
# args = parser.parse_args()
# link1 = args.first 
# image_path_arg = 'C:/Users/Omistaja/Desktop/New folder/img_' +  link1 + "_1.jpg"


#######################################################################################################################################
#######################################################################################################################################



for i in range(9, -1, -1):
     image_path_arg = 'C:/Users/Omistaja/Desktop/New folder/img_' +  str(i) + "_1.jpg"
     image_path = image_path_arg
     make_predict(image_path, model)
