## import library

import numpy as np
import tensorflow as tf
##import  inception_restnet application from keras
#from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
##import restnet
#from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
## import Vgg16 or 19
#from tensorflow.keras.applications.vgg19 import preprocess_input
#import inception_v3
#from tensorflow.keras.applications.inception_v3 import preprocess_input



from tensorflow.keras.preprocessing import image

## create class for model
class LoadModel:
    def __init__(self,model_name):
        self.model_name = model_name
    def create(self,):
        #model_builder = self.model_name
        #model=model_builder(weights="imagenet")
        ## Remove last layer's softmax'
        self.model_name.layers[-1].activation=None
        
        return self.model_name
    
## create  class for preprocess image
class PreprocessImage:
    def __init__(self,image_path,target_size,preprocee_input):
        self.image_path = image_path
        self.target_size=target_size
        self.preprocess_input=preprocee_input
    def preprocess(self,):
        load_image=image.load_img(self.image_path,target_size=self.target_size)
        convert_array=image.img_to_array(load_image)
        expand_dims=np.expand_dims(convert_array,axis=0)
        preprocess_output=self.preprocess_input(expand_dims)
        
        return preprocess_output
    