## load library
import tensorflow as tf
import matplotlib.cm as cm
from PIL import Image
import cv2

import numpy as np
## create class for Heatmap generation using Grad-Cam Algorithm
class HeatmapGenerator:
    def __init__(self,image_preprocess_output,model,last_layer_name):
        self.image_preprocess_output = image_preprocess_output
        self.model=model
        self.last_layer_name=last_layer_name
    def softmax(self,x):
        f = np.exp(x)/np.sum(np.exp(x), axis = 1, keepdims = True)
        return f

    def make_heatmap_using_scorecam(self,model, img_array, layer_name, max_N=-1):

        cls = np.argmax(model.predict(img_array))
        act_map_array = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output).predict(img_array)
        
        # extract effective maps
        if max_N != -1:
            act_map_std_list = [np.std(act_map_array[0,:,:,k]) for k in range(act_map_array.shape[3])]
            unsorted_max_indices = np.argpartition(-np.array(act_map_std_list), max_N)[:max_N]
            max_N_indices = unsorted_max_indices[np.argsort(-np.array(act_map_std_list)[unsorted_max_indices])]
            act_map_array = act_map_array[:,:,:,max_N_indices]

        input_shape = model.layers[0].output_shape[0][1:]  # get input shape
        # 1. upsampled to original input size
        act_map_resized_list = [cv2.resize(act_map_array[0,:,:,k], input_shape[:2], interpolation=cv2.INTER_LINEAR) for k in range(act_map_array.shape[3])]
        # 2. normalize the raw activation value in each activation map into [0, 1]
        act_map_normalized_list = []
        for act_map_resized in act_map_resized_list:
            if np.max(act_map_resized) - np.min(act_map_resized) != 0:
                act_map_normalized = act_map_resized / (np.max(act_map_resized) - np.min(act_map_resized))
            else:
                act_map_normalized = act_map_resized
            act_map_normalized_list.append(act_map_normalized)
        # 3. project highlighted area in the activation map to original input space by multiplying the normalized activation map
        masked_input_list = []
        for act_map_normalized in act_map_normalized_list:
            masked_input = np.copy(img_array)
            for k in range(3):
                masked_input[0,:,:,k] *= act_map_normalized
            masked_input_list.append(masked_input)
        masked_input_array = np.concatenate(masked_input_list, axis=0)
        # 4. feed masked inputs into CNN model and softmax
        pred_from_masked_input_array = self.softmax(model.predict(masked_input_array))
        # 5. define weight as the score of target class
        weights = pred_from_masked_input_array[:,cls]
        # 6. get final class discriminative localization map as linear weighted combination of all activation maps
        cam = np.dot(act_map_array[0,:,:,:], weights)
        cam = np.maximum(0, cam)  # Passing through ReLU
        cam /= np.max(cam)  # scale 0 to 1.0
        
        return cam
    def make_gradcam_heatmap(self, pred_index=None):
    # First, we create a self.model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
        grad_modell = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(self.last_layer_name).output, self.model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_modell(self.image_preprocess_output)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    

## create a class for save save heatmap
class SaveGradCam:
    def __init__(self,image_path,heatmap,save_file_location,alpha_value):
        self.image_path = image_path
        self.heatmap = heatmap
        self.save_file_location = save_file_location
        self.alpha_value = alpha_value
    def save_gradcam(self,file_name):
    # Load the original image
       
        image_file = Image.open(self.image_path)
        image_file.save(self.image_path, quality=95)

        img = tf.keras.preprocessing.image.load_img(self.image_path)
        img = tf.keras.preprocessing.image.img_to_array(img)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(224 * self.heatmap)
        
        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * self.alpha_value + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        


        # Save the superimposed image
        superimposed_img.save(self.save_file_location+"/"+file_name)

