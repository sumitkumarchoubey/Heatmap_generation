import os
import cv2
import time
import PIL
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from multiprocessing import Pool
from PIL import Image, ImageEnhance
import configparser
import time

## create class Image Preprocessing

class ImagePreProcessing:
    def __init__(self,file_save_folder,file_save_location):
        self.folder_name=file_save_folder
        self.file_save=file_save_location
        
    def create_folder(self,):
         if not os.path.exists(self.file_save):
                 os.mkdir(self.file_save)
         else:  
            pass  
    
    def resize(self,file_name):
        
        pil_image=Image.open(self.file_save+file_name)
        length_x,width_y=pil_image.size
        factor = min(1, float(1024.0 / length_x))
        size = int(factor * length_x), int(factor * width_y)
        im2 = pil_image.resize((int(length_x*1), int(width_y*1)), Image.BOX)
        file_name=file_name.split(".")[0]+".png"
        
        im2.save(self.file_save+"/"+file_name,dpi=(400,400))
        return file_name
        
    def enhance_image(self,file_name):

        image_file=Image.open(self.file_save+file_name)
        enhancer = ImageEnhance.Contrast(image_file)
        factor = 1 #increase contrast
        im_output = enhancer.enhance(factor)
        file_name=file_name.split(".")[0]+".png"
        im_output.save(self.file_save+file_name)
        return file_name
    def binarization_thresholding(self,file_name):
        img=cv2.imread(self.file_save+file_name,0)
        blur = cv2.GaussianBlur(img,(5,5),0)
        #bilFilter = cv2.bilateralFilter(img,9,75,75)

        try:
            ret,thresh4 = cv2.threshold(blur,160,255,cv2.THRESH_TOZERO)
         
            time.sleep(1)
            file_name=file_name.split(".")[0]+".png"
            cv2.imwrite(self.file_save+file_name,thresh4)
            return file_name
        except ValueError:
            pass
            #print("Cannot  binarization_thresholding",file_name)


        
    def run_file(self,file_name):
       
        file_name_update =file_name.split(".")[0]+".png"
        im = Image.open(self.folder_name+file_name)
        im.save(self.file_save+file_name_update)

        try:
            resize_image_name=self.resize(file_name_update)
            enhance_image=self.enhance_image(resize_image_name)
            binarization_thresholding_image=self.binarization_thresholding(enhance_image)
            

            return file_name
        except OSError:
            #print("Can't resize image",file_name)
           pass
