import logging
import sys
import os
sys.path.append('./log')
def log_file():
    dirName = './log/'
    if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Directory " , dirName ,  " Created ")
    else:  
         pass  
            #print("Directory " , dirName ,  " already exists")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    #FORMAT = '%(asctime)-15s:%(message)s'
    file_name='./log/heatmap.log'
    logging.basicConfig(filename =file_name,level = logging.INFO,format='%(levelname)s:%(asctime)s:%(message)s') 

