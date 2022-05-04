import os
## avoid tensorflow warning and errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
import  logging,time
import configparser
import urllib.request
from main_code.core import LoadModel,PreprocessImage
from main_code.heatmap_generation import HeatmapGenerator,SaveGradCam
from main_code.image_preprocessing import ImagePreProcessing
from flask import Flask
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from flask import Flask, flash, request, redirect, url_for, render_template,send_file
from werkzeug.utils import secure_filename
from PIL import Image

from markupsafe import escape
from log import log_file
## run log files 
log_file()
config = configparser.ConfigParser()
config.read('config.ini')

UPLOAD_FOLDER =config['foldername']['upload_image']

app = Flask(__name__)
app.secret_key =config['secret_key']['secret_key_value']
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] =16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

## load model
start_time = time.time()
reconstructed_model = keras.models.load_model("model",compile=False)
load_model=LoadModel(model_name=reconstructed_model)
create_model=load_model.create()


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

	
@app.route('/')
def upload_form():
	return render_template('upload_2.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		logging.info('Uploadupload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed below')
		
		return render_template('upload_2.html', filename=filename)
	else:
		logging.error('file not successfully uploaded')
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	start_time = time.time()
	file_location=config['foldername']['upload_image']+filename
	preprocess_file_location=config['foldername']['preprocess_output']+filename.split(".")[0]+".png"
	## preprocessing
	start_time = time.time()
	image_file=ImagePreProcessing(config['foldername']['upload_image'],config['foldername']['preprocess_output'])
	save_preprocess_image=image_file.run_file(filename)
	logging.info("load model successfully created")
	logging.info('preprocessing done  ' + filename)
	logging.info('process start ' + filename)
	extract_file_name=file_location.split("/")[-1]
	save_filename=config['foldername']['save_prediction_folder']+extract_file_name
	start_time = time.time()
	preprocessing_image=PreprocessImage(image_path=preprocess_file_location,target_size=(299,299),preprocee_input=tf.keras.applications.inception_resnet_v2.preprocess_input)
	preprocess_image_output=preprocessing_image.preprocess()
	

	logging.info("image preprocessing done")
	start_time = time.time()
	heat_map_create=HeatmapGenerator(image_preprocess_output=preprocess_image_output,model=create_model,last_layer_name=config['model']['last_layer_name'])
	make_heat_map_grad_cam=heat_map_create.make_gradcam_heatmap()
	make_heat_map=heat_map_create.make_heatmap_using_scorecam(create_model,preprocess_image_output,config['model']['last_layer_name'],max_N=8)
	## merge two heatmap
	merge_heatmap=(make_heat_map+make_heat_map_grad_cam)
	logging.info("heatmap array created")
	save_heatmap_output=SaveGradCam(image_path=file_location,heatmap=merge_heatmap,save_file_location=config['foldername']['save_file_location'],alpha_value=int(config['model']['alpha_value']))
	save_heatmap_data=save_heatmap_output.save_gradcam(file_name=extract_file_name)
	logging.info("final output saved"+config['foldername']['save_prediction_folder']+extract_file_name)
	
	##fixed the size of each output image so that the watermark add proper size
	basewidth = 1600
	img = Image.open("static/Heatmap_result/"+filename)
	wpercent = (basewidth/float(img.size[0]))
	hsize = int((float(img.size[1])*float(wpercent)))
	img = img.resize((basewidth,hsize), Image.ANTIALIAS)
	img.save("static/Heatmap_result/"+filename)
	## save watermark on image
	image = Image.open("static/Heatmap_result/"+filename)
	logo = Image.open('static/logo.png')
	image_copy = image.copy()
	position = ((image_copy.width - logo.width), (image_copy.height - logo.height))
	image_copy.paste(logo, position,logo)
	image_copy.save("static/Heatmap_result/"+filename)


	return redirect(url_for(config['foldername']['save_data_folder_name'],filename=config['foldername']['save_prediction_folder']+filename,),code=config['admin']['code'])

@app.route('/file_download/<filename>')
def download_file(filename):
	filename_data=config['foldername']['save_file_location']+filename
	return send_file(filename_data, as_attachment=True)



