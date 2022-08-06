import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import cv2
import fiftyone as fo
import fiftyone.zoo as foz
print(tf.version.VERSION)


model_anotasi = foz.load_zoo_model("faster-rcnn-resnet50-fpn-coco-torch")

sdir = 'anotasi/'
file_name = ''+sdir
resp = {}
#buat folder
if os.path.exists(file_name) == False:
    os.mkdir(''+sdir)

dir = "newtest/"
list_class = os.listdir(dir) 
total_class = len(list_class)
for i in range(total_class):

    folder = ""+sdir+list_class[i]
    if os.path.exists(folder) == False:
        os.mkdir(folder)
    list_data = os.listdir(dir+list_class[i]) 
    total_data = len(list_data)

    lokasi_save = folder+'/'+'1.jpg'
    lokasi_file = dir+list_class[i]+'/'+list_data[0]

    anotasi = cv2.imread(lokasi_file)
    height, width, channels = anotasi.shape
    lokasi_dataset = "newtest/"+list_class[i]+"/"
    dataset = fo.Dataset.from_images_dir(lokasi_dataset)
    samples = dataset.take(1)
    samples.apply_model(model_anotasi, label_field="faster_rcnn", confidence_thresh=0.7,classes=["bird"])

    sample = dataset.first()
    bird_count = 0
    bird_count = len(sample.faster_rcnn.detections)
    response = []
    bouding_box = []
    no_bird = False
    for j in range(len(sample.faster_rcnn.detections)):
        # check anotasi kosong
        if len(sample.faster_rcnn.detections) != 0 :

            if j<=4:
                detection = sample.faster_rcnn.detections[j]
                resp = {}
                box = {}
                # check anotasi burung
                if (detection.label=='bird'):
                    # konvert anotasi ke kordinat
                    no_bird = True
                    x, y, w, h = detection.bounding_box
                    bbox = [x * width, y * height, w * width, h * height]
                    X = int(bbox[0])
                    Y = int(bbox[1])
                    W = int(bbox[2])
                    H = int(bbox[3])
                    X = X-20
                    box["x"] = X
                    Y = Y-20
                    box["y"] = Y
                    W = W+20
                    box["w"] = W
                    H = H+20
                    box["h"] = H
                    # crop gambar dengan anotasi
                    cropped_image = anotasi[Y:Y+H, X:X+W]
                    cropped_image = cv2.resize(cropped_image, (224, 224))
                    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    row,col= cropped_image.shape
                    mean = 0.1
                    var = 0.1            
                    gauss = np.random.normal(mean,var,(row,col))
                    gauss = gauss.reshape(row,col)
                    cropped_image = cropped_image + gauss
                    cv2.imwrite("anotasi/"+list_class[i]+"/"+str(j)+'.jpg', cropped_image)
        
        
        
        
        