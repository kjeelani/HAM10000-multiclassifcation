import numpy as np
import pandas as pd
import os
import shutil
from PIL import Image

'''
Credits to DigitalSreeni for loading HAM10000 dataset:
https://www.youtube.com/channel/UC34rW-HtPJulxr5wp2Xa04w
'''

'''
7 Classes of Skin Cancer Lesions

akiec - Actinic Keratoses (Benign)
bcc - Basal Cell Carcinoma (Benign)
bkl - Benign Keratoses (Benign)
df - Dermatofibroma (Benign)
nv - Melanocytic nevi (Benign)
mel - Melanoma (Malignant)
vasc - Vascular Skin Lesions (Benign or Malignant)
'''

#Set data and destination directories
data_dir = os.getcwd() + "/data/"
dest_dir = os.getcwd() + "/data_sorted/"

skin_df = pd.read_csv("metadata/HAM10000_metadata.csv")
print(skin_df['dx'].value_counts())

label = skin_df['dx'].unique().tolist()
label_images = []
print(label)

for i in label:
    os.mkdir(dest_dir + str(i) + "/")
    sample = skin_df[skin_df['dx'] == i]['image_id']
    label_images.extend(sample)
    for id in label_images:
        shutil.copyfile((data_dir + "/" + id + ".jpg"), (dest_dir + i + "/" + id + ".jpg"))
    label_images=[]

