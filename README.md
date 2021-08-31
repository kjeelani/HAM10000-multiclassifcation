# Overview
This repository contains a CNN model used to classify seven different types of skin cancers based on the HAM-10000 dataset. The model was made in Keras and saved in the models folder for anyone's use.


# Downloading Dataset
The current repository contains an **empty** folder named _data_. The data folder is empty due to Github's recommendation to keep repositories less than 1GB. Therefore one will have to download the full HAM10000 dataset seperately. The process for doing so is as follows:

1.) Go to the dataset's [webpage](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).

2.) Click "Access Dataset" and download as an original zip file

3.) Extract the zip file for the images. The images may be in two seperate folders. Transfer all files from both folders to the _data_ folder


# Loading the Model
```python3
from keras.models import model_from_json

jf = open('models/model.json', 'r')
model = jf.read()
jf.close()
model = model_from_json(model)
model.load_weights("model.h5")
```

# Model Overview
The model consists of three convolutional layers and two dense layers (one being the output layer). The loss function is categorical-crossentropy and the optimizer is Adam. 
The data was augmented with random rotations, shifts, and zooms to allow for better generalizations. With around 30 epochs, the model performed relatively well with 87.8% train accuracy and 0.34 loss. With some more optimizations or swithcing to a ResNet34 given more processing power, we predict the model can reach up to 89% accuracy and 0.25 loss.
