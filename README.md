# Overview
This repository contains a CNN model used to classify seven different types of skin cancers based on the HAM-10000 dataset. The model was made in Keras and saved in the models folder for anyone's use.

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
The data was augmented with random rotations, shifts, and zooms to allow for better generalizations. With around 30 epochs, the model performed relatively well with 85.4% train accuracy and 0.38 loss. With some more optimizations or swithcing to a ResNet34 given more processing power, we predict the model can reach up to 89% accuracy and 0.3 loss.
