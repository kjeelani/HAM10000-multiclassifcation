#Reading Images
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from PIL import Image
#ML Imports
import keras
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder

np.random.seed(60)
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample

'''
Credits to DigitalSreeni for basic setup of model:
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

metadata = pd.read_csv("metadata/HAM10000_metadata.csv")
PIXEL_SIZE = 64

le = LabelEncoder()
le.fit(metadata['dx'])
LabelEncoder()
print(list(le.classes_))
metadata['id'] = le.transform(metadata['dx'])


# Plot counts of different classes
'''
data_graph = plt.figure(figsize=(8,5))
axis = data_graph.add_subplot(111)
metadata['dx'].value_counts().plot(kind='bar', ax=axis)
axis.set_ylabel('Count')
axis.set_xlabel('Labels')
axis.set_title('Cancer Type')
plt.show()
'''

#Balance Data
unbalanced_dfs = [metadata[metadata['id'] == i] for i in range(7)]
N = 775
balanced_df = pd.concat([resample(df, replace=True, n_samples=N, random_state=50) for df in unbalanced_dfs])

#Read Images
image_paths = {os.path.splitext(os.path.basename(x))[0]: "./data/" + x for x in os.listdir('./data')}
balanced_df['path'] = metadata['image_id'].map(image_paths.get)
balanced_df['image'] = balanced_df['path'].map(lambda x : np.asarray(Image.open(x).resize((PIXEL_SIZE, PIXEL_SIZE))))

#Create test/train split
X = np.asarray(balanced_df['image'].tolist())
X = X/255
Y = balanced_df['id']
Y = to_categorical(Y, num_classes=7)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=60)

#Create model
model = Sequential([
    Conv2D(256, (3,3), activation="relu", input_shape=(PIXEL_SIZE, PIXEL_SIZE, 3)),
    MaxPool2D(pool_size=(2,2)),
    Dropout(0.3),

    Conv2D(128, (3,3), activation="relu"),
    MaxPool2D(pool_size=(2,2)),
    Dropout(0.3),

    Conv2D(64, (3,3), activation="relu"),
    MaxPool2D(pool_size=(2,2)),
    Dropout(0.3),
    Flatten(),

    Dense(32),
    Dense(16),
    Dense(7, activation="softmax")
])

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

#Train model
batch_size = 16
epochs = 50

hist = model.fit(
    x_train, y_train,
    epochs = epochs,
    batch_size = batch_size,
    validation_data = (x_test,y_test),
    verbose=2
)

score = model.evaluate(x_test, y_test)
print('Test accuracy', score[1])


#plot training epochs
'''
loss = hist.history['acc']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
'''


#Saving the model for future reference
model_json = model.to_json()
with open("models/model.json", "w") as jf:
    jf.write(model_json)
model.save_weights("models/model.h5")
print("Model Saved")


#Usage of model
'''
jf = open('models/model.json', 'r')
model = jf.read()
jf.close()
model = model_from_json(model)
model.load_weights("model.h5")
'''