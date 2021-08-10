#Reading Images
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from PIL import Image
#ML Imports
import keras
from keras import callbacks
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.preprocessing import LabelEncoder

np.random.seed(60)
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
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
plt.xticks(rotation=0)
plt.show()
'''



#Balance Data
unbalanced_dfs = [metadata[metadata['id'] == i] for i in range(7)]
N = 700
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
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25, random_state=60)

#Augment Data
aug = ImageDataGenerator(
    width_shift_range=2,
    height_shift_range=2,
    rotation_range=25,
    zoom_range=.15,
)

#Create model
model = Sequential([
    Conv2D(64, (3,3), activation="relu", input_shape=(PIXEL_SIZE, PIXEL_SIZE, 3)),
    MaxPool2D(pool_size=(2,2), padding="same"),
    Dropout(0.25),

    Conv2D(128, (3,3), activation="relu"),
    MaxPool2D(pool_size=(2,2), padding="same"),
    Dropout(0.25),

    Conv2D(256, (3,3), activation="relu"),
    MaxPool2D(pool_size=(2,2), padding="same"),
    Dropout(0.25),
    Flatten(),

    Dense(512, activation="relu"),
    Dense(7, activation="softmax")
])

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train model
batch_size = 32
epochs = 100

earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 6, 
                                        restore_best_weights = True)
hist = model.fit(
    x = aug.flow(x_train,y_train,batch_size=batch_size),
    epochs = epochs,
    validation_data = (x_test,y_test),
    callbacks=[earlystopping],
    verbose=2
)

score = model.evaluate(x_test, y_test)
print('Test accuracy', score[1])


#plot training and validation for accuracy and loss
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch #')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("graphs/accuracyTest.png")
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch #')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("graphs/lossTest.png")
plt.show()


#Saving the model for future reference
model_json = model.to_json()
with open("models/model.json", "w") as jf:
    jf.write(model_json)
model.save_weights("models/model.h5")
print("Model Saved")
