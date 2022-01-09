from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import img_to_array, load_img
import os
from keras import models
from keras import layers
from keras import optimizers
from keras.applications.vgg16 import VGG16
# from keras.applications import Xception
import datetime
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import utils
from keras.utils import np_utils
import cv2
import itertools
import matplotlib.pyplot as plt
from keras.models import load_model

#path of dataset
path = "/D:/MUZIK/AKRAM/DUVET/Sign-Language-Detection-master/dataset_final_3"


noOfClasses = 27

# dictionary for classes from char to numbers
classes = {
	'A': 0,
	'B': 1,
	'C': 2,
	'D': 3,
	'E': 4,
	'F': 5,
	'G': 6,
	'H': 7,
	'I': 8,
	'J': 9,
	'K': 10,
	'L': 11,
	'M': 12,
	'N': 13,
	'O': 14,
	'P': 15,
	'Q': 16,
	'R': 17,
	'S': 18,
	'T': 19,
	'U': 20,
	'V': 21,
	'W': 22,
	'X': 23,
	'Y': 24,
	'Z': 25,
	'n': 26, #n for none class
}

X = []
y = [] 
input_size = 224


vgg_conv = VGG16(weights = 'imagenet', include_top = False, input_shape = (input_size, input_size, 3))

for layer in vgg_conv.layers[:-4]:
	layer.trainable = False
	
for layer in vgg_conv.layers:
	print(layer, layer.trainable)

number=100

def load_data():
	flag=0
	for root, directories, filenames in os.walk('dataset_final_3'):
		for filename in filenames:
			if filename.endswith(".jpg"):
				if filename[:5] == 'space':
					if int(filename[7:-4])%2==0: #taking alternate pictures. Remove this if more data is required.
						fullpath = os.path.join(root, filename)
						img = load_img(fullpath)
						img = img_to_array(img)
						img = cv2.resize(img,(224,224))
						X.append(img)
						if not flag:
							print ("fp: ",fullpath)

						t = fullpath.rindex('\\')
						if not flag:
							print ("t: ",t)
						fullpath = fullpath[0:t]
						if not flag:
							print ("fp: ",fullpath)
						n = fullpath.rindex('\\')
						if not flag:
							print ("n: ",n)
							print ("n1 t",fullpath[n+1:t])
							flag=1
						y.append(classes[fullpath[n + 1:t]])
				elif filename[:4] == 'none':
					if int(filename[5:-4])%2==0:
						fullpath = os.path.join(root, filename)
						img = load_img(fullpath)
						img = img_to_array(img)
						img = cv2.resize(img,(224,224))
						X.append(img)
						t = fullpath.rindex('\\')
						fullpath = fullpath[0:t]
						n = fullpath.rindex('\\')
						y.append(classes[fullpath[n + 1:t]])
				elif int(filename[2:-4])%2	==0:
					fullpath = os.path.join(root, filename)
					img = load_img(fullpath)
					img = img_to_array(img)
					img = cv2.resize(img,(224,224))
					X.append(img)
					t = fullpath.rindex('\\')
					fullpath = fullpath[0:t]
					n = fullpath.rindex('\\')
					y.append(classes[fullpath[n + 1:t]])
	print("Data loaded!!!")
	return X, y


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')


def createModel():
	model = models.Sequential()
	for layer in vgg_conv.layers:
		model.add(layer)
	model.add(layers.Flatten())
	model.add(layers.Dense(1024, activation = 'relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(noOfClasses, activation = 'softmax'))
	
	return model

def train():
	print("Loading data.....")
	X, y = load_data()
	y = np.asarray(y)
	y = y.reshape(y.shape[0], 1)
	X = np.asarray(X).astype('float32')
	X = X / 255.0
	y = np_utils.to_categorical(y, noOfClasses)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
	print(y_test)
		
	model = createModel()

	print("Model summary")
	model.summary()

	batch_size = 10
	epochs = 10
	model.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	print("Training........")
	model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (X_test, y_test))
	print("Training complete!!!!")
	model.save('./keras.FINAL_MODEL5')
	
                  
train()