import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
import sys

img_width = 28
img_height = 28

# Set Softmax Layer Neuron count for output layer
num_output_neurons = 5

# Set kernel size for Conv1 
conv_1_kernel = 16
# Set kernel size for Conv2
conv_2_kernel = 32


def create_model(summary_spec):
	model = Sequential()

	# Conv1 and Maxpool
	model.add(Conv2D(conv_1_kernel,(5,5),activation='relu',input_shape=(img_width,img_height,3)))
	#model.add(Conv2D(conv_1_kernel,(5,5),activation='relu',input_shape=(img_width,img_height,3)))
	model.add(MaxPooling2D(2,2))
	
	# Conv2 and Maxpool
	model.add(Conv2D(conv_2_kernel,(5,5),activation='relu',input_shape=(img_width,img_height,3)))
	#model.add(Conv2D(conv_2_kernel,(5,5),activation='relu',input_shape=(img_width,img_height,3)))
	model.add(MaxPooling2D(2,2))

	# Flatten for FCL
	model.add(Flatten())
	
	# FCL for backprop and softmax for distrib
	model.add(Dense(1000,activation='relu'))

	# Dropout
	#model.add(Dropout(0.5))
	
	# Final Softmax FC layer
	model.add(Dense(num_output_neurons,activation='softmax'))
	
	if summary_spec:
		model.summary()

	return model


img = cv2.imread(sys.argv[1])
img = cv2.resize(img,(img_width,img_height))

model = create_model(summary_spec=False)
model.load_weights('convo_neural_net.h5')

#img_asArray = np.array(img).reshape(img_width,img_height,3) -> useless yeah?
img_asArray = np.expand_dims(np.array(img),axis=0)

prediction = model.predict(img_asArray)[0]

pred_class = ""
maxProbability = 0

labels = ["daisy","dandelion","rose","sunflower","tulip"]

print("="*20)
print(" ")
print(labels)
print(list(prediction))

for n in range(0,5):
	if prediction[n] > maxProbability:
		pred_class = labels[n]
		maxProbability = prediction[n]

print(" ")
print(pred_class+str(" % = ")+str(maxProbability*100))