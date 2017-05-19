import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout

img_width = 28
img_height = 28

train_data_directory = 'data\\train'
test_data_directory  = 'data\\test'

train_samples = 600
test_samples = 100

epochs = 30

# Set Softmax Layer Neuron count for output layer
num_output_neurons = 5

# Set batch size so the GPU doesnt raise a ResourceExhaustedError
# GPU: GTX 960M 4GB GDDR5; tensorflow-gpu 1.1.0 <pip>; cuDNN 5.1 (cudnn64_5.dll) 
# NOTE: tensorflow-cpu from conda-forge works without issue however.
batch_size = 32

# Set kernel size for convolutional layers. Attempt to change it to 32 and 64 for conv1 and conv2
# resulted in ResourceExhausted errors. Fine tuning required!

# Set kernel size for Conv1 
conv_1_kernel = 16

# Set kernel size for Conv2
conv_2_kernel = 32

# =====================================  Model  =================================

# image => conv2d => conv2d => maxpool => conv2d =>conv2d =>maxpool =>flatten (for FC)=> FC => FC => softmax

# Note: No dropout since this is a really small CNN
model = Sequential()

# Conv1 kernel size 32
model.add(Conv2D(conv_1_kernel,(5,5),activation='relu',input_shape=(img_width,img_height,3)))
# Conv1 kernel size 32
# model.add(Conv2D(conv_1_kernel,(5,5),activation='relu',input_shape=(img_width,img_height,3)))
# Maxpool 2x2
model.add(MaxPooling2D(2,2))
# Conv2 kernel size 64
model.add(Conv2D(conv_2_kernel,(5,5),activation='relu',input_shape=(img_width,img_height,3)))
# Conv2 kernel s ize 64
#model.add(Conv2D(conv_2_kernel,(2,2),activation='relu',input_shape=(img_width,img_height,3)))
# Maxpool 2x2
model.add(MaxPooling2D(2,2))

# Flatten for Fully Connected Layers
model.add(Flatten())

# Fully Connected Layers for backprop weight learning
model.add(Dense(1000,activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(num_output_neurons,activation='softmax'))

# Softmax yields class probability distribution

# =======================================================

# Configure learning process
model.compile(loss='binary_crossentropy',
	optimizer = 'rmsprop', # rmsprop with momentum
	metrics = ['accuracy'])

# Generate Training and Testing Data
train_data_gen = ImageDataGenerator(
		rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_data_gen = ImageDataGenerator(
	horizontal_flip = True)

train_generator = train_data_gen.flow_from_directory(
	train_data_directory,
	target_size = (img_width,img_height),
	batch_size = batch_size,
	class_mode = 'categorical'	)

test_generator = test_data_gen.flow_from_directory(
	test_data_directory,
	target_size = (img_width,img_height),
	batch_size = batch_size,
	class_mode = 'categorical')

# Run through model
model.fit_generator(
	train_generator,
	validation_steps = train_samples, #No. of samples to train on in each epoch
	steps_per_epoch = batch_size,
	epochs = epochs, # Set above
	validation_data = test_generator) 

# Save weights
model.save_weights('convo_neural_net.h5')