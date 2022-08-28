"""
Created on Tue Dec  8 12:25:53 2020
@author: ARJUN 
"""
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np
import os
from numpy import asarray
from numpy import save
from os import makedirs
from os import listdir
from shutil import copyfile
from random import random
import sys
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.optimizers import SGD


imagegen = ImageDataGenerator()
folder = 'Healthy and Unhealthy Train/'

#create directories
dataset_home = 'AET Sr Research Project/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
	# create label subdirectories
	labeldirs = ['healthy/', 'unhealthy/']
	for labldir in labeldirs:
		newdir = dataset_home + subdir + labldir
		makedirs(newdir, exist_ok=True)
# define proportion of pictures to use for validation
val_proportion = 0.25
# copy training dataset images into subdirectories
src_directory = 'Healthy and Unhealthy Train/'
for file in listdir(src_directory):
	src = src_directory + '/' + file
	dst_dir = 'train/'
	if random() < val_proportion:
		dst_dir = 'test/'
	if file.startswith('healthy'):
		dst = dataset_home + dst_dir + 'healthy/'  + file
		copyfile(src, dst)
	elif file.startswith('unhealthy'):
		dst = dataset_home + dst_dir + 'unhealthy/'  + file
		copyfile(src, dst)
        
  
# define convolutional neural network 
def define_the_model():
	# load model
	model = VGG16(include_top=False, input_shape=(224, 224, 3))
	# mark loaded layers as cannot be trained
	for layer in model.layers:
		layer.trainable = False
	# add new classifier layers
	flatLayer1 = Flatten()(model.layers[-1].output)
	classifierLayer1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flatLayer1)
	output = Dense(1, activation='sigmoid')(classifierLayer1) #Add sigmoid for classifier function (0 or 1 = Yes or no)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model
 
 
# run the model
def run_the_model():
	# define model
	model = define_the_model()
	# create data generator
	datagen = ImageDataGenerator(featurewise_center=True)
	# specify imagenet mean values for centering
	datagen.mean = [110.68, 106.843, 141.479]
	# prepare iterator
	train_it = datagen.flow_from_directory('AET Sr Research Project/train/',
		class_mode='binary', batch_size=64, target_size=(224, 224))
	test_it = datagen.flow_from_directory('AET Sr Research Project/test/',
		class_mode='binary', batch_size=64, target_size=(224, 224))
	# fit model
	history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=4, verbose=1)
	# evaluate model
	_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
	print('> %.3f' % (acc * 100.0))
 
run_the_model()