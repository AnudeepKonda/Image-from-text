import tensorflow as tf
import numpy as np, os, sys, re
import random, json, string, pickle
import keras
from keras.models import Sequential
from keras.models import Model
import keras.layers
from keras.layers import Merge, Dense, merge
import keras.models
from keras.optimizers import SGD
import keras.callbacks
<<<<<<< HEAD
=======
import re
import os
>>>>>>> origin/master
from keras.preprocessing import image

img_names = []
encoding = []
<<<<<<< HEAD
file_path = os.path.dirname(os.path.abspath(__file__)) + "/pubfig_imgs"
=======
file_path = os.path.dirname(os.path.abspath(__file__)) + "\\pubfig_imgs"
>>>>>>> origin/master
with open("pubfig_attributes.txt") as img_file:
    count = 0
    print("Starting to fetch attributes")
    count = 0
    for line in img_file:
        arr = re.split(' |\t', line)
        if arr[0] == '#':
            continue
        img_name = arr[0]
        num = 0
        for i in range(1,20):
            try:
                test = int(arr[i])
                img_name += '-'+arr[i]
                num = i
                break
            except:
                img_name += '-'+arr[i]
        img_name += ".jpg"
        img_attr = arr[num:]
        for i in range(len(img_attr)):
            item = float(img_attr[i])
            if item <= 0:
                img_attr[i] = 0
            else:
                img_attr[i] = 1
        img_names.append(img_name)
        encoding.append(img_attr)
        count += 1
        if count % 10000 == 0:
            print("10000 more done")

encoding = np.asarray(encoding)
img_names = np.asarray(img_names)

<<<<<<< HEAD
concatenated_vector = keras.layers.Input(shape=(4, 4, 128))

gen1_input = keras.layers.Input(shape=(4, 4, 1152))
di1_input = keras.layers.Input(shape=(128, 128, 3))
=======
gen1_input = keras.layers.Input(shape=(4, 4, 1024))
>>>>>>> origin/master
real_image = keras.layers.Input(shape=(128, 128, 3))

di2_input = keras.layers.Input(shape=(256, 256, 6))
<<<<<<< HEAD
=======
di1_input = keras.layers.Input(shape=(128, 128, 3))
>>>>>>> origin/master
gen2_input = keras.layers.Input(shape=(128, 128, 3))
main_model = Sequential()
full_model = Sequential()

s1_gen = Sequential()
s1_dis = Sequential()
s1_di = Sequential()
s1_c = Sequential()


s2_gen = Sequential()
s2_dis = Sequential()

train_model_s1 = Sequential()
train_model_s2 = Sequential()

train_model_full = Sequential()

text_input = Sequential()
text_input.add(keras.layers.convolutional.Conv2D(128, 1, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=True, bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=(4, 4, 128)))

########################################  S 1 Generator ###############################################

s1_gen = keras.layers.convolutional.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=(4, 4, 1152))(gen1_input)
s1_gen = keras.layers.convolutional.Conv2D(512, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_gen)

s1_gen = keras.layers.convolutional.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_gen)
s1_gen = keras.layers.convolutional.Conv2D(256, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_gen)

s1_gen = keras.layers.convolutional.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_gen)
s1_gen = keras.layers.convolutional.Conv2D(128, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_gen)

s1_gen = keras.layers.convolutional.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_gen)
s1_gen = keras.layers.convolutional.Conv2D(64, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_gen)

s1_gen = keras.layers.convolutional.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_gen)
gen1_output = keras.layers.convolutional.Conv2D(3, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_gen)

s1_generator = Model(inputs=[gen1_input], outputs=[gen1_output])
s1_generator.summary()

################################## S 1 Partial - Discriminator // S 2 Generator ###################################

s1_di = keras.layers.convolutional.Conv2D(64, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=(128, 128, 3))(di1_input)
s1_di = keras.layers.convolutional.Conv2D(64, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_di)
s1_di = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s1_di)

s1_di = keras.layers.convolutional.Conv2D(128, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_di)
s1_di = keras.layers.convolutional.Conv2D(128, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_di)
s1_di = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s1_di)

s1_di = keras.layers.convolutional.Conv2D(256, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_di)
s1_di = keras.layers.convolutional.Conv2D(512, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_di)
s1_di = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s1_di)

s1_di = keras.layers.convolutional.Conv2D(512, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_di)
s1_di = keras.layers.convolutional.Conv2D(768, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_di)
s1_di = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s1_di)

s1_di = keras.layers.convolutional.Conv2D(768, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_di)
s1_di = keras.layers.convolutional.Conv2D(1024, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_di)
di1_output = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last', name='inter')(s1_di)

s1_c = merge([di1_output, concatenated_vector], mode='concat',)
s1_c = keras.layers.core.Reshape((1, 18432))(s1_c)
s1_c = keras.layers.core.Dense(1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s1_c)

s1_discriminator = Model(inputs=[di1_input, concatenated_vector], outputs=[s1_c])
#s1_discriminator.summary()

#############################################  S 1 Full model  #########################################

s1 = s1_discriminator([s1_generator(gen1_input), concatenated_vector])
train_model_s1 = Model(inputs=[gen1_input, concatenated_vector], outputs=[s1])


#train_model_s1.summary()

#from keras.utils import plot_model
#plot_model(train_model_s1, to_file='test.png')
#plot_model(model_1, to_file='model1.png')
#plot_model(model_2, to_file='model2.png')

##########################  Train Loop stage 1 #############################

<<<<<<< HEAD
d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
s1_generator.compile(loss='binary_crossentropy', optimizer="SGD")
train_model_s1.compile(
loss='binary_crossentropy', optimizer=g_optim)
s1_discriminator.trainable = True
s1_discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

for n in range(294, 304):
    img_path = img_names[n] #Real Img
    try:
        img = image.load_img(os.path.join("./pubfig_resized/",img_path), target_size=(128, 128))
    except Exception as e:
        print(e)
        print(os.path.join("./pubfig_resized/",img_path))
        continue
    img = image.img_to_array(img)
    print("image" + str(img.shape))
    print(encoding[n])
    conc = np.concatenate([encoding[n], np.zeros(54)])
    print(conc.shape)
    enc = np.tile(conc,16)
    print(enc.shape)
    enc = enc.reshape((1,4,4,128)) #Textual embedding
    noise = np.random.normal(size=(1,4,4,1024)) # Noise vector
    generator_input = np.concatenate((noise, enc), axis=3)

    print(generator_input.shape, type(generator_input), gen1_input.shape, type(gen1_input))
    synth_img = s1_generator.predict(x=generator_input) #Generated synthetic image
    both_img = np.array([img,synth_img]) #Both images in array
    both_gt = [1,0] #Ground truth values for both images
    s1_discriminator.train_on_batch(x=[both_img, enc], y=both_gt)
    train_model_s1.train_on_batch(x=np.array([img]), y=np.array([1]))



=======
for n in range(10000):
    img_path = img_names[n] #Real Img
    try:
        img = image.load_img(os.path.join("pubfig_imgs",img_path), target_size=(128, 128))
    except Exception as e:
        print(e)
        print("Image not found")
        print(img_path)
        continue
    img = image.img_to_array(img)
>>>>>>> origin/master

    print(encoding[n])
    conc = np.concatenate([encoding[n], np.zeros(54)])
    print(conc.shape)
    enc = np.tile(conc,16)
    print(enc.shape)
    enc = enc.reshape((4,4,128)) #Textual embedding
    noise = np.random.normal(size=(4,4,1024)) # Noise vector
    generator_input = np.concatenate((noise, enc), axis=2)
    synth_img = s1_generator.predict(x=generator_input) #Generated synthetic image
    both_img = np.array([img,synth_img]) #Both images in array
    both_gt = [1,0] #Ground truth values for both images
    s1_discriminator.train(x=both_img, y=both_gt)
    train_model_s1.train(x=np.array([img]), y=np.array([1]))


<<<<<<< HEAD
################################################## S 2 Generator  ##########################################################

'''s2_gen = keras.layers.convolutional.Conv2D(64, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=(128, 128, 3))(gen2_input)
s2_gen = keras.layers.convolutional.Conv2D(64, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_gen)
s2_gen = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s2_gen)

s2_gen = keras.layers.convolutional.Conv2D(128, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_gen)
s2_gen = keras.layers.convolutional.Conv2D(128, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_gen)
s2_gen = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s2_gen)

s2_gen = keras.layers.convolutional.Conv2D(256, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_gen)
s2_gen = keras.layers.convolutional.Conv2D(512, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_gen)
s2_gen = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s2_gen)

s2_gen = keras.layers.convolutional.Conv2D(512, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_gen)
s2_gen = keras.layers.convolutional.Conv2D(768, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_gen)
s2_gen = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s2_gen)

s2_gen = keras.layers.convolutional.Conv2D(768, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_gen)
s2_gen = keras.layers.convolutional.Conv2D(1024, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_gen)
gen2_output_inter = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last', name='inter')(s2_gen)

s2_c = merge([gen2_output_inter, concatenated_vector], mode='concat',)

s2_c = keras.layers.convolutional.Conv2D(832, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
s2_c = keras.layers.convolutional.Conv2D(832, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
s2_c = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s2_c)

s2_c = keras.layers.convolutional.Conv2D(1024, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
s2_c = keras.layers.convolutional.Conv2D(1024, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
s2_c = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s2_c)

s2_c = keras.layers.convolutional.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
s2_c = keras.layers.convolutional.Conv2D(512, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)

s2_c = keras.layers.convolutional.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
s2_c = keras.layers.convolutional.Conv2D(256, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)

s2_c = keras.layers.convolutional.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
s2_c = keras.layers.convolutional.Conv2D(128, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)

s2_c = keras.layers.convolutional.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
s2_c = keras.layers.convolutional.Conv2D(64, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
=======
>>>>>>> origin/master



################################################## S 2 Generator  ##########################################################
#
# s2_gen = keras.layers.convolutional.Conv2D(64, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=(128, 128, 3))(gen2_input)
# s2_gen = keras.layers.convolutional.Conv2D(64, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_gen)
# s2_gen = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s2_gen)
#
# s2_gen = keras.layers.convolutional.Conv2D(128, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_gen)
# s2_gen = keras.layers.convolutional.Conv2D(128, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_gen)
# s2_gen = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s2_gen)
#
# s2_gen = keras.layers.convolutional.Conv2D(256, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_gen)
# s2_gen = keras.layers.convolutional.Conv2D(512, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_gen)
# s2_gen = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s2_gen)
#
# s2_gen = keras.layers.convolutional.Conv2D(512, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_gen)
# s2_gen = keras.layers.convolutional.Conv2D(768, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_gen)
# s2_gen = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s2_gen)
#
# s2_gen = keras.layers.convolutional.Conv2D(768, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_gen)
# s2_gen = keras.layers.convolutional.Conv2D(1024, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_gen)
# gen2_output_inter = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last', name='inter')(s2_gen)
#
# s2_c = merge([gen2_output_inter, concatenated_vector], mode='concat',)
#
# s2_c = keras.layers.convolutional.Conv2D(832, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
# s2_c = keras.layers.convolutional.Conv2D(832, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
# s2_c = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s2_c)
#
# s2_c = keras.layers.convolutional.Conv2D(1024, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
# s2_c = keras.layers.convolutional.Conv2D(1024, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
# s2_c = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s2_c)
#
# s2_c = keras.layers.convolutional.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
# s2_c = keras.layers.convolutional.Conv2D(512, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
#
# s2_c = keras.layers.convolutional.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
# s2_c = keras.layers.convolutional.Conv2D(256, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
#
# s2_c = keras.layers.convolutional.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
# s2_c = keras.layers.convolutional.Conv2D(128, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
#
# s2_c = keras.layers.convolutional.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
# s2_c = keras.layers.convolutional.Conv2D(64, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
#
# s2_c = keras.layers.convolutional.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
# s2_c = keras.layers.convolutional.Conv2D(32, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
#
# s2_c = keras.layers.convolutional.Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
# s2_c = keras.layers.convolutional.Conv2D(16, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
#
# s2_c = keras.layers.convolutional.Conv2DTranspose(8, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
# s2_c = keras.layers.convolutional.Conv2D(8, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
#
# s2_c = keras.layers.convolutional.Conv2DTranspose(6, (4, 4), strides=(2, 2), padding='valid', data_format='channels_last', activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
# gen2_output = keras.layers.convolutional.Conv2D(6, 3, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c)
#
#
# s2_generator = Model(inputs=[gen2_input, concatenated_vector], outputs=[gen2_output])
# #s2_generator.summary()
#
#
# ####################################  S 2 Discriminator ####################################
#
# s2_di = keras.layers.convolutional.Conv2D(64, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, input_shape=(256, 256, 6))(di2_input)
# s2_di = keras.layers.convolutional.Conv2D(64, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_di)
# s2_di = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s2_di)
#
# s2_di = keras.layers.convolutional.Conv2D(128, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_di)
# s2_di = keras.layers.convolutional.Conv2D(128, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_di)
# s2_di = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s2_di)
#
# s2_di = keras.layers.convolutional.Conv2D(256, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_di)
# s2_di = keras.layers.convolutional.Conv2D(512, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_di)
# s2_di = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s2_di)
#
# s2_di = keras.layers.convolutional.Conv2D(512, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_di)
# s2_di = keras.layers.convolutional.Conv2D(768, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_di)
# s2_di = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s2_di)
#
# s2_di = keras.layers.convolutional.Conv2D(512, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_di)
# s2_di = keras.layers.convolutional.Conv2D(768, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_di)
# s2_di = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last')(s2_di)
#
# s2_di = keras.layers.convolutional.Conv2D(768, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_di)
# s2_di = keras.layers.convolutional.Conv2D(1024, 3, strides=(1, 1), padding='same', data_format='channels_last', dilation_rate=(1, 1), activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_di)
# di2_output = keras.layers.pooling.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last', name='inter2')(s2_di)
#
# s2_c2 = merge([di2_output, concatenated_vector], mode='concat',)
# print(s2_c2.shape)
# s2_c2 = keras.layers.core.Reshape((1, 18432))(s2_c2)
# s2_c2 = keras.layers.core.Dense(1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(s2_c2)
#
# s2_discriminator = Model(inputs=[di2_input,concatenated_vector], outputs=[s2_c2])
#s2_discriminator.summary()

#############################################  S 2 Full model  ############################################
#s2 = s2_generator
#s2 = s2_discriminator([s2_generator([gen2_input, concatenated_vector]), concatenated_vector])
#train_model_s2 = Model(inputs=[gen2_input, concatenated_vector], outputs=[s2])
#train_model_s2.summary()


#from keras.utils import plot_model
#plot_model(train_model_s2, to_file='model1.png')
#plot_model(model_1, to_file='model1.png')
#plot_model(model_2, to_file='model2.png')

###############################################  Stack GAN  ###############################################

#stack_gan = s2_discriminator([s2_generator([s1_generator(gen1_input), concatenated_vector]), concatenated_vector])

#stack_gan = Model(inputs=[gen1_input, concatenated_vector], outputs=[stack_gan])

#stack_gan.summary()
#from keras.utils import plot_model
#plot_model(stack_ganc, to_file='stack_gan.png')


####################################  TrainLoop Stage 2  ####################################

'''
