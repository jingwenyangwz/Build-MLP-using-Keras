from __future__ import print_function

#import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt


#print('tensorflow:', tf.__version__)
#print('tensorflow.keras:', tensorflow.keras.__version__)


#load (first download if necessary) the MNIST dataset
# (the dataset is stored in your home direcoty in ~/.keras/datasets/mnist.npz
#  and will take  ~11MB)
# data is already split in train and test datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train : 60000 images of size 28x28, i.e., x_train.shape = (60000, 28, 28)
# y_train : 60000 labels (from 0 to 9)
# x_test  : 10000 images of size 28x28, i.e., x_test.shape = (10000, 28, 28)
# x_test  : 10000 labels
# all datasets are of type uint8

#To input our values in our network Dense layer, we need to flatten the datasets, i.e.,
# pass from (60000, 28, 28) to (60000, 784)
#flatten images
num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels)
x_test = x_test.reshape(x_test.shape[0], num_pixels)

#Convert to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalize inputs from [0; 255] to [0; 1]
x_train = x_train / 255
x_test = x_test / 255


#Convert class vectors to binary class matrices ("one hot encoding")
## Doc : https://keras.io/utils/#to_categorical
y_train = tensorflow.keras.utils.to_categorical(y_train)
y_test = tensorflow.keras.utils.to_categorical(y_test)

#get the number of classes and number of samples
num_classes = y_train.shape[1]
samples_num = x_train.shape[0]
test_accuracy = []
test_losslist = []

#default parameter settings:
neuron_num=300#64
batch = 32

#try out 8 different optimizers
#found on https://faroit.com/keras-docs/1.0.6/optimizers/
#the first three are the most common optimizers

#optimizers = ['adam','rmsprop','SGD','adagrad','Adadelta','Adamax','Nadam','Ftrl'] 
optimizers = ['adam']
#opt = tensorflow.keras.optimizers.Adam(learning_rate=0.002)
#optimizers = [opt] 

#Here we are building multi-class neural network
#modify the directory you want to save outputs
dir='/Users/anne-claire/Desktop/final/outputs/task3'
# define the keras model

#TODO
#try out different optimizers:
#find the architexture that maximize the accuracy
for the_optimizer in optimizers:

    model = Sequential()
    model.add(Dense(neuron_num, input_dim=784, activation='relu'))
    #TODO maybe try this dropout value
    #model.add(Dropout(0.3))

    model.add(Dense(num_classes, input_dim=neuron_num, activation='softmax'))

    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer= the_optimizer, metrics=['accuracy'])

    # fit the keras model on the dataset
    nn = model.fit(x_train, y_train, validation_split = 0.4, epochs = 100, batch_size= batch)
    # evaluate the keras model
    test_loss, test_acc  = model.evaluate(x_test, y_test, batch_size = batch)
    test_losslist.append(test_loss)
    test_accuracy.append('Test Accuracy: %.2f' % (test_acc*100))
    
    """
    #save the accuracy curve on training set and validation set
    plt.plot(nn.history['accuracy'])
    plt.plot(nn.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(dir+'/task3_acc_'+ the_optimizer +'.png')
    #plt.show()
    plt.close()

    #save the loss curve on training set and validation set
    plt.plot(nn.history['loss']) 
    plt.plot(nn.history['val_loss']) 
    plt.title('Model loss') 
    plt.ylabel('Loss') 
    plt.xlabel('Epoch') 
    plt.legend(['Train', 'Test'], loc='upper left') 
    plt.savefig(dir+'/task3_loss_'+ the_optimizer +'.png')
    #plt.show()
    plt.close()

    model.save_weights(dir+'/weights_'+ the_optimizer +'.h5')
    """
    del model

#save the accuracy curve on test set 
plt.plot(test_accuracy) 
plt.title('Test Accuracy') 
plt.ylabel('Accuracy') 
plt.xlabel('Optimizers') 
plt.savefig(dir+'/test_accuracy_optimizers.png')
#plt.show()
plt.close()

#save the loss curve on test set 
plt.plot(test_losslist) 
plt.title('Test loss') 
plt.ylabel('Loss') 
plt.xlabel('Optimizers') 
plt.savefig(dir+'/test_loss_neuron.png')
#plt.show()
plt.close()
