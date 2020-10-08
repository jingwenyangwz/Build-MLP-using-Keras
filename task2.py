from __future__ import print_function

#import tensorflow as tf
#import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt

#print('tensorflow:', tf.__version__)
#print('keras:', tensorflow.keras.__version__)


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


#We want to have a binary classification: digit 0 is classified 1 and 
#all the other digits are classified 0

y_new = np.zeros(y_train.shape)
y_new[np.where(y_train==0.0)[0]] = 1
y_train = y_new

y_new = np.zeros(y_test.shape)
y_new[np.where(y_test==0.0)[0]] = 1
y_test = y_new

num_classes = 1
neuron_num_hidden = [16,32,64,128]
batch = 30
activation_funcs = ['sigmoid','relu','tanh']
#Let start our work: creating a neural network
#Here we are using one hidden layer
#####TO COMPLETE
samples_num = x_train.shape[0]
test_accuracy = []
test_losslist = []
dir='/media/jingwen/04ccc39c-85bc-41b4-9194-f080cab179d5/00UBX/deeplearning_cv/lab3/outputs/task2'
# define the keras model
'''
#try out different activation funcitons: relu, tanh, sigmoid
for activation_one in activation_funcs:
    neuron_num=64
    model = Sequential()
    model.add(Dense(neuron_num, input_dim=784, activation=activation_one))
    model.add(Dense(num_classes, input_dim=neuron_num, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    nn = model.fit(x_train, y_train, validation_split = 0.4, epochs = 100, batch_size= batch)
    # evaluate the keras model
    test_loss, test_acc  = model.evaluate(x_test, y_test, batch_size = batch)
    test_losslist.append(test_loss)
    test_accuracy.append('Test Accuracy: %.2f' % (test_acc*100))

    plt.plot(nn.history['accuracy'])
    plt.plot(nn.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(dir+'/task2_acc_'+ activation_one +'.png')
    #plt.show()
    plt.close()

    plt.plot(nn.history['loss']) 
    plt.plot(nn.history['val_loss']) 
    plt.title('Model loss') 
    plt.ylabel('Loss') 
    plt.xlabel('Epoch') 
    plt.legend(['Train', 'Test'], loc='upper left') 
    plt.savefig(dir+'/task2_loss_'+ activation_one +'.png')
    #plt.show()
    plt.close()

    model.save_weights(dir+'/weights_'+ activation_one +'.h5')
    del model

'''
#model.load_weights('model.h5')

#try out different neuron numbers: 16,32,64,128
for neuron_num in neuron_num_hidden:
    model = Sequential()
    model.add(Dense(neuron_num, input_dim=784, activation='sigmoid'))
    model.add(Dense(num_classes, input_dim=neuron_num, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    nn = model.fit(x_train, y_train, validation_split = 0.4, epochs = 100, batch_size= batch)
    # evaluate the keras model
    test_loss, test_acc  = model.evaluate(x_test, y_test, batch_size = batch)
    test_losslist.append(test_loss)
    test_accuracy.append('Test Accuracy: %.2f' % (test_acc*100))

    plt.plot(nn.history['accuracy'])
    plt.plot(nn.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('{}/task2_acc_neuron{}.png'.format(dir, neuron_num), format='png')
    #plt.show()
    plt.close()

    plt.plot(nn.history['loss']) 
    plt.plot(nn.history['val_loss']) 
    plt.title('Model loss') 
    plt.ylabel('Loss') 
    plt.xlabel('Epoch') 
    plt.legend(['Train', 'Test'], loc='upper left') 
    plt.savefig('{}/task2_loss_neuron{}.png'.format(dir, neuron_num), format='png')
    #plt.show()
    plt.close()

    model.save_weights(dir+'/weights_neuron'+str(neuron_num) +'.h5')
    with open('ACCURACY'+str(neuron_num)+'.txt', 'w') as f:
        for item in nn.history['accuracy']:
            f.write("%s\n" % item)
    with open('ACCURACY'+str(neuron_num)+'.txt', 'w') as f:
            for item in nn.history['val_accuracy']:
                f.write("%s\n" % item)
    with open('ACCURACY'+str(neuron_num)+'.txt', 'w') as f:
            for item in nn.history['loss']:
                f.write("%s\n" % item)
    with open('ACCURACY'+str(neuron_num)+'.txt', 'w') as f:
            for item in nn.history['val_loss']:
                f.write("%s\n" % item)
    del model

plt.plot(test_accuracy) 
plt.title('Test Accuracy') 
plt.ylabel('Accuracy') 
plt.xlabel('Neuron Numbers') 
plt.savefig('{}/test_accuracy_neuron{}.png'.format(dir,neuron_num), format='png')

plt.xlabel('Activation Functions') 
#plt.savefig(dir+'/test_accuracy_'+ activation_one+'png')
#plt.show()
plt.close()

plt.plot(test_losslist) 
plt.title('Test loss') 
plt.ylabel('Loss') 
plt.xlabel('Neuron Numbers') 
plt.savefig('{}/test_loss_neuron{}.png'.format(dir,neuron_num), format='png')
#plt.xlabel('Activation Functions') 
#plt.savefig(dir+'/test_loss_'+activation_one+'.png')
#plt.show()
plt.close()