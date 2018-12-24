'''
Using CNN to choose the right region 
---------------
1. run this file to train or test
2. include this file to use the model 

'''



import tensorflow as tf 
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

def ModelGetter():
    network = input_data(shape=[None, 64, 64, 1], name='input')
    network = conv_2d(network, 4, 5, strides=2, name='C1')
    network = conv_2d(network, 6, 5, strides=1, name='C2')
    network = max_pool_2d(network, 2, strides=2, name='S1')
    network = conv_2d(network, 16, 5, strides=1, name='C3')
    network = max_pool_2d(network, 2, strides=2, name='S2')
    network = fully_connected(network, 400, activation='relu')
    network = fully_connected(network, 2, activation='softmax')
    targets = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='target')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                        loss='categorical_crossentropy', name='target',
                        placeholder=targets)
    return network


def QRCodePredictor():

    return []


def ModelTrainer(model):
    # first load the data 

    model = tflearn.DNN(model, tensorboard_verbose=0,
                        tensorboard_dir='./nnlog', checkpoint_path='./cpts/cpt', max_checkpoints=50)
    # model.fit
    return model 

def ModelTester(model):

    return


if __name__ == "__main__":
    model = ModelGetter()
    model = ModelTrainer(model)
    ModelTester(model)