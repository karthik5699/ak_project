# Importing necessary libraries/modules
import cv2  # OpenCV library for image processing
import numpy as np  # Library for numerical computing
import os  # Library for interacting with the operating system
from random import shuffle  # Function to shuffle lists randomly
from tqdm import tqdm  # tqdm is used to display progress bars while iterating over sequences

# Defining directories and parameters
TRAIN_DIR = 'train/train'  # Directory containing training images
TEST_DIR = 'test/test'  # Directory containing testing images
IMG_SIZE = 50  # Size of images to be resized to
LR = 1e-3  # Learning rate for training the model
MODEL_NAME = 'healthyvsunhealthy-{}-{}.model'.format(LR, '2conv-basic')  # Name of the trained model file

# Function to label images based on their filenames
def label_img(img):
    word_label = img[0]
    if word_label == 'h':
        return [1,0,0,0]  # If the image filename starts with 'h', it belongs to class 1 (healthy)
    elif word_label == 'b':
        return [0,1,0,0]  # If the image filename starts with 'b', it belongs to class 2 (unhealthy)
    elif word_label == 'v':
        return [0,0,1,0]  # If the image filename starts with 'v', it belongs to class 3 (virus)
    elif word_label == 'l':
        return [0,0,0,1]  # If the image filename starts with 'l', it belongs to class 4 (bacterial)

# Function to create training data
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)  # Get label for the image
        path = os.path.join(TRAIN_DIR,img)  # Construct path to the image
        img = cv2.imread(path,cv2.IMREAD_COLOR)  # Read the image
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))  # Resize the image to IMG_SIZE x IMG_SIZE
        training_data.append([np.array(img),np.array(label)])  # Append image and its label to training_data list
    shuffle(training_data)  # Shuffle the training data
    np.save('train_data.npy', training_data)  # Save the training data as a .npy file
    return training_data  # Return the training data

# Function to process test data
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)  # Construct path to the image
        img_num = img.split('.')[0]  # Get the image number from the filename
        img = cv2.imread(path,cv2.IMREAD_COLOR)  # Read the image
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))  # Resize the image to IMG_SIZE x IMG_SIZE
        testing_data.append([np.array(img), img_num])  # Append image and its number to testing_data list
    shuffle(testing_data)  # Shuffle the testing data
    np.save('test_data.npy', testing_data)  # Save the testing data as a .npy file
    return testing_data  # Return the testing data

# Calling the functions to create training and testing data
train_data = create_train_data()  # Create training data
# test_data = process_test_data()  # Process testing data (not used in this code)

# Importing necessary modules for building the neural network model
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf

# Resetting the default graph
tf.reset_default_graph()

# Building the convolutional neural network (CNN) model
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')  # Input layer

# Adding convolutional and max pooling layers
convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

# Adding fully connected layers and dropout for regularization
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 4, activation='softmax')  # Output layer with softmax activation
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

# Creating the DNN model
model = tflearn.DNN(convnet, tensorboard_dir='log')

# Loading the model if it already exists
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

# Splitting the training data into train and test sets
train = train_data[:-500]  # Training set
test = train_data[-500:]  # Testing set

# Preparing the training and testing data
X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)  # Features for training
Y = [i[1] for i in train]  # Labels for training

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)  # Features for testing
test_y = [i[1] for i in test]  # Labels for testing

# Training the model
model.fit({'input': X}, {'targets': Y}, n_epoch=8, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=40, show_metric=True, run_id=MODEL_NAME)

# Saving the trained model
model.save(MODEL_NAME)
