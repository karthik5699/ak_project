import tkinter as tk  # Importing the tkinter module and aliasing it as 'tk'
from tkinter.filedialog import askopenfilename  # Importing the askopenfilename function from the filedialog module
import shutil  # Importing the shutil module for file operations
import os  # Importing the os module for operating system related operations
import sys  # Importing the sys module for system-specific parameters and functions
from PIL import Image, ImageTk  # Importing Image and ImageTk classes from the PIL module

# Creating a Tkinter window
window = tk.Tk()

# Setting window title
window.title("Plant Disease Detection using Image Processing")

# Setting window size
window.geometry("500x510")

# Setting window background color
window.configure(background="palegoldenrod")

# Creating a label widget for the title
title = tk.Label(text="Click below to choose picture for testing disease.", background="palegoldenrod",
                 fg="royalblue", font=("", 15))
title.grid()

# Function for handling bacterial disease
def bact():
    window.destroy()  # Destroying the current window
    window1 = tk.Tk()  # Creating a new window
    window1.title("Plant Disease Detection using Image Processing")
    window1.geometry("500x510")
    window1.configure(background="palegoldenrod")

    # Function to exit the window
    def exit():
        window1.destroy()

    # Displaying remedies for Bacterial Spot
    rem = "The remedies for Bacterial Spot are:\n\n "
    remedies = tk.Label(text=rem, background="lightyellow", fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " Discard or destroy any affected plants. \n Do not compost them. \n Rotate your tomato plants yearly to prevent re-infection next year. \n Use copper fungicides"
    remedies1 = tk.Label(text=rem1, background="lightyellow", fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)
    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)
    window1.mainloop()

# Function for handling viral disease
def vir():
    window.destroy()
    window1 = tk.Tk()
    window1.title("Plant Disease Detection using Image Processing")
    window1.geometry("650x510")
    window1.configure(background="palegoldenrod")

    def exit():
        window1.destroy()

    rem = "The remedies for Yellow leaf curl virus are: "
    remedies = tk.Label(text=rem, background="lightyellow", fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " Monitor the field, handpick diseased plants and bury them. \n Use sticky yellow plastic traps. \n Spray insecticides such as organophosphates, carbamates during the seedling stage. \n Use copper fungicides"
    remedies1 = tk.Label(text=rem1, background="lightyellow", fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)
    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)
    window1.mainloop()

# Function for handling late blight disease
def latebl():
    window.destroy()
    window1 = tk.Tk()
    window1.title("Plant Disease Detection using Image Processing")
    window1.geometry("520x510")
    window1.configure(background="palegoldenrod")

    def exit():
        window1.destroy()

    rem = "The remedies for Late Blight are: "
    remedies = tk.Label(text=rem, background="lightyellow", fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " Monitor the field, remove and destroy infected leaves. \n Treat organically with copper spray. \n Use chemical fungicides, the best of which for tomatoes is chlorothalonil."
    remedies1 = tk.Label(text=rem1, background="lightyellow", fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)
    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)
    window1.mainloop()

# Function for analysis
def analysis():
    # Importing necessary libraries for image processing
    import cv2
    import numpy as np
    from random import shuffle
    from tqdm import tqdm

    # Directory for verification images
    verify_dir = 'testpicture'
    IMG_SIZE = 50  # Size of the image
    LR = 1e-3  # Learning rate
    MODEL_NAME = 'healthyvsunhealthy-{}-{}.model'.format(LR, '2conv-basic')  # Model name

    # Function to process verification data
    def process_verify_data():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
        np.save('verify_data.npy', verifying_data)
        return verifying_data

    verify_data = process_verify_data()
    verify_data = np.load('verify_data.npy')

    # Importing libraries for deep learning
    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    import tensorflow as tf

    tf.reset_default_graph()

    # Building the convolutional neural network
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')
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
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)
    convnet = fully_connected(convnet, 4, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')
    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')

    import matplotlib.pyplot as plt
    fig = plt.figure()

    for num, data in enumerate(verify_data):
        img_num = data[1]
        img_data = data[0]
        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        model_out = model.predict([data])[0]
        if np.argmax(model_out) == 0:
            str_label = 'healthy'
        elif np.argmax(model_out) == 1:
            str_label = 'bacterial'
        elif np.argmax(model_out) == 2:
            str_label = 'viral'
        elif np.argmax(model_out) == 3:
            str_label = 'lateblight'
        if str_label == 'healthy':
            status = "HEALTHY"
        else:
            status = "UNHEALTHY"
        message = tk.Label(text='Status: ' + status, background="lightsteelblue1", fg="Brown", font=("", 15))
        message.grid(column=0, row=3, padx=10, pady=10)
        if str_label == 'bacterial':
            diseasename = "Bacterial Spot "
            disease = tk.Label(text='Disease Name: ' + diseasename, background="lightsteelblue1", fg="Black", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='Click below for remedies...', background="lightsteelblue1", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=bact)
            button3.grid(column=0, row=6, padx=10, pady=10)
        elif str_label == 'viral':
            diseasename = "Yellow leaf curl virus "
            disease = tk.Label(text='Disease Name: ' + diseasename, background="lightsteelblue1", fg="Black", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='Click below for remedies...', background="lightsteelblue1", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=vir)
            button3.grid(column=0, row=6, padx=10, pady=10)
        elif str_label == 'lateblight':
            diseasename = "Late Blight "
            disease = tk.Label(text='Disease Name: ' + diseasename, background="lightsteelblue1", fg="Black", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='Click below for remedies...', background="lightsteelblue1", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=latebl)
            button3.grid(column=0, row=6, padx=10, pady=10)
        else:
            r = tk.Label(text='Plant is healthy', background="lightsteelblue1", fg="Black", font=("", 15))
            r.grid(column=0, row=4, padx=10, pady=10)
        button = tk.Button(text="Exit", command=exit)
        button.grid(column=0, row=9, padx=20, pady=20)

# Function for opening photo
def openphoto():
    dirPath = "testpicture"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)

    fileName = askopenfilename(initialdir='I:\PlantDiseaseDetection-master', title='Select image for analysis ',
                               filetypes=[('image files', '.jpg')])

    dst = "testpicture"
    shutil.copy(fileName, dst)
    load = Image.open(fileName)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="250", width="500")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=0, row=1, padx=10, pady=10)
    title.destroy()
    button1.destroy()
    button2 = tk.Button(text="Analyse Image", command=analysis)
    button2.grid(column=0, row=2, padx=10, pady=10)
    button1 = tk.Button(text="Get Photo", command=openphoto)
    button1.grid(column=0, row=1, padx=10, pady=10)
    window.mainloop()
