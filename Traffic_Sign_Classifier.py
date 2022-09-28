import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

from tkinter import Tk, filedialog, Button, BOTTOM, Label


parent_directory = "Datasets"


""" EXPLORE THE DATA SET """

data_set = []
train_data = pd.read_csv(os.path.join(parent_directory, 'Train.csv'))
class_labels = train_data['ClassId'].values
images_data = train_data['Path'].values

for x in images_data:
    try:
        print(os.path.join(parent_directory, x))
        image = Image.open(os.path.join(parent_directory, x))
        image = image.resize((30, 30))
        data_set.append(np.array(image))
    except:
        print("Error loading the image")

""" SPLITTING THE DATA SET - TRAIN & TEST """

x_train, x_test, y_train, y_test = train_test_split(data_set, class_labels, test_size = 0.3, random_state = 42)
print(x_train.shape, x_test.shape)

y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

""" BUILDING A CNN MODEL"""

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5, 5), activation = 'relu', input_shape = x_train.shape[1:]))
model.add(Conv2D(filters = 32, kernel_size = (5, 5), activation='relu'))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.25))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.25))
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(rate = 0.5))
model.add(Dense(43, activation = 'softmax'))

#Compilation of the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

""" TRAIN AND VALIDATE THE MODEL """

epochs = 15
history = model.fit(x_train, y_train, batch_size = 32, epochs = epochs, validation_data = (x_test, y_test))
model.save("trained_model.h5")

plt.figure(0)
plt.plot(history.history['accuracy'], label = 'Training Accuracy')
plt.plot(history.history['val_accuracy'], label = 'val accuracy')
plt.title('ACCURACY')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label = 'Training loss')
plt.plot(history.history['val_loss'], label = 'val loss')
plt.title('LOSS')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

""" TESTING THE MODEL """

y_test = pd.read_csv(os.path.join(parent_directory, 'Test.csv'))
class_labels = y_test['ClassId'].values
images_data = y_test['Path'].values

data = []
for x in images_data:
    image = Image.open(os.path.join(parent_directory, 'Test.csv'))
    image = image.resize((30, 30))
    data.append(np.array(image))

x_test = np.array(data)
predicted_labels = model.predict_classes(x_test)
print(accuracy_score(class_labels, predicted_labels))


""" TRAFFIC SIGNAL CLASSIFIER GUI """
#load the trained model to classify sign
model = load_model('trained_model.h5')

# Create a dictionary hashing the output of the trained_model.h5 to what the value actually depicts
classes = { 1 : 'Speed limit (20km/h)', 2 : 'Speed limit (30km/h)', 3 : 'Speed limit (50km/h)', 4 : 'Speed limit (60km/h)', 5 : 'Speed limit (70km/h)', 6 : 'Speed limit (80km/h)', 7 : 'End of speed limit (80km/h)', 8 : 'Speed limit (100km/h)', 9 : 'Speed limit (120km/h)', 10 : 'No passing', 11 : 'No passing veh over 3.5 tons', 12 : 'Right-of-way at intersection', 13 : 'Priority road', 14 : 'Yield', 15 : 'Stop', 16 : 'No vehicles', 17 : 'Veh > 3.5 tons prohibited', 18 : 'No entry', 19 : 'General caution', 20 : 'Dangerous curve left', 21 : 'Dangerous curve right', 22 : 'Double curve', 23 : 'Bumpy road', 24 : 'Slippery road', 25 : 'Road narrows on the right', 26 : 'Road work', 27 : 'Traffic signals', 28 : 'Pedestrians', 29 : 'Children crossing', 30 : 'Bicycles crossing', 31 : 'Beware of ice/snow', 32 : 'Wild animals crossing', 33 : 'End speed + passing limits', 34 : 'Turn right ahead', 35 : 'Turn left ahead', 36 : 'Ahead only', 37 : 'Go straight or right', 38 : 'Go straight or left', 39 : 'Keep right', 40 : 'Keep left', 41 : 'Roundabout mandatory', 42 : 'End of no passing', 43 : 'End no passing veh > 3.5 tons'}

#initialise GUI
top=Tk()
top.geometry('800x600')
top.title('Traffic sign classification')
top.configure(background='#cae7fc')

label = Label(top, background = '#cae7fc', font = ('Helvetica', 15, 'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((30, 30))
    image = np.expand_dims(image, axis = 0)
    image = np.array(image)
    print(image.shape)
    pred = model.predict_classes([image])[0]
    sign = classes[pred + 1]
    print(sign)
    label.configure(foreground = '#011638', text = sign)

def show_classify_button(file_path):
    classify_b = Button(top, text = "Classify Image",
    command = lambda: classify(file_path), padx = 10, pady = 5)
    classify_b.configure(background = '#364156', foreground = 'white',
    font=('Helvetica', 10, 'bold'))
    classify_b.place(relx = 0.79, rely = 0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image = im)
        sign_image.image = im
        label.configure(text = '')
        show_classify_button(file_path)
    
    except:
        pass

upload = Button(top, text = "Upload an image", command = upload_image, padx = 10, pady = 5)
upload.configure(background = '#364156', foreground = 'white',font = ('Helvetica', 10, 'bold'))

upload.pack(side = BOTTOM, pady = 50)
sign_image.pack(side = BOTTOM, expand = True)
label.pack(side = BOTTOM, expand = True)
heading = Label(top, text = "KNOW THE INPUT TRAFFIC SIGN", pady = 20, font=('arial', 20, 'bold'))
heading.configure(background = '#cae7fc', foreground = '#364156')
heading.pack()
top.mainloop()