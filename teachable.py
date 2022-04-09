import cv2
import os
import shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import numpy as np

'''
Class 1 
'''

class1 = input("Enter class 1 : ")

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter1 = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)
    k = cv2.waitKey(1)

    if k%256 == 27:     #if u press esc
        break


    elif k % 256 == 32:   #if u press space
        img_name = "opencv_frame_{}.png".format(img_counter1)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        path1 = r"\{0}".format(img_name)     #path inside main folder
        path2 = r"/class1"     #path to class1
        shutil.move(path1, path2)
        img_counter1 += 1

cam.release()

cv2.destroyAllWindows()

try :
    initial = r"/class1"         #path to class1
    final = r"/data/{0}".format(class1)      #path inside data
    os.rename(initial, final)
except :
    pass


'''
Class 2
'''

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

class2 = input("Enter class 2 : " )

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)
    k = cv2.waitKey(1)

    if k % 256 == 27 :    #esc pressed
        break

    elif k%256 == 32:     #space pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        path1 = r"\{0}".format(img_name)    #path inside main folder
        path2 = r"\data\class2"       #path to class2
        shutil.move(path1, path2)
        img_counter += 1

    

cam.release()

cv2.destroyAllWindows() 

try :
    initial = r"/data/class2"    #path to class2 
    final = r"/data/{0}".format(class2)    #path inside data
    os.rename(initial, final)
except :
    pass


'''
cnn training model
'''

img_height = 28
img_width = 28
batch_size = 32

model = keras.Sequential([
    layers.Input((28,28,3)),
    layers.Conv2D(16, 3, padding = "same"),
    layers.Conv2D(32, 3, padding = "same"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(100),
])

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    r'/data/',      #path inside data
    labels = "inferred",
    label_mode = "int",
    class_names = [class1, class2],
    color_mode = "rgb",
    batch_size = batch_size,
    image_size = (img_height,img_width),
    shuffle = True,
    seed = 321,
    validation_split = 0.2,
    subset = "training",
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    r'/data/',      #path inside data
    labels = "inferred",
    label_mode = "int",
    class_names = [class1, class2],
    color_mode = "rgb",
    batch_size = batch_size,
    image_size = (img_height,img_width),
    shuffle = True,
    seed = 321,
    validation_split = 0.2,
    subset = "validation",
)


def augment(x,y) :
    image = tf.image.random_brightness(x, max_delta = 0.05)
    return image, y


ds_train = ds_train.map(augment)

for epoch in range (10) :
    for x,y in ds_train :
        pass
    for x,y in ds_validation :
        pass

model.compile(
    optimizer = keras.optimizers.Adam(),
    loss = [keras.losses.SparseCategoricalCrossentropy(from_logits = True)],
    metrics = ["accuracy"],
)


model.fit(ds_train, epochs = 10, verbose = 1)
val = model.evaluate(ds_validation)


print()
print(model.summary())
    

f = open("labels.txt","w+")
f.write("0 {}".format(class1))
f.write("\n")
f.write("1 {}".format(class2))

'''
testing the model :
'''

np.set_printoptions(suppress = True)

with open('labels.txt', 'r') as f:
   class_names = f.read().split('\n')

data = np.ndarray(shape=(1, 28, 28, 3), dtype=np.float32)

size = (28, 28) 

cap = cv2.VideoCapture(0)

while cap.isOpened():
    start = time.time()
    ret, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, channels = img.shape

    scale_value = width / height

    img_resized = cv2.resize(imgRGB, size, fx=scale_value, fy=1, interpolation=cv2.INTER_NEAREST)

   # Turn the image into a numpy array
    img_array = np.asarray(img_resized)

   # Normalize the image
    normalized_img_array = (img_array.astype(np.float32) / 127.0) - 1

   # Load the image into the array
    data[0] = normalized_img_array

    #model prediction function
    prediction = model.predict(data)

    index = np.argmax(prediction)

    try :
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        if confidence_score > 1 :
            confidence_score = 1
    except :
        class_name = "none"
        confidence_score = 1
        


    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime
   
    cv2.putText(img, class_name, (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)
    cv2.putText(img, str(float("{:.2f}".format(confidence_score*100))) + "%", (75,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)

    cv2.imshow('Classification Resized', img_resized)
    cv2.imshow('Classification Original', img)


    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap.release()
