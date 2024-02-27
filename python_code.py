import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
print(tf.__version__)

#preprocessing the training data
train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
trainig_set = train_datagen.flow_from_directory(r"C:\Users\Deepak\Machine Learning Projects\dogs and cat project\dogs_cats\training_set", target_size=(64, 64), batch_size=32, class_mode="binary")

#preprocessing the testing data
test_datagen=ImageDataGenerator(rescale=1./255)
test_set=test_datagen.flow_from_directory(r"C:\Users\Deepak\Machine Learning Projects\dogs and cat project\dogs_cats\test_set",target_size=(64,64),batch_size=32,class_mode="binary")

# initializing the cnn
cnn=tf.keras.models.Sequential()

#convolution
cnn.add(tf.keras.layers.Conv2D(filters=100,kernel_size=3,activation="relu",input_shape=[64,64,3]))

#pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#adding the second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu",input_shape=[64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#Flattening
cnn.add(tf.keras.layers.Flatten())

#Full connection
cnn.add(tf.keras.layers.Dense(units=128,activation="relu"))

#Output Layer
cnn.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))

# Training the CNN
cnn.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
cnn.fit(x=trainig_set,validation_data=test_set,epochs=25)

#single prediction
import numpy as np
from keras.preprocessing import image
test_image=image.load_img(r"C:\Users\Deepak\Machine Learning Projects\dogs and cat project\dogs_cats\single_prediction\cat_dog.jpg",target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=cnn.predict(test_image)
trainig_set.class_indices
if result[0][0]==1:
    prediction="dog"
else:
    prediction="cat"
print("The Image contains a ",prediction)