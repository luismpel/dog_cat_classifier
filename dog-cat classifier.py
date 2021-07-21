# CNN - precisión obtenida = 99.41%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Carga del modelo 
from keras.models import model_from_json

# carga del json y creación del modelo
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# carga de los valores de los pesos al nuevo modelo
loaded_model.load_weights("model.h5")
print("El modelo se ha cargado correctamente")
classifier = loaded_model

# Carga de imagen y predicción
from tensorflow.keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import tkinter as tk
from tkinter import filedialog, messagebox

root = tk.Tk()
root.withdraw()

img_path = filedialog.askopenfilename()

img = image.load_img(img_path, 
                     target_size=(128, 128))
plt.imshow(img)
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = preprocess_input(img_batch)

y_pred = classifier.predict(img_preprocessed)
print(y_pred)
if y_pred[0] > 0.5:
    print('\n\n\n\nEsto es un perro\n\n\n')
    messagebox.showinfo(message = '\tEs un perro\t', title = 'Clasificación CNN')
else: 
    print('\n\n\n\nEsto es un gato\n\n\n')
    messagebox.showinfo(message = '\tEs un gato\t', title = 'Clasificación CNN')
'''
# Construcción del modelo de CNN #

# Importar las librerias y paquetes
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Inicializar la CNN
classifier = Sequential()

# Se añade la capa de convolución (32 mapas de características de 3x3)
# El input son imágenes de 64x64 con 3 canales de color (RGB)
classifier.add(
    Conv2D(
        filters = 64, 
        kernel_size = (3, 3), 
        input_shape = (128, 128, 3), 
        activation = "relu")
    )

# Se añade la capa de Max Pooling con un kernel 2x2 (reduce las imágenes a la mitad)
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Se añade una segunda capa de convolución y pooling
classifier.add(
    Conv2D(
        filters = 64, 
        kernel_size = (3, 3), 
        activation = "relu")
    )

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Se aplanan los datos
classifier.add(Flatten())

# Se añade la ANN que procesará los datos (Full Connection)
classifier.add(
    Dense(
        units = 256, # hay que experimentar con este valor
        activation = "relu", 
    ))

classifier.add(
    Dense(
        units = 128, # hay que experimentar con este valor
        activation = "relu", 
    ))

# Se añade la capa de salida
classifier.add(
    Dense(
        units = 1, # Al haber dos categorías se puede usar un único nodo de salida
        activation = "sigmoid", 
    ))

# Compilación de la CNN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# 2. Entrenamiento de la CNN #
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_dataset = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

testing_dataset = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_dataset,
        steps_per_epoch = len(training_dataset),
        epochs = 100,
        validation_data = testing_dataset,
        validation_steps = len(testing_dataset))

# Almacenado del modelo
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights("model.h5")
print("Se ha guardado el modelo 'correctamente")
'''