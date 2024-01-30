
print("Start of program")

import tensorflow as tf 
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split
import numpy as np




#fashiondata = tf.keras.datasets.mnist
TempsData = pd.read_csv('GlobalTemperatures.csv', index_col='dt')

X = TempsData.copy()

# Remove target
y = X.pop('LandAverageTemperature')

print(X.head())

preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse=False),
     make_column_selector(dtype_include=object)),
)

X = preprocessor.fit_transform(X)
#y = np.log(y)#log will not work here as working with negative data
y = X

input_shape = [X.shape[1]]
print("Input shape: {}".format(input_shape))


x_train, x_test, y_train, y_test = train_test_split(X, y)

#Checking if data loaded correctly
print(x_test.shape)
print(x_train.shape)

#pre-proccessing
x_train, x_test = x_train/255, x_test/255

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=input_shape),#input layer
    tf.keras.layers.Dense(128, activation='relu'),#activation layer
    tf.keras.layers.Dropout(0.2),#dropout layer
    tf.keras.layers.Dense(7, activation='softmax')#activation layer
])

model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=500)

model.evaluate(x_test, y_test)

