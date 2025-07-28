#Assignment : Deep Learning Assignment 01,
#Name : Majid Muhammad Hamza,
#Neptun : G4PCXW,
#Date : 23/09/2024.

from tensorflow import keras
from tensorflow.keras import layers, callbacks
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

red_wine = pd.read_csv('https://raw.githubusercontent.com/karsarobert/DeepLearning2024/main/red-wine.csv')

y = red_wine.quality
X = red_wine.drop(['quality'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


#early stopping parametrization
early_stopping = callbacks.EarlyStopping(
    min_delta = 0.001,
    patience = 15,
    restore_best_weights = True,
)

#one-hot encoding for quality
max_quality = y_train.max()+1  #get the maximum value of quality

y_train = keras.utils.to_categorical(y_train, max_quality)
y_test = keras.utils.to_categorical(y_test, max_quality)


#scale to [0,1]
max_ = X_train.max(axis = 0)
min_ = X_train.min(axis = 0)
X_train = (X_train - min_) / (max_ - min_)
X_test = (X_test - min_) / (max_ - min_)

#buliding the specified model
model = keras.Sequential([

    #the three hidden layers

    #input shape determined by the database columns
    layers.Dense(1024, activation = 'relu', input_shape = [X.shape[1]]),
    layers.Dropout(0.3),#dropout = 30%
    layers.Dense(1024, activation = 'relu'),
    layers.Dropout(0.3),
    layers.Dense(1024, activation = 'relu'),
    layers.Dropout(0.3),

    #the ouput layer
    layers.Dense(max_quality, activation = 'softmax'),#softmax activation used for classification task
])

#compiling the model
model.compile(
    optimizer = "adam", loss = "categorical_crossentropy", metrics = ['accuracy']
)

#starting model training
history = model.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    batch_size = 256, epochs = 100,
    callbacks = [early_stopping],
    verbose = 1,
)

#evaluating the losses
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss', 'accuracy', 'val_accuracy']].plot()
plt.show()