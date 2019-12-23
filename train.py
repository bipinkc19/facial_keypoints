import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

from model import get_model

training = pd.read_csv('./data_set.csv').drop(columns=['Unnamed: 0'])
training = training.dropna()
training['Image'] = [list(map(lambda x: float(x), training['Image'][0].split())) for i in range(len(training))]

X_train = np.asarray([training['Image']], dtype=np.float32).reshape(training.shape[0], 96, 96, 1)
y_train = training.drop(['Image'], axis=1).to_numpy()

model = get_model()
save_model_each_epoch = keras.callbacks.ModelCheckpoint('./model{epoch:05d}.h5', period=50)

model.compile(optimizer='Adam', 
              loss='mse', 
              metrics=['mae'])

model.fit(X_train, y_train, epochs=500, callbacks=[save_model_each_epoch])
model.save('./final_model.hdf5')
