import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import get_model
training = pd.read_csv('./data/training.csv')
training = training.dropna()

training['Image'] = training['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape((96,96)))

X_train = np.asarray([training['Image']], dtype=np.uint8).reshape(training.shape[0], 96, 96, 1).to_numpy()
y_train = training.drop(['Image'], axis=1).to_numpy()

test = pd.read_csv('./data/training.csv')
test = test.dropna()

test['Image'] = test['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape((96,96)))

X_test = np.asarray([test['Image']], dtype=np.uint8).reshape(test.shape[0], 96, 96, 1).to_numpy()
y_test = test.drop(['Image'], axis=1).to_numpy()

model = get_model()

model.compile(optimizer='Adam', 
              loss='mse', 
              metrics=['mae'])

model.fit(X_train, y_train, epochs=500)
