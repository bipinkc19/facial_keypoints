import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model

def show_all_keypoints(image, predicted_key_pts):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(np.resize(image, (96, 96)), cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    plt.show()

data = pd.read_csv('./data/test.csv')

# img = np.array(list(map(lambda x: float(x), data.iloc[0]['Image'].split())))
img = cv2.imread("./test_2.jpeg", 0).astype('float32')
# print(img)
img = cv2.resize(img, (96, 96))
img = np.resize(img, (1, 96, 96, 1))

model = load_model('./model_new_008.hdf5')

points = model.predict(img)
points = points.astype(int)

show_all_keypoints(img, np.resize(points, (-1, 2)))
