import cv2
import random
import numpy as np
import pandas as pd
import skimage as sk
from skimage import util
from skimage import transform
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import matplotlib.image as imgsave


def random_rotation(image, xy):
    angle = int(random.uniform(-90, 90))
    im_rot = rotate(image, angle) 
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new_x, new_y = np.array([org[:, 0]*np.cos(a) + org[:, 1]*np.sin(a),
            -org[:, 0]*np.sin(a) + org[:, 1]*np.cos(a)])

    return cv2.resize(im_rot, (96, 96)), (new_x + rot_center[0]) * 96/im_rot.shape[0], (new_y + rot_center[1]) * 96/im_rot.shape[0]


def random_noise(image_array, xy):
    ''' Adds random noise to the image. '''
    
    return sk.util.random_noise(image_array), xy[:, 0], xy[:, 1]


def vertical_flip(image_array, xy):
    ''' Retuns vertically flipped image. '''

    angle = 180
    im_rot = rotate(image, angle) 
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new_x, new_y = np.array([org[:, 0]*np.cos(a) + org[:, 1]*np.sin(a),
            -org[:, 0]*np.sin(a) + org[:, 1]*np.cos(a)])
    
    return cv2.resize(im_rot, (96, 96)), (new_x + rot_center[0]) * 96/im_rot.shape[0], (new_y + rot_center[1]) * 96/im_rot.shape[0]


available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'vertical_flip': vertical_flip
}

training = pd.read_csv('./data/training.csv')
training = training.dropna()

training['Image'] = training['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape((96,96)))

X_train = np.asarray([training['Image']], dtype=np.uint8).reshape(training.shape[0], 96, 96)
y_train = training.drop(['Image'], axis=1).to_numpy()

X_augmented = []
Y_augmented = []
for image, label in zip(X_train, y_train):
    
    # Randomly selecting how many transformations to occur for each image.
    num_transformations_to_apply = random.randint(1, len(available_transformations))
    
    num_of_transformations_occured = 0

    label = np.resize(label, (15, 2))
    image = image/255.0

    while (num_of_transformations_occured <= num_transformations_to_apply):
        
        # Randomly selecting which transformation to do.
        key = random.choice(list(available_transformations))
        transformed_image, x, y = available_transformations[key](image, label)

        # transformed_image = cv2.resize(transformed_image, (96, 96))

        num_of_transformations_occured += 1
        new_label = label.copy()
        new_label[:, 0], new_label[:, 1] = x, y
        # print(transformed_image.shape, image.shape)

        X_augmented.append(transformed_image.flatten())
        Y_augmented.append(new_label.flatten())
    X_augmented.append(image.flatten())
    Y_augmented.append(label.flatten())
    break
transformed_df = pd.DataFrame(Y_augmented, columns=training.columns.difference(['Image']))
transformed_df['Image'] = X_augmented
transformed_df['Image'] = [' '.join(list(map(str, i))) for i in transformed_df['Image']]
transformed_df.to_csv('data_set.csv')
