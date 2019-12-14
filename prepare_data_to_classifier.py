import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import binarizeImage
import random

PATTERS_PER_CLASS = 1650
NUM_CLASSES = 29
IMG_SIZE = 25
TRAIN_RATIO = 0.8

def removeBlacks(img):
    # img = binarizeImage(img)
    horizontalIndexes = np.array(np.where(np.array(np.sum(img, axis=0)) > 0))[0, [0, -1]]
    verticalIndexes = np.array(np.where(np.array(np.sum(img, axis=1)) > 0))[0, [0, -1]]
    return img[verticalIndexes[0]: verticalIndexes[1] + 1, horizontalIndexes[0]: horizontalIndexes[1] + 1]

def prepareCharImg(img):
    img = removeBlacks(img)
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_AREA)
    thresh = np.array(resized > 10, dtype='int').flatten()
    return thresh

def getDataset(datasetPath):
    X_ALL = np.empty((PATTERS_PER_CLASS * NUM_CLASSES, IMG_SIZE * IMG_SIZE))
    Y_ALL = np.empty((PATTERS_PER_CLASS * NUM_CLASSES,))
    counter = 0
    for i in tqdm(range(NUM_CLASSES)):
        files = os.listdir(datasetPath + f"{i}/")
        random.shuffle(files)
        files = files[:PATTERS_PER_CLASS]
        for f in files:
            img = cv2.imread(datasetPath + f"{i}/" + f, cv2.IMREAD_GRAYSCALE)
            features = prepareCharImg(img)
            X_ALL[counter] = features
            Y_ALL[counter] = i
            counter += 1
    indexes = list(range(PATTERS_PER_CLASS * NUM_CLASSES))
    random.shuffle(indexes)
    X_ALL = X_ALL[indexes]
    Y_ALL = Y_ALL[indexes]
    num_train = int(X_ALL.shape[0] * TRAIN_RATIO)
    X_train = X_ALL[:num_train]
    Y_train = Y_ALL[:num_train]
    X_test = X_ALL[num_train:]
    Y_test = Y_ALL[num_train:]
    print("Data loaded successfully")
    print(f"number of training patterns: {X_train.shape[0]}")
    print(f"number of test patterns: {X_test.shape[0]}")
    return X_train, Y_train, X_test, Y_test
    


if __name__ == "__main__":
    datasetPath = "CleanedLettersDataset/"
    # folder = "4/"
    # imgName = "3.png"
    # for i in range(29):
    #     img = cv2.imread(datasetPath + f"{i}/" + imgName, cv2.IMREAD_GRAYSCALE)
    #     img = prepareCharImg(img)
    #     cv2.imwrite(f"{i}.png", img)
    X_train, Y_train, X_test, Y_test = getDataset(datasetPath)
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)