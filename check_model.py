import os 
from time import time
from glob import glob

import numpy as np
import cv2

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


def prepareImage(img):
    w, h = 19, 19
    img = img[:h,:w]
    hist = cv2.reduce(img, 0, cv2.REDUCE_AVG).reshape(-1)
    maxpoint = np.argmin(hist)
    newimg = np.roll(img, (0, w//2-maxpoint ))
    return np.array( newimg > 150 , dtype='int').flatten()

base_images_path = glob(os.path.join('LettersDataset','*'))
base_letters = list(range(len(base_images_path)))
letters_images_path = []
letters = []
for i, path in enumerate(base_images_path):
    paths = glob(os.path.join(path,'*','Effra_Md.png'))
    letters_images_path.extend(paths)
    letters.extend([i]*len(paths))

letters = letters * 30
letters_images = [ prepareImage(cv2.cvtColor( cv2.imread(p), cv2.COLOR_BGR2GRAY))  for p in letters_images_path ]  * 30
X_train, y_train = letters_images, letters
X_test, y_test = letters_images[:len(letters_images_path)], letters[:len(letters_images_path)]



names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", 
         "Random Forest", "Neural Net", "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    GaussianNB()]

results = []
for name, clf in zip(names, classifiers):
    tic = time()
    clf.fit(X_train, y_train)
    toc = time()
    score = clf.score(X_test, y_test)
    tooc = time()
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(X_test)
    else:
        Z = clf.predict_proba(X_test)
    results += [Z]
    print(name, str(score)[:4], 'train time: ', str(toc-tic)[:6], ' test time: ', str(tooc -toc)[:6])

