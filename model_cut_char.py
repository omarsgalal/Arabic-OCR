import os 
from glob import glob
from preprocessing import preprocess
import cv2 
from size_dict import characterDict
from tqdm import tqdm
import re

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


from sklearn.model_selection import train_test_split
import numpy as np 

def get_correct_images():
    with open("validimages.txt") as fin:
        paths = fin.read().strip().split('\n')
    return [(os.path.join('dataset','scanned',p), os.path.join('dataset','text',p.replace('.png','.txt'))) for p in paths]

def get_text(path):
    with open(path) as fin:
        return fin.read().strip()

def process_all(files):
    count = 0
    geven_images = []
    geven_char = []
    for img_path, txt_path in tqdm(files):
        txt = get_text(txt_path)
        # if len(re.sub(r'[\u061F-\u066A|\s]', "", txt) ) != 0: continue
        img = cv2.imread(img_path)
        linesOfWords, numWords, linesImages = preprocess(img)
        if numWords != len(txt.split(' ')): continue
        words = txt.split(' ')
        words_ind = 0
        for l, line_img in enumerate(linesOfWords):
            for w, word_img in enumerate(line_img):
                word = words[words_ind]
                count += len(word)
                words_ind += 1
                img_indx = len(word_img[0])
                geven_images += [word_img]
                geven_char += [word[0]]
                # for i, char in enumerate(word):
                #     if i == 0:
                #         img_indx -= characterDict[char][0]
                #         char_img = word_img[:,img_indx:img_indx+characterDict[char][0]]
                #     elif i == len(word) - 1:
                #         img_indx -= characterDict[char][2]
                #         char_img = word_img[:,img_indx:img_indx+characterDict[char][2]]
                #     else:
                #         img_indx -= characterDict[char][1]
                #         char_img = word_img[:,img_indx:img_indx+characterDict[char][1]]
                #     if char_img.shape[1] == 0: continue
                #     # cv2.imwrite(os.path.join('dataset','chars', f"{txt_path.split('/')[-1].split('.')[0]}_{words_ind-1}_{l}_{w}_{i}_{ord(char)}.png"), char_img)
    print(len(geven_images))        
    return geven_images, geven_char

def prepare_to_train(imgs):
    shape = [0,0]
    newImgs = []
    for img in imgs: shape = np.maximum(shape,img.shape)
    for img in imgs:
        # import pdb; pdb.set_trace()
        newImg = np.zeros((shape[0], 20))
        w,h = img.shape
        newImg[:w,:min(20,h)] = img[:, :20]
        newImgs += [newImg.flatten()]
    return newImgs
    

def train(imgs, chrs):
    imgs = prepare_to_train(imgs)
    X_train, X_test, y_train, y_test = train_test_split(imgs, chrs, test_size=0.1, random_state=42)

    clf = SVC(gamma=2, C=1, probability=True)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    
    print(score)

    
if __name__ == "__main__":
    files_path = get_correct_images()
    
    imgs, chrs = process_all(files_path)
    train(imgs, chrs)
