from chars_frequencies import chars_codes, chars_decode
import os
import random
import shutil
from tqdm import tqdm
import cv2

def officialCharsStats(datasetPath):
    folders = [int(i) for i in os.listdir(datasetPath)]
    file = open(datasetPath + "stats.txt", 'w')
    mini = ['', 1000000000]
    for folder in sorted(folders):
        files = os.listdir(datasetPath + str(folder))
        if len(files) < mini[1]:
            mini[1] = len(files)
            mini[0] = chars_decode[folder]
        file.write(f"{chars_decode[folder]}\t{len(files)}\n")
    file.write(f"minimum:\t{mini[0]}\t{mini[1]}")
    file.close()

def copyLetters(src, dst):
    folders = list(range(29))
    for folder in sorted(folders):
        files = os.listdir(src + str(folder))
        random.shuffle(files)
        files = files[:2000]
        for i, file in enumerate(files):
            shutil.copy2(src + str(folder) + "/" + file, dst + str(folder) + "/" + f"{i}.png")


def deleteFilesWithThr(path, thr):
    files = os.listdir(path)
    for f in files:
        img = cv2.imread(path + f)
        if img.shape[1] < thr:
            os.remove(path + f)

def deleteFilesWithThrGreater(path, thr):
    files = os.listdir(path)
    for f in files:
        img = cv2.imread(path + f)
        if img.shape[1] > thr:
            os.remove(path + f)


if __name__ == "__main__":
    officialCharsStats("CleanedLettersDataset/")

    # copyLetters("OfficialLettersDataset/", "CleanedLettersDataset/")

    # deleteFilesWithThr("CleanedLettersDataset/28/", 18)

    # deleteFilesWithThrGreater("CleanedLettersDataset/26/", 35)