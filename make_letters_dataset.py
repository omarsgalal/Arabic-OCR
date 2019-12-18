import os
from allpreprocessing import img2Chars
from checkwordsegmentations import getText, word2Chars
import cv2
from chars_frequencies import chars_zeros, chars_codes
from tqdm import tqdm
import random

lastChars = chars_zeros.copy()
pathText = "NewDataset/text/"
pathImg = "NewDataset/scanned/"
fileName = "capr1000.png"
pathToSave = "OfficialLettersDataset/"

def saveSeparateLetters(fileName, pathText, pathImg, pathToSave):
    image = cv2.imread(pathImg + fileName)
    segmentedChars = img2Chars(image)
    wordsText = getText(pathText + fileName[:-4] + ".txt")
    if len(segmentedChars) == len(wordsText):
        print("Saving Segmented Character ...")
        for i in tqdm(range(len(segmentedChars))):
            realLetters = word2Chars(wordsText[i])
            if len(realLetters) == segmentedChars[i][1]:
                for j, letImg in enumerate(segmentedChars[i][2]):
                    cv2.imwrite(pathToSave + str(chars_codes[realLetters[j]]) + f"/{lastChars[realLetters[j]]}.png", letImg)
                    lastChars[realLetters[j]] += 1

if __name__ == "__main__":
    # validImages = [i[:-1] for i in open("validimages_newdataset.txt", 'r').readlines()]
    # random.shuffle(validImages)
    # for fileName in validImages:
    #     print(f"Working on {fileName} ...")
    #     saveSeparateLetters(fileName, pathText, pathImg, pathToSave)

    for i in range(29):
        os.mkdir("CleanedLettersDataset/" + str(i))