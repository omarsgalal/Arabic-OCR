from preprocessing import preprocess
from charSeg2 import validCutRegionsFinal
from checkwordsegmentations import getText
import cv2
import os
from tqdm import tqdm

'''
dataset/text/cmar1437.txt
dataset/text/cmar1340.txt
dataset/text/cdec546.txt
dataset/text/cmar1339.txt
dataset/text/cfeb370.txt
dataset/text/cnov723.txt
dataset/text/capr1336.txt
dataset/text/capr114.txt
dataset/text/cjun185.txt
'''

def img2Chars(img):
    linesOfWords, numWords, linesImages = preprocess(img)
    segmentedChars = []
    print("Segmenting image ...")
    for i in tqdm(range(len(linesOfWords))):
        for j, word in enumerate(linesOfWords[i]):
            numRegions, wordBeforeFilter, wordColor, listOfseperateChars = validCutRegionsFinal(linesImages[i], word)
            segmentedChars.append([wordColor, numRegions+1, listOfseperateChars])

    return segmentedChars

def printSegmentedWordChars(fileName, pathText, pathImg):
    image = cv2.imread(pathImg + fileName)
    segmentedChars = img2Chars(image)
    wordsText = getText(pathText + fileName[:-4] + ".txt")

    # if numWords != len(wordsText):
    if not os.path.exists(fileName[:-4]):
        os.mkdir(fileName[:-4])
    if not os.path.exists(fileName[:-4] + "/correct"):
        os.mkdir(fileName[:-4] + "/correct")
    if not os.path.exists(fileName[:-4] + "/false"):
        os.mkdir(fileName[:-4] + "/false")
    for i, word in enumerate(segmentedChars):
        wordLength = len(wordsText[i])
        if "لا" in wordsText[i]:
            wordLength -= 1
        if word[1] == wordLength:
            cv2.imwrite(f"{fileName[:-4]}/correct/{i}.png", word[0])
        else:
            cv2.imwrite(f"{fileName[:-4]}/false/{i}.png", word[0])
    file = open(f"{fileName[:-4]}/words.txt", 'w')
    for j, i in enumerate(wordsText):
        file.write(f"{i}\n")
    file.close()


if __name__ == "__main__":
    pathText = "NewDataset/text/"
    pathImg = "NewDataset/scanned/"
    fileName = "capr1.png"

    printSegmentedWordChars(fileName, pathText, pathImg)