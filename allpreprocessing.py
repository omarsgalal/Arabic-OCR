from preprocessing import preprocess
from charSeg2 import validCutRegionsFinal
from checkwordsegmentations import getText
import cv2
import os

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
    for i, line in enumerate(linesOfWords):
        for j, word in enumerate(line):
            numRegions, wordBeforeFilter, wordColor = validCutRegionsFinal(linesImages[i], word)
            segmentedChars.append(wordColor)

    return segmentedChars

def printSegmentedWordChars(fileName, pathText, pathImg):
    image = cv2.imread(pathImg + fileName)
    segmentedChars = img2Chars(image)
    wordsText = getText(pathText + fileName[:-4] + ".txt")

    # if numWords != len(wordsText):
    if not os.path.exists(fileName[:-4]):
        os.mkdir(fileName[:-4])
    for i, word in enumerate(segmentedChars):
        cv2.imwrite(f"{fileName[:-4]}/{i}.png", word)
    file = open(f"{fileName[:-4]}/words.txt", 'w')
    for i in wordsText:
        file.write(f"{i}\n")
    file.close()


if __name__ == "__main__":
    pathText = "Dataset/text/"
    pathImg = "Dataset/scanned/"
    fileName = "cmar1437.png"

    printSegmentedWordChars(fileName, pathText, pathImg)