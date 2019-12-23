from ocr import img2txt
from Levenshtein import distance as levenshtein_distance
import os
import random
from checkwordsegmentations import getText

def img2txtMultiFiles(imgsPath, imgNames, FilePath):
    for imgName in imgNames:
        print(imgsPath + imgName + ".png")
        img2txt(imgsPath + imgName + ".png", FilePath + imgName + ".txt")


def img2txtRandomFiles(numFiles=2000, imgsPath="NewDataset/scanned/", FilePath="OutputTextFiles/"):
    imgs = os.listdir(imgsPath)
    random.shuffle(imgs)
    workingImgs = [imgName[:-4] for imgName in imgs[:numFiles]]
    img2txtMultiFiles(imgsPath, workingImgs, FilePath)


def calculateErrorRate(groundTruthPath="NewDataset/text/", predictedPath="OutputTextFiles/", statsFile="CER.txt"):
    with open(statsFile, 'w') as f:
        files = os.listdir(predictedPath)
        totalError = 0
        for file in files:
            realText = ' '.join(getText(groundTruthPath + file))
            predictedText = ' '.join(getText(predictedPath + file))
            error = levenshtein_distance(realText, predictedText) / len(realText)
            totalError += error
            f.write(f"file: {file}\t\tCER: {error}\n")
        totalError /= len(files)
        f.write(f"Total CER: {totalError}")


if __name__ == "__main__":
    img2txtRandomFiles(500)
    #calculateErrorRate()