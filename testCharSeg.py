import os
import codecs
from charSeg2 import validCutRegions 
from preprocessing import preprocess
import cv2

def getText(fileName):
    f = codecs.open(fileName, encoding='utf-8')
    i = 0
    for line in f:
        words = line.split(' ')
        if words[0] == '':
            words = words[1:]
        if words[-1] == "\r\n":
            words = words[:-1]
        i += 1
    assert i == 1
    newWords = []
    i = 0
    while i < len(words):
        if len(words[i]) > 1:
            newWords.append(words[i])
            i += 1
        else:
            subword = ""
            while i < len(words) - 1 and len(words[i]) <= 1:
                subword += words[i]
                i += 1
            subword += words[i]
            newWords.append(subword)
            i += 1
            
    return newWords

def se7rText(textFileName):
    f_input = open("text\\" +textFileName + ".txt", 'r', encoding='utf-8')
    arr = []
    for line in f_input:
        line = line.rstrip()
        wordList = line.split(' ')  
        for word in wordList:
            arr.append(len(word))  
    f_input.close()
    return arr

def se7rImage(textFileName):
    #textArr = se7rText(textFileName)
    linesOfWords, numWords, linesImages = preprocess(image)
    #errorsList = []
    #allFiles = os.listdir(path="output\\")
    lineImage = ""
    errorSum = 0
    total = 0
    #i = 1
    # cumSum = 0
    # last = "0"
    #maxAdded = 0
    for line in linesOfWords:
        i = 0
        for f in line:
            # if('.png' not in f):
            #     continue
            # if len(f) <= 5:
            #     lineImage = f
            #     if len(last) != 1:
            #         added = int(last[2:3])
            #         if len(last) > 7:
            #             added = int(last[2:4])
            #         cumSum += maxAdded + 1
            #         maxAdded = 0
            #     last = f
            #     continue
            # last = f
            # if('8_' in f):
            #     x = 10
            nVR = validCutRegions(lineImage, f)
            # added = int(last[2:3])
            # if len(last) > 7:
            #     added = int(last[2:4])
            # maxAdded = max(maxAdded, added)
            # index = cumSum + added
            lenCompared = len(textArr[index])
            if('لا' in textArr[index]):
                lenCompared -= 1
            if nVR + 1 != lenCompared:
                print(f)
                errorsList.append(abs(nVR + 1 - lenCompared))
                errorSum += abs(nVR + 1 - lenCompared)
                total += lenCompared

    return errorSum, total

# errorSum = 0
# totalSum = 0
# allFiles = os.listdir(path="../text/")
# for f in allFiles:
#     f = f[:-4]
#     error, total = se7rImage(f)
#     print(error, total)
#     errorSum += error
#     totalSum += total 
#-4

#f_output = open('output.txt', 'w')