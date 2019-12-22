from preprocessing import preprocess
import codecs
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

pathText = "NewDataset/text/"
pathImg = "NewDataset/scanned/"
fileName = "capr1277.png"

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
    return words

def word2Chars(word):
    i = 0
    chars = []
    while i < len(word):
        if word[i] == 'ل' and i < len(word) - 1 and word[i+1] == 'ا':
            chars.append(word[i:i+2])
            i += 2
        else:
            chars.append(word[i])
            i += 1
    return chars

def checkImgwithText(fileName, pathText, pathImg):
    image = cv2.imread(pathImg + fileName)
    linesOfWords, numWords, linesImages = preprocess(image)
    wordsText = getText(pathText + fileName[:-4] + ".txt")

    if numWords != len(wordsText):
        if not os.path.exists(fileName[:-4]):
            os.mkdir(fileName[:-4])
        if not os.path.exists(f"{fileName[:-4]}/smalls"):
            os.mkdir(f"{fileName[:-4]}/smalls")
        if not os.path.exists(f"{fileName[:-4]}/smalls/lines"):
            os.mkdir(f"{fileName[:-4]}/smalls/lines")
        for j, line in enumerate(linesOfWords):
            for i, word in enumerate(line):
                cv2.imwrite(f"{fileName[:-4]}/{j}_{i}.png", word)
                if word.shape[1] < 10:
                    cv2.imwrite(f"{fileName[:-4]}/smalls/{j}_{i}.png", word)
                    cv2.imwrite(f"{fileName[:-4]}/smalls/lines/{j}.png", linesImages[j])
        file = open(f"{fileName[:-4]}/words.txt", 'w')
        file2 = open(f"{fileName[:-4]}/smalls.txt", 'w')
        for i in wordsText:
            file.write(f"{i}\n")
            if len(i) <= 1:
                file2.write(f"{i}\n")
        file.close()
        file2.close()
    return numWords, len(wordsText)

# print(checkImgwithText(fileName, pathText, pathImg))

# import os
# imagesNames = os.listdir(pathImg)[:]
# errorsDict = {}
# maxError = 0
# minError = 0
# numCorrect = 0
# with progressbar.ProgressBar(max_value=len(imagesNames)) as bar:
#     for i, imgName in enumerate(imagesNames):
#         wordsImg, wordsTxt = checkImgwithText(imgName, pathText, pathImg)
#         if wordsImg != wordsTxt:
#             print(imgName, "\t", wordsImg,"\t", wordsTxt)
#         errorsDict[imgName] = wordsImg - wordsTxt
#         numCorrect += int(errorsDict[imgName] == 0)
#         maxError = max(maxError, errorsDict[imgName])
#         minError = min(minError, errorsDict[imgName])
#         bar.update(i)

# f = open("errorsdict_all_newdataset.txt", 'w')
# f.write(f"total correct: {numCorrect}\n")
# f.write(f"max Error: {maxError}\n")
# f.write(f"min Error: {minError}\n")
# for key, value in errorsDict.items():
#     f.write(f"{key}    {value}\n")
# f.close()