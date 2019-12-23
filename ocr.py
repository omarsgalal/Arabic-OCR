
from allpreprocessing import img2Chars
from check_model import loadModel
from tqdm import tqdm
from prepare_data_to_classifier import prepareCharImg
from chars_frequencies import chars_decode
import cv2
from Levenshtein import distance as levenshtein_distance
import editdistance
from helper_functions import loadWordsList
from checkwordsegmentations import getText
import json
import numpy as np
import copy
from time import time
import multiprocessing
from glob import glob
import os
import random
import sys
from checkwordsegmentations import getText, writeText


charaterModels = ''
wordModels = ''

MODEL_NAME = "SVMnew"
# loading the model
model = loadModel(MODEL_NAME)

def getNearestWord(word, wordsList):
    bestWord = word
    distance = 1000000000
    for w in wordsList:
        d = levenshtein_distance(word, w)
        if d < distance:
            distance = d
            bestWord = w
    return bestWord




# i = 2
# # def 
# imgName, txtPath , outfile = f"dataset/scanned/capr{i}.png", f"dataset/text/capr{i}.txt", f"dataset/results/capr{i}.txt"
def img2txt(imgName, txtPath , outfile):
    # read image
    img = cv2.imread(imgName)
    # import pdb; pdb.set_trace()
    # segment character from image
    print("Segmenting image to characters ...")
    segmentedChars = img2Chars(img)

    # determine spaces
    spaces = [len(l[2]) for l in segmentedChars]

    # list of character images only
    segmentedChars = [c for l in segmentedChars for c in l[2]]

    # compute features of chars
    print("\nExtracting features for characters ...")
    features = [prepareCharImg(c) for c in segmentedChars]



    # predict characters
    print("\nPredicting characters ...")
    # predictions_prop = model.predict_proba(features)
    # predictions = predictions_prop.argsort(axis=1)[:,-3:][:,::-1]
    # predictions = np.argmax(predictions_prop, axis=1)
    predictions = model.predict(features)
    # adding spaces

    wordsEncodes = []
    for space in spaces: wordsEncodes.append(predictions[:space]); predictions = predictions[space:]

    # decoding characters
    textList = []
    for wordEncoding in wordsEncodes: textList.append(''.join([chars_decode[charCode] for charCode in wordEncoding]))
    finalText = ' '.join(textList)

    # for wordEncoding in wordsEncodes: textList.append([[chars_decode[charCode] for charCode in charsCode] for charsCode in wordEncoding])
    # # import pdb; pdb.set_trace()
    # finalText = ' '.join([ ''.join([ c[0] for c in w ])  for w in textList])

    # post processing heeeeere

    print("\npost processing ...")
    # withNGram = postProcessing(textList)
    # withNGram = postprocessing_v2(finalText)
    withNGram = finalText
    # wordsList = loadWordsList()
    # text = [getNearestWord(text[i], wordsList) for i in tqdm(range(len(text)))]
    # text = ' '.join(text)

    # writing text
    writeText(outfile, finalText)
    # writeText(outfile.split('.')[0] + '-Ngram.txt', withNGram)

    originalTxt = getText(txtPath)
    # finalText = originalTxt[:4] + finalText[4:]
    error = editdistance.eval(finalText, originalTxt) / len(originalTxt)
    errorNgram = editdistance.eval(withNGram, originalTxt) / len(originalTxt)
    print('acc: ', 1-error, ' acc n gram: ', 1-errorNgram)
    return 1-error, 1-errorNgram



def postprocessing(textList):
    # take a list of best 3 characters predicted from the classifier
    # it make the total accuarcy worse by 3% 
    
    global charaterModels
    if charaterModels == '':
        with open('dataset/ngram_char2.json') as fin:
            charaterModels = json.load(fin)

    txt = copy.deepcopy(textList)
    newTxt = []
    for t in txt: 
        newTxt.extend(t)
        newTxt += [['  ', ' ', ' ']]
    txt = newTxt

    n = 4
    lastTxt = ''.join([t[0] for t in txt[:n]])
    newTxt = lastTxt[:-1] + ' '
    i = n
    while(i < len(txt)):
        lastTxt = newTxt[-n:]
        # print(lastTxt)
        chars = txt[i]
        bestChoice = ''
        # print('last txt ' ,lastTxt, ' next chat ', char)
        if lastTxt in charaterModels[n]:
            sor = sorted(charaterModels[n][lastTxt], key=charaterModels[n][lastTxt].get, reverse=True)[:15]
            if 'null' in sor: sor.remove('null')
            highProb = -1
            for char in chars:
                # if char in sor and charaterModels[n][lastTxt][char] - highProb > 0.5: 
                if char in sor: 
                    bestChoice = char
                    break
                    # highProb = charaterModels[n][lastTxt][char]

            if bestChoice == '':
                pred = sor[0]
                if i+1 < len(txt) and pred == txt[i+1][0]:
                    nextChar = lastTxt[1:] + pred
                    nextSor = sorted(charaterModels[n][nextChar], key=charaterModels[n][nextChar].get, reverse=True)[:15]
                    if nextChar not in charaterModels[n] or pred not in nextSor:
                        
                        # print('remove char ', char)
                        pass
                else:
                    bestChoice = pred

                # print('change', char , ' to ', txt[i])
        else:
            bestChoice = chars[0]

        newTxt += bestChoice
        # if i == 10: break
        i += 1
    return newTxt


NEAR_PREDCTED_WORD = 0.3
def nearPredectedWord(preWord, word, wordModels):
    if preWord in wordModels[1]:
        nextWords = sorted(wordModels[1][preWord], key=wordModels[1][preWord].get, reverse=True)
        for w in nextWords:
            distance = editdistance.eval(word, w) / len(word)
            if distance < NEAR_PREDCTED_WORD:
                return w

    return ''


NEAR_WORD = 0.4
NEAR_WORD = 0.3
PREDICTIONS = {}
def nearWord(word, wordModels):
    # nextWords = sorted(wordModels[1], key=lambda x : len(wordModels[1][x]), reverse=True)
    if word in PREDICTIONS:
        return PREDICTIONS[word]

    for w in wordModels[1].keys():
        distance = editdistance.eval(word, w) / len(word)
        if distance < NEAR_WORD:
            PREDICTIONS[word] = w
            return w
    PREDICTIONS[word] = ''
    return ''


def selectWord(inp):
    global wordModels
    lastWord, word = inp
    if word in wordModels[1]: return word        
    predNearWord = nearPredectedWord(lastWord, word, wordModels)
    if predNearWord != '': return predNearWord
    # predNearWord = nearWord(word, wordModels)
    # if predNearWord != '': return predNearWord

    return word


def postprocessing_v2(predected):
    global wordModels
    if wordModels == '':
        print('loading')
        with open('NewDataset/ngram_word2.json') as fin:
            wordModels = json.load(fin)
        print('loaded')

    n = 4
    predected = predected.replace('  ', ' ')
    # txt = copy.deepcopy(predected)
    txt = predected
    # newTxt = copy.deepcopy(predected)
    words = txt.split(' ')
    # newwords = txt.split(' ')
    wordInd = 0
    # charInd = 0
    
    # pbar = tqdm(total=len(newTxt))  
    tic = time()

    

    # newWords = postprocessing_v2.p.map(selectWord, [(words[i-1], words[i]) for i in range(len(words))])
    # words = newWords

    for wordInd in tqdm(range(1, len(words))):
        if words[wordInd] in wordModels[1]: continue        
        predNearWord = nearPredectedWord(words[wordInd-1], words[wordInd], wordModels)
        if predNearWord != '': words[wordInd] = predNearWord; continue
    #     predNearWord = nearWord(words[wordInd], wordModels)
    #     if predNearWord != '': words[wordInd] = predNearWord; continue

    print('\n\npost takes: ', time()-tic)
    return ' '.join(words)

def load_txt(path):
    with open(path) as f:
        return f.read().strip()

def writeFile(path, txt):
    with open(path, 'w') as f:
        f.write(txt)


def img2txt2(imgName, textFile, outfile):
    # read image
    img = cv2.imread(imgName)

    # real text
    realText = ' '.join(getText(textFile))

    # segment character from image
    print("Segmenting image to characters ...")
    segmentedChars = img2Chars(img)

    # loading the model
    model = loadModel(MODEL_NAME)


    
    f = open(outfile, 'w')
    allText = ""
    for i in tqdm(range(len(segmentedChars))):
        currentWord = ""
        for c in segmentedChars[i][2]:
            prediction = model.predict([prepareCharImg(c)])
            char = chars_decode[prediction[0]]
            currentWord += char
            allText += char
        allText += ' '
        f.write(currentWord)
        f.write(' ')
    f.close()

    with open("hello.txt", 'w') as f:
        f.write(realText)
        f.write("\n\n\n\n")
        f.write(allText)
    
    print("\n\n accuracy:")
    print(len(allText), len(realText))
    print(levenshtein_distance(realText, allText) / len(realText))

    print("another accuracy")
    realList = realText.split(' ')
    allList = allText.split(' ')
    errors = 0
    for i in range(len(realList)):
        errors += levenshtein_distance(realList[i], allList[i])
    print(errors / len(realText))


if __name__ == "__main__":
    # img2txt2("dataset/scanned/capr1.png","dataset/text/capr1.txt", "omar2.txt")
    imagesLen = 1000
    # startInd = 1
    values = np.zeros((imagesLen,2))
    # with open('dataset/ngram_word2.json') as fin:
    #     wordModels = json.load(fin)
    # postprocessing_v2.p = multiprocessing.Pool(4)
    img2Chars.pool = multiprocessing.Pool(8)
    filesImages = glob(os.path.join('NewDataset', 'scanned', '*.png'))
    random.shuffle(filesImages)
    filesImages = filesImages[:imagesLen]
    timeFile = open('time.txt', 'w')

    filesImages = [os.path.join('NewDataset', 'scanned', 'caug1141.png')]
    for i, imgPath in tqdm(enumerate(filesImages)):
        imgName = os.path.split(imgPath)[-1]
        Name = ''.join(imgName.split('.')[:-1]) + '.txt'
        txtPath = os.path.join("NewDataset", 'text' ,Name)
        outPath = os.path.join("NewDataset", 'results' ,Name)

        # sys.stdout = nullFile
        tic = time()
        values[i] = img2txt(imgPath, txtPath, outPath)
        toc = time()
        # sys.stdout = sys.__stdout__

        print( toc - tic, file=timeFile)
        print(f'***************************************************{i}***************************************************')
    
    print('acc mean ngram = ', (values[:,1]).mean())
    print('acc mean normal = ', (values[:,0]).mean())
    print('acc mean differance ngram - normal = ', (values[:,1] - values[:,0]).mean())