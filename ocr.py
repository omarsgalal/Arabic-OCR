from allpreprocessing import img2Chars
from check_model import loadModel
from tqdm import tqdm
from prepare_data_to_classifier import prepareCharImg
from chars_frequencies import chars_decode
import cv2
from Levenshtein import distance as levenshtein_distance
from helper_functions import loadWordsList
from checkwordsegmentations import getText

MODEL_NAME = "Neural Net"

def getNearestWord(word, wordsList):
    bestWord = word
    distance = 1000000000
    for w in wordsList:
        d = levenshtein_distance(word, w)
        if d < distance:
            distance = d
            bestWord = w
    return bestWord


def img2txt(imgName, outfile):
    # read image
    img = cv2.imread(imgName)

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

    # loading the model
    model = loadModel(MODEL_NAME)

    # predict characters
    print("\nPredicting characters ...")
    predictions = model.predict(features)

    # adding spaces
    wordsEncodes = []
    for space in spaces: wordsEncodes.append(predictions[:space]); predictions = predictions[space:]

    # decoding characters
    textList = []
    for wordEncoding in wordsEncodes: textList.append(''.join([chars_decode[charCode] for charCode in wordEncoding]))
    finalText = ' '.join(textList)

    # post processing heeeeere
    print("\npost processing ...")
    # wordsList = loadWordsList()
    # text = [getNearestWord(text[i], wordsList) for i in tqdm(range(len(text)))]
    # text = ' '.join(text)

    # writing text
    with open(outfile, 'w') as f:
        f.write(finalText)


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
    # img2txt2("NewDataset/scanned/capr1.png","NewDataset/text/capr27.txt", "omar2.txt")
    img2txt("NewDataset/scanned/capr27.png", "omar2.txt")