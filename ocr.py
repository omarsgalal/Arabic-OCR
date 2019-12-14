from allpreprocessing import img2Chars
from check_model import loadModel
from tqdm import tqdm
from prepare_data_to_classifier import prepareCharImg
from chars_frequencies import chars_decode
import cv2
from Levenshtein import distance as levenshtein_distance
from helper_functions import loadWordsList

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
    spaces = [l[1] for l in segmentedChars]

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

    # text without spaces
    textList = ''.join([chars_decode[pred] for pred in predictions])

    # adding spaces
    text = []
    for space in spaces:
        text.append(textList[:space])
        textList = textList[space:]

    # post processing heeeeere
    print("\npost processing ...")
    wordsList = loadWordsList()
    text = [getNearestWord(text[i], wordsList) for i in tqdm(range(len(text)))]
    text = ' '.join(text)

    # writing text
    with open(outfile, 'w') as f:
        f.write(text)



if __name__ == "__main__":
    img2txt("NewDataset/scanned/capr27.png", "omar.txt")