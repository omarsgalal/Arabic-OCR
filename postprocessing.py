from Levenshtein import distance as levenshtein_distance
from helper_functions import loadWordsList
from tqdm import tqdm
import os
from checkwordsegmentations import getText, writeText

def getNearestWord(word, wordsList):
    bestWord = word
    distance = 1000000000
    for w in wordsList:
        d = levenshtein_distance(word, w)
        if d < distance:
            distance = d
            bestWord = w
    return bestWord

def postProcessTextWithLev(text):
    wordsList = loadWordsList()
    words = text.split(' ')
    for i in tqdm(range(len(words))):
        words[i] = getNearestWord(words[i], wordsList)
    return ' '.join(words)

def postProcessFolder(srcFolder="OutputTextFiles/", dstFolder="LevTextFiles/"):
    files = os.listdir(srcFolder)
    for f in files:
        text = ' '.join(getText(srcFolder + f))
        processedText = postProcessTextWithLev(text)
        writeText(dstFolder + f, processedText)

if __name__ == "__main__":
    postProcessFolder()