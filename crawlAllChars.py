import codecs
import os
from tqdm import tqdm

textPath = "NewDataset/text/"

charsDictFreq = {}

def getText(fileName):
    f = codecs.open(fileName, encoding='utf-8')
    f = ''.join(f)
    i = 0
    while i < len(f):
        if f[i] == 'ل' and i < len(f) - 1 and f[i+1] == 'ا':
            try:
                charsDictFreq['لا'] += 1
            except:
                charsDictFreq['لا'] = 0
            i += 2
        else:
            try:
                charsDictFreq[f[i]] += 1
            except:
                charsDictFreq[f[i]] = 0
            i += 1

textNames = os.listdir(textPath)[:]

for i in tqdm(range(len(textNames))):
    getText(textPath + textNames[i])

print(charsDictFreq)
    