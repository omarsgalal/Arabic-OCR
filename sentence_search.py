import os 
from glob import glob
from tqdm import tqdm

sentenc = 'كبيرة من أجل التعاون'
paths = glob(os.path.join('dataset','text','*.txt'))
for path in tqdm(paths):
    with open(path) as fin:
        line = fin.readline()
        if line.find(sentenc) != -1:
            print(path)
