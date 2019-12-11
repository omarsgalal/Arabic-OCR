import numpy as np
import cv2
from scipy.ndimage import interpolation as inter
import progressbar

def binarizeImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return thresh

def getBaseline(image):
    hist = np.sum(image, axis=1)
    # nonzeros = np.nonzero(hist)
    return np.argmax(hist)#, nonzeros[0][0], nonzeros[0][-1]

def correctSkew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle 
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # print (rotated)
    return rotated

def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def correctSkew2(image):
    delta = 1
    limit = 5
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(image, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]

    # correct skew
    data = inter.rotate(image, best_angle, reshape=False, order=0)
    return data

def getLines(image):
    hist = cv2.reduce(image,1, cv2.REDUCE_AVG).reshape(-1)
    avgFilter = np.array([0.25, 0.25, 0.25, 0.25])
    hist = np.convolve(hist, avgFilter, 'same')
    
    th = 0
    H,W = image.shape[:2]
    uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]
    # rotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    linesImages = []
    maxAbovebaseline = 0
    maxBelowbaseline = 0
    avgAbovebaseline = 0
    avgBelowbaseline = 0

    if len(uppers) == len(lowers):
        for i in range(len(uppers)):
            lineImageI = image[uppers[i]:lowers[i], :]
            baseline = getBaseline(lineImageI)
            lineImageI = lineImageI[max(baseline - 13, 0): min(baseline + 9, lineImageI.shape[0] + 1), :]
            if lineImageI.shape[0] < 8 + 13 + 1: 
                newLine = np.zeros(( 8 + 13 + 1, lineImageI.shape[1]))
                #to be modified
                newLine[1:lineImageI.shape[0] + 1, :] = lineImageI
                lineImageI = newLine
            assert lineImageI.shape[0] == 8 + 13 + 1
            linesImages.append(lineImageI)
            
            # baseline += uppers[i]
            # maxAbovebaseline = max(maxAbovebaseline, baseline - uppers[i])
            # maxBelowbaseline = max(maxBelowbaseline, lowers[i] - baseline)
            # avgAbovebaseline += baseline - uppers[i]
            # avgBelowbaseline += lowers[i] - baseline
            
            # maxAbovebaseline = max(maxAbovebaseline, baseline - up)
            # maxBelowbaseline = max(maxBelowbaseline, low - baseline)
            # avgAbovebaseline += baseline - up
            # avgBelowbaseline += low - baseline
            # cv2.line(rotated, (0,uppers[i]), (W, uppers[i]), (255,0,0), 1)
            # cv2.line(rotated, (0,lowers[i]), (W, lowers[i]), (0,255,0), 1)
    else:
        raise Exception("Uppers not equal Lowers in lines")

    stats = [0, 0, 0, 0]
    if len(uppers) != 0:
        avgAbovebaseline /= len(uppers)
        avgBelowbaseline /= len(uppers)
        stats = [maxAbovebaseline, maxBelowbaseline, avgAbovebaseline, avgBelowbaseline]

    return uppers, lowers, linesImages, stats

def getWords(lineImage):
    hist = cv2.reduce(lineImage, 0, cv2.REDUCE_AVG).reshape(-1)
    avgFilter = np.array([1/3, 1/3, 1/3])
    hist = np.convolve(hist, avgFilter, 'same')
    th = 1
    H,W = lineImage.shape[:2]
    uppers = [y for y in range(W-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y for y in range(W-1) if hist[y]>th and hist[y+1]<=th]
    wordImages = []
    # colored = cv2.cvtColor(lineImage, cv2.COLOR_GRAY2BGR)

    if len(uppers) == len(lowers):
        i = len(uppers) - 1
        j = len(lowers) - 1
        while i >= 0:
            smallWord = lineImage[:, uppers[i]:lowers[i]]
            if smallWord.shape[1] > 8 or i == 0:
                # if i == 0 and i == j and len(wordImages) > 0:
                #     wordImages[-1] = lineImage[:, uppers[i]:lowers[j + 1]]
                # else:
                wordImages.append(lineImage[:, uppers[i]:lowers[j]])
                i -= 1
                j = i
            else:
                i -= 1
            # cv2.line(colored, (uppers[i], 0), (uppers[i], H), (255,0,0), 1)
            # cv2.line(colored, (lowers[i], 0), (lowers[i], H), (0,255,0), 1)
    else:
        raise Exception("Uppers not equal Lowers in words")
        
    return uppers, lowers, wordImages

def preprocess(image):
    binarizedImage = binarizeImage(image)
    rotatedImage = correctSkew(binarizedImage)
    _, _, linesImages, _ = getLines(rotatedImage)
    linesOfWords = []
    numWords = 0
    for img in linesImages:
        _, _, wordImages = getWords(img)
        linesOfWords.append(wordImages)
        numWords += len(wordImages)
    return linesOfWords, numWords, linesImages

def getAvgMaxMinBaseline():
    import os
    x = os.listdir("Dataset/scanned/")[0:1000]
    f = open("stats.txt", 'w')
    f.write("\t\t\tmaxAbove\tmaxBelow\tavgAbove\tavgBelow\n")
    allstats = [0, 0, 0, 0]
    with progressbar.ProgressBar(max_value=len(x)) as bar:
        for j, i in enumerate(x):
            image = cv2.imread(f"Dataset/scanned/{i}")
            binarizedImage = binarizeImage(image)
            rotatedImage = correctSkew(binarizedImage)
            _, _, _, stats = getLines(rotatedImage)
            f.write(f"{i}\t{stats[0]}\t{stats[1]}\t{stats[2]}\t{stats[3]}\n")
            allstats[0] = max(allstats[0], stats[0])
            allstats[1] = max(allstats[1], stats[1])
            allstats[2] += stats[2]
            allstats[3] += stats[3]
            bar.update(j)
    allstats[2] /= len(x)
    allstats[3] /= len(x)
    f.write(f"\t\t\t{allstats[0]}\t{allstats[1]}\t{allstats[2]}\t{allstats[3]}")
    f.close()

if __name__ == "__main__":
    # import os
    # x = os.listdir("images")
    # for i in x:
    #     image = cv2.imread(f"images/{i}")
    #     preprocess(image)

    # import sys
    # np.set_printoptions(threshold=sys.maxsize)
    # image = cv2.imread("Dataset/scanned/capr100.png")
    # linesOfWords, numWords, linesImages = preprocess(image)
    # for j, line in enumerate(linesOfWords):
    #     for i, word in enumerate(line):
    #         # cv2.imshow("word", word)
    #         # cv2.waitKey(0)
    #         cv2.imwrite(f"output/{j}_{i}.png", word)

    getAvgMaxMinBaseline()