import numpy as np
import cv2

def binarizeImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return thresh

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
    return rotated

def getLines(image):
    hist = cv2.reduce(image,1, cv2.REDUCE_AVG).reshape(-1)
    avgFilter = np.array([0.2, 0.2, 0.2])
    hist = np.convolve(hist, avgFilter, 'same')
    
    th = 0
    H,W = image.shape[:2]
    uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]
    # rotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    linesImages = []

    if len(uppers) == len(lowers):
        for i in range(len(uppers)):
            linesImages.append(image[uppers[i]:lowers[i], :])
            # cv2.line(rotated, (0,uppers[i]), (W, uppers[i]), (255,0,0), 1)
            # cv2.line(rotated, (0,lowers[i]), (W, lowers[i]), (0,255,0), 1)
    else:
        raise Exception("Uppers not equal Lowers in lines")

    return uppers, lowers, linesImages

def getWords(lineImage):
    hist = cv2.reduce(lineImage, 0, cv2.REDUCE_AVG).reshape(-1)
    avgFilter = np.array([0.2, 0.2, 0.2])
    hist = np.convolve(hist, avgFilter, 'same')
    th = 0
    H,W = lineImage.shape[:2]
    uppers = [y for y in range(W-1) if hist[y]<=th and hist[y+1]>th]
    lowers = [y for y in range(W-1) if hist[y]>th and hist[y+1]<=th]
    wordImages = []
    # colored = cv2.cvtColor(lineImage, cv2.COLOR_GRAY2BGR)

    if len(uppers) == len(lowers):
        for i in range(len(uppers) - 1, -1, -1):
            wordImages.append(lineImage[:, uppers[i]:lowers[i]])
            # cv2.line(colored, (uppers[i], 0), (uppers[i], H), (255,0,0), 1)
            # cv2.line(colored, (lowers[i], 0), (lowers[i], H), (0,255,0), 1)
    else:
        raise Exception("Uppers not equal Lowers in words")
        
    return uppers, lowers, wordImages

def preprocess(image):
    binarizedImage = binarizeImage(image)
    rotatedImage = correctSkew(binarizedImage)
    _, _, linesImages = getLines(rotatedImage)
    linesOfWords = []
    for img in linesImages:
        _, _, wordImages = getWords(img)
        linesOfWords.append(wordImages)
    return linesOfWords

if __name__ == "__main__":
    # import os
    # x = os.listdir("images")
    # for i in x:
    #     image = cv2.imread(f"images/{i}")
    #     preprocess(image)

    image = cv2.imread("images/capr2.png")
    linesOfWords = preprocess(image)
    for line in linesOfWords:
        for word in line:
            cv2.imshow("word", word)
            cv2.waitKey(0)