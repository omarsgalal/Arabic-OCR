# -*- coding: utf-8 -*-

import numpy as np
import cv2
from scipy import stats
# from skeletone import zhangSuen

class SeparationRegions:
    def __init__(self):
        self.startIndex = 0
        self.startIndex = 0
        self.cutIndex = 0

def detectBaselineIndex(img):
    
    HP = np.sum(img, axis=1)
    index = np.argmax(HP)
    #print(HP[index])
    return index
    #start_point = (0, index)
    #end_point = (img.shape[1], index)
    #thickness = 1
    #color = (255,0,0)
    #output = cv2.line(img, start_point, end_point, color, thickness)
    #return output

#m7tag yt3ad 3la el code 
def maxTransitionIndex(img, baselineIndex):
    maxTrans = 0
    maxTransIndex = baselineIndex
    for i in range(baselineIndex - 1, -1, -1):
        currentTransition = 0
        flag = 0
        for j in range(img.shape[1] - 1, -1, -1):
            if(img[i,j] == 255 and flag == 0):
                currentTransition += 1
                flag = 1
            elif(img[i,j] != 255 and flag == 1):
                flag = 0
        if(currentTransition >= maxTrans):
            maxTrans = currentTransition
            maxTransIndex = i
        if(baselineIndex - i > 6):
            break
    if(baselineIndex - maxTransIndex <= 2):
        maxTransIndex = baselineIndex + 3
    return maxTransIndex

#  -------------------->
#  start-----mid-----end
def vp0NearstToMid(VP, start, mid, end):
    zeroIndex = np.where(VP == 0)[0]
    zeroIndex = zeroIndex[(zeroIndex > start) & (zeroIndex < end)]
    Index = np.abs(zeroIndex - mid)
    if(len(Index) > 0):
        return zeroIndex[np.argmin(Index)]
    return -1

'''def vpMFVbetMidEnd(VP, start, mid, end, MFV):
    index = np.where(VP <= MFV)[0]
    index = index[(index > start) & (index < end)]
    dummy = np.abs(index - (mid - 1))
    #dummy = np.abs(index - 0)
    if(len(dummy) > 0):
        return index[np.argmin(dummy)]
    return -1

def vpMFVbetMidStart(VP, start, mid, end, MFV):
    index = np.where(VP <= MFV)[0]
    index = index[(index <= mid) & (index > start)]
    dummy = np.abs(index - (mid - 1))
    if(len(dummy) > 0):
        return index[np.argmin(dummy)]
    return -1'''

def vpbetStartEnd(VP, start, mid, end, MFV):
    smallestIndex = np.argmin(VP[start+1:end])
    smallestArr = np.where(VP[start+1:end] == VP[start+1+smallestIndex])[0]
    smallestArr += (start+1)
    dummy = np.abs(smallestArr - start)
    if len(dummy) > 1:
        dummy = np.abs(smallestArr - (start-1))
    #dummy = np.abs(index - 0)
    if(len(dummy) > 0):
        return smallestArr[np.argmin(dummy)]
    return -1

def cutPoints(line, word, maxTransitionIndex):
    flag = 0
    lastpixel = 255
    VP = np.sum(word, axis=0)
    MFV = stats.mode(VP)[0][0]
    separationRegions = []
    for i in range(word.shape[1]-1, -1, -1):
        if(word[maxTransitionIndex, i] == 255 and lastpixel == 255):
            flag = 1
            lastpixel = word[maxTransitionIndex, i]
            continue
        elif(word[maxTransitionIndex, i] == 0 and lastpixel == 255 and flag):
            SR = SeparationRegions()
            SR.endIndex = i + 1
            #flag = 1
        elif(word[maxTransitionIndex, i] == 255 and lastpixel == 0):
            if(not flag):
                flag = 1
                lastpixel = word[maxTransitionIndex, i]
                continue
            SR.startIndex = i
            midIndex = int((SR.endIndex + SR.startIndex)/2)
            
            if(vp0NearstToMid(VP, SR.startIndex, midIndex, SR.endIndex) != -1):
                SR.cutIndex = vp0NearstToMid(VP, SR.startIndex, midIndex, SR.endIndex)
            elif(vpbetStartEnd(VP, SR.startIndex, midIndex, SR.endIndex, MFV) != -1):
                SR.cutIndex = vpbetStartEnd(VP, SR.startIndex, midIndex, SR.endIndex, MFV)
            else:
                SR.cutIndex = midIndex
            '''elif(VP[midIndex] == MFV):
                SR.cutIndex = midIndex'''
            '''elif(vpMFVbetMidEnd(VP, SR.startIndex, midIndex, SR.endIndex, MFV) != -1):
                SR.cutIndex = vpMFVbetMidEnd(VP, SR.startIndex, midIndex, SR.endIndex, MFV)'''
            separationRegions.append(SR)
            #print(SR.endIndex, SR.cutIndex, SR.startIndex)
        lastpixel = word[maxTransitionIndex, i]
    return separationRegions

def connectComp(a):
    b = a.copy()
    b[b == 255] = 1 
    '''if(b.shape[0] > 16):
        b[16,0] = 1'''
    n, comp = cv2.connectedComponents(b, connectivity=4)
    #print(comp)
    #print(n)
    return n, comp

#SEGP: the segment between previous cut index and next cut index
start = (0,0)
def isSEGholed(segment, MTI):
    for i in range(segment.shape[1]):
        if(segment[MTI,i] == 255):
            global start
            start = (MTI,i)
            #print(MTI,i)
            return checkCircle(segment, i, MTI, {(0,0)}, None)
    return False

def checkCircle(circle, i, j, pixelSet, direction):
    if(j >= circle.shape[0] or i >= circle.shape[1] or j < 0 or i < 0):
        return False
    elif(circle[j,i] == 255):
        if((j,i) not in pixelSet):
            pixelSet.add((j,i))
            #x = circle.copy()
            #x[j, i] = 127
            #cv2.imshow("img", x)
            #cv2.waitKey(0)
            if(direction == None or direction ==  'up'):
                return checkCircle(circle, i, j - 1, pixelSet, "up") or checkCircle(circle, i + 1, j, pixelSet, "right") or checkCircle(circle, i - 1, j, pixelSet, "left")
            elif(direction ==  'down'):
                return checkCircle(circle, i, j + 1, pixelSet, "down") or checkCircle(circle, i + 1, j, pixelSet, "right") or checkCircle(circle, i - 1, j, pixelSet, "left")
            elif(direction ==  'left'):
                return checkCircle(circle, i, j - 1, pixelSet, "up") or checkCircle(circle, i, j + 1, pixelSet, "down") or checkCircle(circle, i - 1, j, pixelSet, "left")
            elif(direction ==  'right'):
                return checkCircle(circle, i, j - 1, pixelSet, "up") or checkCircle(circle, i + 1, j, pixelSet, "right") or checkCircle(circle, i, j + 1, pixelSet, "down")
        elif(start == (j,i) and len(pixelSet) > 8):
            #for tup in pixelSet:
                #print(tup[0], tup[1])
                #circle[tup[0], tup[1]] = 127
                
            #cv2.imshow("img", circle)
            #cv2.waitKey(0)
            return True
        else:
            return False
    return False

def isStroke(MFV, segment, MTI, baselineIndex):
    SHPA = np.sum(np.sum(segment[:baselineIndex+1,:], axis=1))
    SHPB = np.sum(np.sum(segment[baselineIndex+1:,:], axis=1))
    HP = np.sum(segment, axis=1)
    mode = stats.mode(HP)[0][0]

    baselineVP = np.sum(segment[baselineIndex-1:baselineIndex+3, :], axis=0)
    baseline = True
    if(0 in baselineVP):
        baseline = False

    #return (SHPA > SHPB) and (mode == MFV) and (not isSEGholed(segment, MTI))
    returnValue = (SHPA > SHPB) and connectComp(255 - segment[baselineIndex - 15:baselineIndex,:])[0] <= 2
    Alef = True
    for i in range(MTI, 0, -1):
        alefRowIndex = np.where(segment[i,:] == 255)[0]
        if(len(alefRowIndex) == 0):
            if(MTI - i < 5*3):
                Alef = False
                break  
    return returnValue and not Alef and baseline# and (mode == MFV) 


def isDotted(segment, MTI):
    dotted = False
    flag = 0
    for i in range(MTI, segment.shape[0]):
        if(255 in segment[i,:]):
            if flag:
                dotted = True
        else:
            flag = 1
    flag2 = 0
    for i in range(MTI-1, -1, -1):
        if(255 in segment[i,:]):
            if flag2:
                dotted = True
        else:
            flag2 = 1
    return dotted

def countTransitions(arr, index):
    transitions = 0
    for i in range(index + 1, len(arr)):
        if(arr[i] == 255):
            transitions += 1
            break
    for i in range(index - 1, -1, -1):
        if(arr[i] == 255):
            transitions += 1
            break
    if(transitions == 2):
        return True
    return False

def countUpTransitions(arr, index):
    down = 0
    up = 0
    
    for i in range(index - 1, -1, -1):
        if(arr[i] == 255):
            up = 1 
            break

    for i in range(index + 1, len(arr)):
        if(arr[i] == 255):
            down = 1
            break

    return up, down    

def sumLeftVsRight(left, right):
    leftSum = np.sum(np.sum(left))
    rightSum = np.sum(np.sum(right))
    if(leftSum > rightSum):
        return True
    return False

def specialGEEMcase(word, MTI, SR):
    up    = (word[MTI-1, SR.cutIndex  ] == 255) or (word[MTI-2, SR.cutIndex  ] == 255)
    left  = (word[MTI  , SR.cutIndex-1] == 255) or (word[MTI  , SR.cutIndex-2] == 255)
    right = (word[MTI  , SR.cutIndex+1] == 255) or (word[MTI  , SR.cutIndex+2] == 255) 
    return up and left and right

def separationRegionFilter(line, word, sepRegionList, BaselineIndex, MTI):
    VP = np.sum(word, axis=0)
    MFV = stats.mode(VP)[0][0]
    validSepRegion = []
    i = 0
    while i < len(sepRegionList):       
        if(i == 6):
            x = 250
            y = 1
        SR = sepRegionList[i]
        #baseline = np.sum(word[BaselineIndex,SR.startIndex+1:SR.endIndex]) >= 255 * (SR.endIndex - SR.startIndex - 2)
        #baseline |= np.sum(word[BaselineIndex+1,SR.startIndex+1:SR.endIndex]) >= 255 * (SR.endIndex - SR.startIndex - 2)
        baselineVP = np.sum(word[BaselineIndex-1*3:BaselineIndex+2*3, SR.startIndex+1:SR.endIndex], axis=0)
        baseline = True
        if(0 in baselineVP):
            baseline = False
        noBaseline = not baseline
        _, comp = connectComp(word[:,SR.startIndex-1:SR.endIndex+2*4])
        #7eta fadya
        if(specialGEEMcase(word, MTI, SR)):
            i += 1
        elif(countTransitions(word[:,SR.cutIndex], MTI) and np.sum(word[:MTI,:], axis=0)[SR.cutIndex] > 2*255):
            i += 1
        elif(VP[SR.cutIndex] == 0):
            validSepRegion.append(SR)
            i += 1
        #elif(connectComp(word[:,SR.startIndex:SR.endIndex]) != 1):
        # elif(i+1 == len(sepRegionList) and comp[MTI,1] == comp[MTI, -4]):
        #     i += 1
        # last letter
        elif(i + 1 == len(sepRegionList) and baseline):# or sepRegionList[i+1].cutIndex == 0): and height(word[:,.cutIndex:SR.cutIndex], MTI) < 6):
            # ة ر 
            if (sumLeftVsRight(word[:,:SR.cutIndex], word[:,SR.cutIndex:SR.endIndex+1])):
                validSepRegion.append(SR)
                i += 1
            # ف ق ت ب ث
            else:
                i += 1
        #R
        elif(VP[SR.cutIndex] != 0 and comp[BaselineIndex,1] != comp[BaselineIndex, -4]):
            up, down = countUpTransitions(word[:,SR.cutIndex], MTI)
            if up and not down and 2*255 < VP[SR.cutIndex] and VP[SR.cutIndex] < 6*3*255:
                pass
            else:
                validSepRegion.append(SR)
            i += 1
        elif(noBaseline):
            SHPB = np.sum(np.sum(word[BaselineIndex:,SR.startIndex:SR.endIndex + 1], axis=1))
            SHPA = np.sum(np.sum(word[:BaselineIndex,SR.startIndex:SR.endIndex + 1], axis=1))
            if(SHPB > SHPA):
                i += 1
            elif(VP[SR.cutIndex] < MFV or VP[SR.cutIndex] > MFV):
                validSepRegion.append(SR)
                i += 1
            else:
                i += 1
        #elif(i-1 >= 0 and i+1 < len(sepRegionList) and isSEGholed(word[:,sepRegionList[i+1].cutIndex:sepRegionList[i-1].cutIndex], MTI)):
        #hole
        elif(connectComp(255 - word[:BaselineIndex,SR.startIndex:SR.endIndex+1])[0] >= 3 and countTransitions(word[:,SR.cutIndex], MTI)):
            #connectComp(word[:baselineIndex,sepRegionList[i+1].cutIndex:sepRegionList[i-1].cutIndex])
            i += 1
        
        

        
        elif(not isStroke(MFV, word[:,sepRegionList[i+1].cutIndex:SR.cutIndex + 1], MTI, BaselineIndex)):
            '''noNextBaseline = not bool(np.sum(word[BaselineIndex,sepRegionList[i+1].startIndex:sepRegionList[i+1].endIndex]))
            if(noNextBaseline and sepRegionList[i+1].cutIndex <= MFV):
                i += 1
            else:'''
            validSepRegion.append(SR)
            i += 1
        elif(isStroke(MFV, word[:,sepRegionList[i+1].cutIndex:SR.cutIndex + 1], MTI, BaselineIndex) and not isDotted(word[:,sepRegionList[i+1].cutIndex:SR.cutIndex + 1], MTI)):
            if(i+2 < len(sepRegionList) and isStroke(MFV, word[:,sepRegionList[i+2].cutIndex:sepRegionList[i+1].cutIndex + 1], MTI, BaselineIndex) and not isDotted(word[:,sepRegionList[i+2].cutIndex:sepRegionList[i+1].cutIndex], MTI)):
                if(i+3 < len(sepRegionList) and isStroke(MFV, word[:,sepRegionList[i+3].cutIndex:sepRegionList[i+2].cutIndex + 1], MTI, BaselineIndex) and not isDotted(word[:,sepRegionList[i+3].cutIndex:sepRegionList[i+2].cutIndex], MTI)):
                    # ال.سلطان
                    validSepRegion.append(SR)
                    validSepRegion.append(sepRegionList[i+3])
                    i += 4
                else:
                    # ا.سم
                    if(i+3 == len(sepRegionList) and noBaseline):
                        pass
                    elif (i + 2 < len(sepRegionList)):
                        validSepRegion.append(sepRegionList[i+2])#error hna m3 qabos
                    i += 3
            #SHEEN 
            elif(i+2 < len(sepRegionList) and isStroke(MFV, word[:,sepRegionList[i+2].cutIndex:sepRegionList[i+1].cutIndex + 1], MTI, BaselineIndex) and isDotted(word[:,sepRegionList[i+2].cutIndex:sepRegionList[i+1].cutIndex], MTI)):
                if(i+3 < len(sepRegionList) and isStroke(MFV, word[:,sepRegionList[i+3].cutIndex:sepRegionList[i+2].cutIndex + 1], MTI, BaselineIndex) and not isDotted(word[:,sepRegionList[i+3].cutIndex:sepRegionList[i+2].cutIndex], MTI)):
                    # ال.سلطان
                    validSepRegion.append(SR)
                    validSepRegion.append(sepRegionList[i+3])
                    i += 4
                else:
                    # ا.سم
                    if(i+3 == len(sepRegionList) and noBaseline):
                        pass
                    elif (i + 2 < len(sepRegionList)):
                        validSepRegion.append(sepRegionList[i+2])
                    i += 3
                #validSepRegion.append(sepRegionList[i+2])
                #i += 3
            #stroke of SAAD
            elif(i+2 < len(sepRegionList) and (not isStroke(MFV, word[:,sepRegionList[i+2].cutIndex:sepRegionList[i+1].cutIndex + 1], MTI, BaselineIndex) or (isStroke(MFV, word[:,sepRegionList[i+2].cutIndex:sepRegionList[i+1].cutIndex], MTI, BaselineIndex) and isDotted(word[:,sepRegionList[i+2].cutIndex:sepRegionList[i+1].cutIndex], MTI)))):
                baselineVP2 = np.sum(word[BaselineIndex-1:BaselineIndex+2, sepRegionList[i+2].startIndex+1:sepRegionList[i+2].endIndex], axis=0)
                baseline2 = True
                if(0 in baselineVP2):
                    baseline2 = False
                if i + 3 == len(sepRegionList) and not baseline2:
                    i += 3 
                else:
                    i += 1
            else:
                i += 1
        elif(isStroke(MFV, word[:,sepRegionList[i+1].cutIndex:SR.cutIndex + 1], MTI, BaselineIndex) and isDotted(word[:,sepRegionList[i+1].cutIndex:SR.cutIndex + 1], MTI)):#sepRegionList[i+1].cutIndex:SR.cutIndex
            if(i+2 < len(sepRegionList) and isStroke(MFV, word[:,sepRegionList[i+2].cutIndex:sepRegionList[i+1].cutIndex + 1], MTI, BaselineIndex) and not isDotted(word[:,sepRegionList[i+2].cutIndex:sepRegionList[i+1].cutIndex], MTI)):
                #fy moshkla hna
                #if(i+3 < len(sepRegionList) and ((isStroke(MFV, word[:,sepRegionList[i+3].cutIndex:sepRegionList[i+2].cutIndex + 1], MTI, BaselineIndex) and not isDotted(word[:,sepRegionList[i+3].cutIndex:sepRegionList[i+2].cutIndex], MTI)) or not isStroke(MFV, word[:,sepRegionList[i+3].cutIndex:sepRegionList[i+2].cutIndex + 1], MTI, BaselineIndex))):
                if(i+3 < len(sepRegionList) and not isStroke(MFV, word[:,sepRegionList[i+3].cutIndex:sepRegionList[i+2].cutIndex + 1], MTI, BaselineIndex)):    
                    validSepRegion.append(sepRegionList[i+2])
                    i += 3
                else:
                    validSepRegion.append(SR)
                    i += 1
            else:
                validSepRegion.append(SR)
                i += 1
    return validSepRegion

        


'''im = cv2.imread('1.png')

thresValue = 127

imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,thresValue,255,0)'''
#cv2.imshow("thresholded", thresh)
#cv2.waitKey(0)

'''wordOrg = cv2.imread('0_0.png')
wordGray = cv2.cvtColor(wordOrg, cv2.COLOR_BGR2GRAY)
ret, wordThresh = cv2.threshold(wordGray, thresValue,255,0)
wordColor = cv2.cvtColor(wordThresh, cv2.COLOR_GRAY2BGR)

baselineIndex = detectBaselineIndex(wordThresh)
maxTransIndex = maxTransitionIndex(wordThresh, baselineIndex)

img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)'''

#start_point = (0, baselineIndex)
#end_point = (img.shape[1], baselineIndex)
#thickness = 1
#color = (255,0,0)

#cv2.imwrite("thresholded.png", img)

#output = cv2.line(img, start_point, end_point, color, thickness)

#cv2.imwrite("baseline.png", output)
#cv2.imshow("baseline", output)
#cv2.waitKey(0)

#start_point = (0, maxTransIndex)
#end_point = (img.shape[1], maxTransIndex)
#color = (0,255,0)
#output = cv2.line(output, start_point, end_point, color, thickness)

#print(baselineIndex, maxTransIndex)

#cv2.imwrite("maxTrans.png", output)
#cv2.imshow("maxTrans", output)
#cv2.waitKey(0)


'''sepRegions = cutPoints(thresh, wordThresh, maxTransitionIndex)
for sr in sepRegions:
    wordColor[maxTransIndex, sr.startIndex] = np.array([0,0,255])
    wordColor[maxTransIndex, sr.endIndex]   = np.array([0,0,255])
    wordColor[maxTransIndex, sr.cutIndex]   = np.array([0,255,0])

cv2.imwrite("cutPoints.png", wordColor)

vsr = separationRegionFilter(thresh, wordThresh, sepRegions, baselineIndex, maxTransIndex)
wordColor = cv2.cvtColor(wordThresh, cv2.COLOR_GRAY2BGR)
for sr in vsr:
    wordColor[maxTransIndex, sr.startIndex] = np.array([0,0,255])
    wordColor[maxTransIndex, sr.endIndex]   = np.array([0,0,255])
    wordColor[maxTransIndex, sr.cutIndex]   = np.array([0,255,0])
cv2.imwrite("validCutPoints.png", wordColor)'''

#cv2.imshow("cutPoints", wordColor)
#cv2.waitKey(0)

#cv2.destroyAllWindows()

def validCutRegions(path, lineImage, wordImage):

    im = cv2.imread(path + lineImage)

    thresValue = 127

    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,thresValue,255,0)
    thresh = cv2.resize(thresh , (thresh.shape[1]*3 , thresh.shape[0]*3))
    ret,thresh = cv2.threshold(thresh,thresValue,255,0)
    
    #thresh[thresh == 255] = 1 
    #thresh = zhangSuen(thresh)
    #thresh[thresh == 1] = 255 
    


    wordOrg = cv2.imread(path + wordImage)
    wordGray = cv2.cvtColor(wordOrg, cv2.COLOR_BGR2GRAY)
    ret, wordThresh = cv2.threshold(wordGray, thresValue,255,0)
    wordThresh = cv2.resize(wordThresh , (wordThresh.shape[1]*3 , wordThresh.shape[0]*3))
    ret, wordThresh = cv2.threshold(wordThresh, thresValue,255,0)

    #wordThresh[wordThresh == 255] = 1 
    #wordThresh = zhangSuen(wordThresh)
    #wordThresh[wordThresh == 1] = 255 

    wordColor = cv2.cvtColor(wordThresh, cv2.COLOR_GRAY2BGR)
    baselineIndex = detectBaselineIndex(thresh)
    
    newWordCopy = wordThresh.copy()
    #for SR in range(0, wordThresh.shape[1]):
        

    #print("here")
    #cv2.imwrite("5555555.png", newWordCopy)
    

    maxTransIndex = maxTransitionIndex(newWordCopy, baselineIndex)
    #print(baselineIndex, maxTransIndex)
    img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    sepRegions = cutPoints(thresh, newWordCopy, maxTransIndex)
    for sr in sepRegions:
        wordColor[maxTransIndex, sr.startIndex] = np.array([0,0,255])
        wordColor[maxTransIndex, sr.endIndex]   = np.array([0,0,255])
        wordColor[maxTransIndex, sr.cutIndex]   = np.array([0,255,0])

    cv2.imwrite(path + "wordImage\\" + wordImage + "cutPoints.png", wordColor)

    vsr = separationRegionFilter(thresh, newWordCopy, sepRegions, baselineIndex, maxTransIndex)
    wordColor = cv2.cvtColor(newWordCopy, cv2.COLOR_GRAY2BGR)
    for sr in vsr:
        wordColor[maxTransIndex, sr.startIndex] = np.array([0,0,255])
        wordColor[maxTransIndex, sr.endIndex]   = np.array([0,0,255])
        wordColor[maxTransIndex, sr.cutIndex]   = np.array([0,255,0])
    cv2.imwrite(path + "wordImage\\" + wordImage + "validCutPoints.png", wordColor)

    # start_point = (0, maxTransIndex)
    # end_point = (wordColor.shape[1], maxTransIndex)
    # thickness = 1
    # color = (0,0,255)
    # wordColor = cv2.cvtColor(newWordCopy, cv2.COLOR_GRAY2BGR)
    # output = cv2.line(wordColor, start_point, end_point, color, thickness)
    # cv2.imwrite("wordImage\\" + "maxTransIndex.png", output)

    return len(vsr)

def validCutRegionsFinal(lineImage, wordImage):

    im = lineImage

    thresValue = 127

    # imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # ret,thresh = cv2.threshold(imgray,thresValue,255,0)
    thresh = cv2.resize(im , (im.shape[1]*3 , im.shape[0]*3))
    ret,thresh = cv2.threshold(thresh,thresValue,255,0)
    
    #thresh[thresh == 255] = 1 
    #thresh = zhangSuen(thresh)
    #thresh[thresh == 1] = 255 
    


    wordOrg = wordImage
    # wordGray = cv2.cvtColor(wordOrg, cv2.COLOR_BGR2GRAY)
    # ret, wordThresh = cv2.threshold(wordGray, thresValue,255,0)
    wordThresh = cv2.resize(wordOrg , (wordOrg.shape[1]*3 , wordOrg.shape[0]*3))
    ret, wordThresh = cv2.threshold(wordThresh, thresValue,255,0)

    #wordThresh[wordThresh == 255] = 1 
    #wordThresh = zhangSuen(wordThresh)
    #wordThresh[wordThresh == 1] = 255 

    wordColor = cv2.cvtColor(wordThresh, cv2.COLOR_GRAY2BGR)
    baselineIndex = detectBaselineIndex(wordThresh)
    
    newWordCopy = wordThresh.copy()
    #for SR in range(0, wordThresh.shape[1]):
        

    #print("here")
    #cv2.imwrite("5555555.png", newWordCopy)
    

    maxTransIndex = maxTransitionIndex(newWordCopy, baselineIndex)
    #print(baselineIndex, maxTransIndex)
    img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    sepRegions = cutPoints(thresh, newWordCopy, maxTransIndex)
    for sr in sepRegions:
        wordColor[maxTransIndex, sr.startIndex] = np.array([0,0,255])
        wordColor[maxTransIndex, sr.endIndex]   = np.array([0,0,255])
        wordColor[maxTransIndex, sr.cutIndex]   = np.array([0,255,0])
    wordBeforeFilter = wordColor

    vsr = separationRegionFilter(thresh, newWordCopy, sepRegions, baselineIndex, maxTransIndex)
    segmentedChars = []
    for index, sr in enumerate(vsr):
        if index == 0:
            segmentedChars.append(newWordCopy[:, sr.cutIndex:])
            if index == len(vsr) - 1:
                segmentedChars.append(newWordCopy[:, :sr.cutIndex])
        elif index == len(vsr) - 1:
            segmentedChars.append(newWordCopy[:, vsr[index].cutIndex: vsr[index-1].cutIndex])
            segmentedChars.append(newWordCopy[:, :sr.cutIndex])
        else:
            segmentedChars.append(newWordCopy[:, vsr[index].cutIndex: vsr[index-1].cutIndex])
    wordColor = cv2.cvtColor(newWordCopy, cv2.COLOR_GRAY2BGR)
    for sr in vsr:
        wordColor[maxTransIndex, sr.startIndex] = np.array([0,0,255])
        wordColor[maxTransIndex, sr.endIndex]   = np.array([0,0,255])
        wordColor[maxTransIndex, sr.cutIndex]   = np.array([0,255,0])

    # start_point = (0, maxTransIndex)
    # end_point = (wordColor.shape[1], maxTransIndex)
    # thickness = 1
    # color = (0,0,255)
    # wordColor = cv2.cvtColor(newWordCopy, cv2.COLOR_GRAY2BGR)
    # output = cv2.line(wordColor, start_point, end_point, color, thickness)
    # cv2.imwrite("wordImage\\" + "maxTransIndex.png", output)
    if len(segmentedChars) == 0: segmentedChars = [newWordCopy]
    return len(vsr), wordBeforeFilter, wordColor, segmentedChars


if __name__ == "__main__":
    print(validCutRegions("", '0.png', '0_11.png'))