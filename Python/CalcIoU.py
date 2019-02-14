
import cv2
import json
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import sys
import pandas as pd
import os

def IoU(boxA, boxB):
    alx=float(boxA[0])
    aly=float(boxA[1])
    arx=float(boxA[2])
    ary=float(boxA[3])

    blx=float(boxB[0])
    bly=float(boxB[1])
    brx=float(boxB[2])
    bry=float(boxB[3])
    # determine the (x, y)-coordinates of the intersection rectangle
    xA =float( max(alx, blx))
    yA = float(max(aly, bly))
    xB = float(min(arx, brx))
    yB = float(min(bry, bry))
    # compute the area of intersection rectangle
    interArea = abs(max(xB-xA, 0) * max(yB - yA, 0))

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou
gtcount=0
precount=0
groundtruth={}
predictions={}
iterator=0
threshold=0.45
file=open('/Users/ryanreynolds/Documents/CSU Class Folders/Fall 2018/Senior Design/Ubuntu_Shared/Groundtruth/inputvid2/GroundTruth.csv', 'r')
file = file.readlines()
for lines in file[0:]:
    lines=lines.strip("\r\n")
    groundtruth[gtcount]=lines.split(',')
    gtcount=gtcount+1

iterator=0
for iterator in groundtruth:
    i=1
    while i < 6:
        if groundtruth[iterator][i]!='':
            groundtruth[iterator][i]=int(groundtruth[iterator][i])
        i+=1
        
file2=open('stats.txt')
file2=file2.readlines()
for prelines in file2[0:]:
    prelines=prelines.strip("\r\n")
    predictions[precount]=prelines.split(' ')
    precount=precount+1

for iterator in predictions:
    i=0
    predictions[iterator].append(0.0)
    predictions[iterator].append(0)

    while i < 6:
        predictions[iterator][i]=int(predictions[iterator][i])
        i+=1
test=0       




iterator=0
iterator2=0

for iterator in predictions:
    i=0
    while i<gtcount:
        if groundtruth[i][5]==predictions[iterator][0] and groundtruth[i][1]!='':
            boxA=(groundtruth[i][3], groundtruth[i][4],groundtruth[i][3]+groundtruth[i][2],groundtruth[i][4]+groundtruth[i][1])
            boxB=predictions[iterator][2:]
            iou=IoU(boxB,boxA)
            if test<50:
                print("Comparing Prediction: "+str(predictions[iterator]))
                print("To Ground Truth: "+str(groundtruth[i])+"\n")
            if iou>predictions[iterator][6]:
                predictions[iterator][6]=iou
                predictions[iterator][7]=i
            test+=1
        i+=1               


iterator=0
base={}
preresult={}
result={}
false_positives=0
iterator=0
for iterator in groundtruth:
    base[groundtruth[iterator][5]]=0
for iterator in groundtruth:
    if groundtruth[iterator][5]!='':
        base[groundtruth[iterator][5]]+=1
for iterator in predictions:
    preresult[predictions[iterator][0]]=0
for iterator in predictions:
    if predictions[iterator][6]>threshold and predictions[iterator][7] in groundtruth:
        del groundtruth[predictions[iterator][7]]
        preresult[predictions[iterator][0]]+=1
    elif predictions[iterator][6]<threshold:
        false_positives+=1;
sum3=0
sum2=0
print(predictions)
for i in preresult:
    if i!='':
        try:
            sum3+=preresult[i]
            sum2+=base[i]
            result[i]=float(sum3)/(sum2+false_positives)
        except KeyError:
            sum2+=0
            continue

iterator=0

plt.scatter(result.keys(), result.values(), s=10, c='b', marker="o", label='Hog')
plt.title("Hog and Haar Performance ("+str(100*threshold)+"% Threshold)")
plt.xlabel('Frame #')
plt.ylabel('Predicted TP/Ground Truth TP')
plt.legend(loc='upper right');
plt.show()
