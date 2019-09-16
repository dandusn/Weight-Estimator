import cv2 as cv
import numpy as np
import os.path
import sys
import random
import math
from Engine.model.regression import regression

configThreshold = 0.9
maskThreshold = 0.5
listmasking = []


def findindex(array):
    index = 0
    scale = 0
    for i in range(len(array)):
        if array[i][0] > scale:
            scale = array[i][0]
            index = i
    return index

def masking(frame, classId, conf, left, top, right, bottom, classMask, classes, colors):
    label = '%.2f' % conf
    scale = 0

    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    if classes[classId] == "cow" or classes[classId] == "sheep":
        # Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

        classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
        mask = (classMask > maskThreshold)
        roi = frame[top:bottom + 1, left:right + 1][mask]
        print("area of image = " + str(len(roi)))

        colorIndex = random.randint(0, len(colors) - 1)
        color = colors[colorIndex]

        frame[top:bottom + 1, left:right + 1][mask] = (
                    [0.3 * color[0], 0.3 * color[1], 0.3 * color[2]] + 0.7 * roi).astype(np.uint8)

        # Draw the border on the image
        mask = mask.astype(np.uint8)
        border, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(frame[top:bottom + 1, left:right + 1], border, -1, color, 3, cv.LINE_8, hierarchy, 10)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
        print("length of edge = " + str(len(frame[top:bottom + 1, left:right + 1])))

        lborder = len(frame[top:bottom + 1, left:right + 1])
        scale = len(roi) / lborder
        cv.putText(frame, str(scale), (right, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
    else:
        scale = 0

    return scale, classes[classId]

def postprocess(frame, boxes, masks, classes, colors):
    # Output size of masks is NxCxHxW where0
    # N - number of detected boxes
    # C - number of classes (excluding background)
    # HxW - segmentation shape
    numClasses = masks.shape[1]
    numDetections = boxes.shape[2]

    frameH = frame.shape[0]
    frameW = frame.shape[1]
    count = 0
    score = 0
    scale = 0
    label = None

    listmask = [[0 for j in range(7)] for i in range(10)]

    for i in range(numDetections):
        box = boxes[0, 0, i]
        mask = masks[i]
        score = box[2]
        count += 1

        if score > configThreshold:
            classId = int(box[1])

            # Extract the bounding box
            left = int(frameW * box[3])
            top = int(frameH * box[4])
            right = int(frameW * box[5])
            bottom = int(frameH * box[6])

            left = max(0, min(left, frameW - 1))
            top = max(0, min(top, frameH - 1))
            right = max(0, min(right, frameW - 1))
            bottom = max(0, min(bottom, frameH - 1))

            # Extract the mask for the object
            classMask = mask[classId]
            out = masking(frame, classId, score, left, top, right, bottom, classMask, classes, colors)
            listmasking.append(out)

    if listmasking:
        objek = listmasking[findindex(listmasking)]
        scale = objek[0]
        label = objek[1]
        print('scale =', scale)

    return scale, label

def main(filename, age):

    pred = 0
    classesFile = './Engine/model/mscoco_labels.name'
    classobject = None

    with open(classesFile, 'rt') as f:
        classobject = f.read().rstrip('\n').split('\n')

    # laod the NN
    modelWeights = './Engine/model/frozen_inference_graph.pb'
    textGraph = './Engine/model/graph_txt.pbtxt'
    net = cv.dnn.readNetFromTensorflow(modelWeights, textGraph)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    colorsFile = "./Engine/model/colors.txt"
    with open(colorsFile, 'rt') as f:
        colorsStr = f.read().rstrip('\n').split('\n')
    colors = []

    for i in range(len(colorsStr)):
        rgb = colorsStr[i].split(' ')
        color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
        colors.append(color)
    print('load class ok')


    fn = "./static/upload/" + filename
    if fn:
        if not os.path.isfile(fn):
            print("image ", fn, " not exist")
            sys.exit(1)

        img = cv.imread(fn, flags=cv.IMREAD_UNCHANGED)
        img = cv.resize(img, (800, 600))
        split = str.split(str(fn), "/")
        true = split[len(split) - 1]
        outputFile = './static/output/' + str(true)
        cv.imwrite('./static/output/scaled/' + str(true), img)
        cap = cv.VideoCapture('./static/output/scaled/' + str(true))
    else:
        print('not image')

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv.waitKey(3000)
            break

        blob = cv.dnn.blobFromImage(frame, swapRB=True, crop=False)
        net.setInput(blob)
        boxes, masks = net.forward(['detection_out_final', 'detection_masks'])
        out = postprocess(frame, boxes, masks, classobject, colors)
        label = out[1]
        scale = out[0]

        t, _ = net.getPerfProfile()
        print('Masking time for a frame : %0.0f ms' % abs(t * 1000.0 / cv.getTickFrequency()))

        if scale == 0:
            print("scale tidak didapatkan")
        else:
            if label == None:
                print("objek tidak ditemukan")
            else:
                cv.imwrite(outputFile, frame.astype(np.uint8))
                rg = regression()
                if label == 'cow':
                    rg.trainmodel("./Engine/data/regression/regcow.csv")
                elif label == 'sheep':
                    rg.trainmodel("./Engine/data/regression/regsheep.csv")

                if label == "sheep":
                    pred = rg.prediction(label, scale, age, rg.knn)
                else:
                    pred = rg.prediction(label, scale, age, rg.linier)

    return math.floor(pred)

