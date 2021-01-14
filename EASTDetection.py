# import the necessary packages
from imutils.object_detection import non_max_suppression
from PIL import Image
import numpy as np
import argparse
import time
import uuid
import cv2

class EASTDetection():
    def __init__(self):
        self.model_link = "model/EAST/frozen_east_text_detection.pb"
        self.min_confidence = 0.5
        self.width = 320
        self.height = 320

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        self.layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"
        ]
        print("[INFO] loading EAST text detector...")
        self.net = cv2.dnn.readNet(self.model_link)

    def groupBoxes(self, boxes, need_check = True):
        if len(boxes) > 0:
            origin_boxes = boxes
            final_boxes = []
            need_check_again = []
            while len(origin_boxes) > 1:
                first_box = origin_boxes[0]
                origin_boxes = np.delete(origin_boxes, 0, 0)
                group = [first_box]
                (f_startX, f_startY, f_endX, f_endY) = first_box
                run = 0
                # if need_check == False:
                    # print("origin_box: {}, run: {}".format(len(origin_boxes), run))
                while len(origin_boxes) > 0 and run < len(origin_boxes):
                    # print('len origin box: ', len(origin_boxes))
                    # print('run: ', run)
                    (startX, startY, endX, endY) = origin_boxes[run]
                    if not((f_startY > startY and f_startY > endY) or (f_endY < startY and f_endY < endY)):
                        group.append(origin_boxes[run])
                        origin_boxes = np.delete(origin_boxes, run, 0)
                    else:
                        run = run + 1
                        #start group all box in group
                if len(group) > 1:
                    (min_startX, min_startY, max_endX, max_endY) = (-1, -1, -1, -1)
                    for (startX, startY, endX, endY) in group:
                        # final_boxes.append(box)
                        if min_startX < 0 or startX < min_startX:
                            min_startX = startX
                        if min_startY < 0 or startY < min_startY:
                            min_startY = startY
                        if max_endX < 0 or endX > max_endX:
                            max_endX = endX
                        if max_endY < 0 or endY > max_endY:
                            max_endY = endY
                    final_boxes.append([min_startX, min_startY, max_endX, max_endY])
                elif len(group) == 1:
                    print('group[0]', group[0])
                    need_check_again.append(group[0])
            if len(origin_boxes) > 0:
                for box_or in origin_boxes:
                    need_check_again.append(box_or)
            if len(need_check_again) > 0:
                for box_ca in need_check_again:
                    final_boxes.append(box_ca)
                if need_check == True:
                    # print('here')
                    return self.groupBoxes(final_boxes, False)
            
            return final_boxes
    
    def getMaxSquareBoxes(self, boxesArr, numGet):
        if len(boxesArr) < 3:
            return boxesArr
        sqArr = []
        idx = 0
        for b in boxesArr:
            x1, y1, x2, y2 = b
            sqArr.append([idx, (x2 - x1) * (y2 - y1)])
            idx += 1

        for i in range(0, len(sqArr) - 1, 1):
            for j in range(i, len(sqArr), 1):
                if sqArr[i][1] < sqArr[j][1]:
                    tmp = sqArr[i]  
                    sqArr[i] = sqArr[j]
                    sqArr[j] = tmp
        result = []
        for s in sqArr[:2]:
            result.append(boxesArr[s[0]])
        return result
            
    def detectWordsBoxes(self, image):
        if image is None:
            return []
        orig = image.copy()
        # image = cv2.resize(image, (856, 540), interpolation = cv2.INTER_AREA)
        copyOrigin = image.copy()
        (H, W) = image.shape[:2]

        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (self.width, self.height)
        rW = W / float(newW)
        rH = H / float(newH)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        start = time.time()
        self.net.setInput(blob)
        (scores, geometry) = self.net.forward(self.layerNames)
        end = time.time()
        # show timing information on text prediction
        print("[INFO] text detection took {:.6f} seconds".format(end - start))
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < self.min_confidence:
                    continue
                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        boxes = self.groupBoxes(boxes)
        boxes = self.getMaxSquareBoxes(boxes, 2)
        # loop over the bounding boxes
        c_iter = 0
        print('phóng to: {}, {}'.format(rW, rH))
        # rH = 1 

        print('Number of bounding box: ', boxes)
        iterWord = 0

        result = []
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW) -5
            startY = int(startY * rH) -5
            endX = int(endX * rW) + 5
            endY = int(endY * rH) + 5
            if startX < 0:
                startX = 0
            if startY < 0:
                startY = 0
            if endX < 0:
                endX = 0
            if endY < 0:
                endY = 0
            crop_img_w = orig[startY:endY, startX:endX]
            
            result.append(crop_img_w)
            # draw the bounding box on the image
            cv2.rectangle(copyOrigin, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # cv2.imshow('title', copyOrigin)
        copyOrigin = Image.fromarray(copyOrigin)
        idu = str(uuid.uuid1()) + '.jpg'
        copyOrigin.save(idu, 'JPEG', quality=100)
        return result