import cv2
import matplotlib.pyplot as plt
from detecto import utils, visualize
import tensorflow as tf
import os
import torch
import glob
from imutils import paths
import numpy as np
# import module sys to get the type of exception
import sys
import vietOcrUtils

class PredictUtils:

    originalLabel = {
        'card': [
            'top_left', 'top_right', 'bottom_left', 'bottom_right'
        ],
        'info': [
            'id', 'name', 'birth', 'country', 'home'
        ]
    }

    #Thực hiện predict và crop chứng minh nhân dân
    #Predict các boxes, labels, score
    def get_prediction(model, image):
        # image = utils.read_image('drive/My Drive/Dataset/id_card/orther.jpg')
        # labels, boxes, scores = m.predict(image)
        return model.predict(image)

    # Thực hiện visual các boxes đã được predict trên ảnh
    def get_visualize(image, boxes, labels):
        visualize.show_labeled_image(image, boxes, labels)

    #Dùng NMS để đưa ra được box đáng tin cậy nhất 
    def non_max_suppresion(boxes, scores, num_final_labels = 4, map_label = None):
        selected_indices, selected_scores = tf.image.non_max_suppression_padded(boxes, scores, 4)
        selected_boxes = tf.gather(boxes, selected_indices)
        if map_label is not None:
            selected_indices = [map_label[i] for i in selected_indices]
        return selected_indices, selected_boxes

    #lấy ra điểm trung tâm của box 
    def getCenterPoint(box):
        xmin, ymin, xmax, ymax = box
        return ((xmin + xmax) / 2), ((ymin + ymax) / 2)

    #Thực hiện dịch chuyển ảnh về hệ tọa độ gốc
    def perspective_transoform(image, source_points):
        dest_points = np.float32([[0,0], [856,0], [856,540], [0,540]])
        M = cv2.getPerspectiveTransform(source_points, dest_points)
        dst = cv2.warpPerspective(image, M, (856, 540))
        return dst
    #Kiểm tra xem kết quả predict có đủ các boxes
    def check_if_has_enough_label(final_label):
        for label in PredictUtils.originalLabel['card']:
            if final_label.index(label) < 0:
                return False
        return True

    #Hàm chính : thực hiện đọc ảnh, predict và crop ra ảnh chứng minh 
    def crop_image(model, img_path, num_label = 4, get_visualize=False):
        #read image
        print(img_path)
        image = utils.read_image(img_path + '.jpg')
        #prediction
        labels, boxes, scores = PredictUtils.get_prediction(model, image)
        #get final result
        final_labels, final_boxes = PredictUtils.non_max_suppresion(boxes, scores, num_label, labels)
        #visualize
        # if get_visualize is True:
        #     visualize.show_labeled_image(image, final_boxes.numpy(), final_labels)
        final_points = list(map(PredictUtils.getCenterPoint, final_boxes.numpy()))
        label_boxes = dict(zip(final_labels, final_points))
        allow_crop = PredictUtils.check_if_has_enough_label(final_labels)
        if allow_crop is True:
            source_points = np.float32([
                label_boxes['top_left'], label_boxes['top_right'], label_boxes['bottom_right'], label_boxes['bottom_left']
            ])

            # Transform 
            crop = PredictUtils.perspective_transoform(image, source_points)
            cv2.imwrite('cut_img.jpg', crop)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            # plt.imshow(crop)
            return crop

    def getImageDetectInfoPosition(model, image):
        labels, boxes, scores = model.predict(image)
        final_labels, final_boxes = [], []
        for lb in PredictUtils.originalLabel['info']:
            iterLabels = [i for i in range(len(labels)) if labels[i] == lb]
            labels_id = [i for i in labels if i == lb]
            # print(len(iterLabels))
            # visualize.show_labeled_image(image, boxes, labels)

            boxes_id = []
            score_id = []
            maxiter = 0
            for k in iterLabels:
                boxes_id.append(boxes.numpy()[k].tolist())
                score_id.append(scores[k].tolist())
            boxes_id = np.array(boxes_id)
            boxes_id = torch.from_numpy(boxes_id)

            score_id = np.array(score_id)
            score_id = torch.from_numpy(score_id)

            nms_labels, nms_boxes = PredictUtils.non_max_suppresion(boxes_id, score_id, 1, labels_id)
            # nms_boxes = torch.from_numpy(nms_boxes.numpy())
            final_boxes.append(nms_boxes.numpy()[0].tolist())
            final_labels.append(nms_labels[0])
        final_boxes = torch.from_numpy(np.array(final_boxes))
        result = []
        # visualize.show_labeled_image(image, final_boxes, final_labels)
        return {'labels': final_labels, 'boxes': final_boxes}

    def crop_image_use_yolo(model, img_path, num_label = 4, get_visualize=False):
        #read image
        if '.' in img_path:
            image = cv2.imread(img_path)
        else:
            image = cv2.imread(img_path + '.jpg')
            
        predictions = model.detect_custom_detect(imgCvt=image)
        predictions = predictions.numpy()
        boxes = predictions[:,:4]
        classNumbers = predictions[:,5]

        final_labels = [PredictUtils.originalLabel['card'][int(i)] for i in classNumbers] 
        final_points = list(map(PredictUtils.getCenterPoint, boxes))
        # print(final_points)
        label_boxes = dict(zip(final_labels, final_points))
        allow_crop = PredictUtils.check_if_has_enough_label(final_labels)
        if allow_crop is True:
            source_points = np.float32([
                label_boxes['top_left'], label_boxes['top_right'], label_boxes['bottom_right'], label_boxes['bottom_left']
            ])

            # Transform 
            crop = PredictUtils.perspective_transoform(image, source_points)
            cv2.imwrite('cut_img.jpg', crop)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            # plt.imshow(crop)
            return crop
        return None
    def get_image_detect_info_with_yolo(model, croppedImage):
        predictions = model.detect_custom_detect(imgCvt=croppedImage)
        # classNumbers = predictions[:, 5].numpy()
        # final_labels = [PredictUtils.originalLabel['info'][int(i)] for i in classNumbers]
        # final_boxes = predictions[:, :4]
        predictions = predictions.numpy()
        predictions = PredictUtils.sortBoxesCoordinate(predictions)
        predictions = np.array(predictions)
        # print('predictions sorted: ', np.array(predictions))
        # print(np.array(predictions)[:, :4])
        classNumbers = predictions[:, 5]
        final_labels = [PredictUtils.originalLabel['info'][int(i)] for i in classNumbers]
        final_boxes = predictions[:, :4]
        return {'labels': final_labels, 'boxes': final_boxes}

    def sortBoxesCoordinate(boxes, asc = True):
        print('start sort: ')
        sortedBoxes = []
        if asc is True:
            minBox = boxes[0]
            iter = 0
            while True:
                minIndexValue = 0
                if len(boxes) > 0:
                    minBox = boxes[0]
                    for i in range(len(boxes)):
                        if boxes[i][1] < minBox[1]:
                            minBox = boxes[i]
                            minIndexValue = i
                    
                    sortedBoxes.append(boxes[minIndexValue])
                    boxes = np.delete(boxes, minIndexValue, 0)
                else:
                    break
        print('finish sort!')
        return sortedBoxes
    
    def formatValue(value, label):
        lenValue = len(value)
        if label == 'id':
            count = 0
            while count < len(value):
                if not value[count].isnumeric():
                    lenValue = len(value)
                    value = value[0:count] + value[count+1:lenValue]
                else:
                    count += 1
        elif label == 'birth':
            for i in range(lenValue):
                if not value[i].isnumeric():
                    value  = value.replace(value[i], '-')
        return value
    