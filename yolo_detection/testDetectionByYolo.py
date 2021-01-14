import detect_custom
import cv2

img = cv2.imread('/home/quan/server/yolo_detection/data/IMG_4989.JPG')

detect_custom.detect_custom_detect(imgCvt = img)
# detect_custom.detect_custom_detect()