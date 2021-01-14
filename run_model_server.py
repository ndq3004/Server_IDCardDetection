# import the necessary packages
# from keras.applications import imagenet_utils
from detecto import core, utils, visualize
from predict_utils import PredictUtils
from vietOcrUtils import VietOcrUtils
from EASTDetection import EASTDetection
from yolo_detection.detect_custom import YoloPredictCustom as YoloPredict
from PIL import Image
from matplotlib import cm
import timeit
import cv2
import numpy as np
import settings
import helpers
import codecs
import uuid
import redis
import time
import json
import sys

# connect to Redis server
db = redis.StrictRedis(host=settings.REDIS_HOST,
	port=settings.REDIS_PORT, db=settings.REDIS_DB)
#defined label arr and link
originalLabel = {
	'card': [
		'top_left', 'top_right', 'bottom_left', 'bottom_right'
	],
	'info': [
		'id', 'name', 'birth', 'country', 'home'
	]
}
	
modelLink = {
	'card': 'model/model_detect_card/id_card_model_detect_id_card_499img_120ep_0311.pth',
	'info': 'model/model_detect_info/id_card_model_detect_info_50ep_161_img.pth'
}

modelYolo = {
	'card': 'yolo_detect_card.pt',
	'info': 'model_detect_info_relabel.pt'
}

def processPredictionWithYolo(model, listPath):
	result = []
	for path in listPath:
		# crop = None
		# crop = crop_image(model, path, num_label, True)
		print("process ", path)
		eResult = {}
		num_label_card = 4
		try:
			#crop ra riêng khung thẻ (loại bỏ background)
			time_start_crop = timeit.default_timer()
			crop = PredictUtils.crop_image_use_yolo(model['yolo_card'], path, num_label_card, False)
			time_end_crop = timeit.default_timer()
			print("Detect and crop with Yolo model take: ", time_end_crop - time_start_crop)

			if crop is not None:
				listInfoImgCropped = PredictUtils.get_image_detect_info_with_yolo(model['yolo_info'], crop)
				# print(listInfoImgCropped)
				listBoxes = listInfoImgCropped['boxes']
				listLabels = listInfoImgCropped['labels']
				
				for (label, box) in zip(listInfoImgCropped['labels'], listBoxes):
					startX, startY, endX, endY = box
					img = crop[int(startY):int(endY), int(startX):int(endX)]
					print('detect with label: ', label)
					if label == 'country' or label == 'home':
						img = Image.fromarray(img)
						strPredit = model['ocr'].predict(img)
						try:
							tmpStr = eResult[label]
							eResult[label] = tmpStr + ', ' + strPredit
						except KeyError:
							eResult[label] = strPredit
						print(strPredit)
					else:
						img = Image.fromarray(img)
						strPredit = model['ocr'].predict(img)
						strPredit = PredictUtils.formatValue(strPredit, label)
						eResult[label] = strPredit
						print(strPredit)

			result.append(eResult)
		except:
			print(sys.exc_info())
			pass
	print(result)
	return result

def processPrediction(model, listPath):
	result = []
	for path in listPath:
		# crop = None
		# crop = crop_image(model, path, num_label, True)
		eResult = {}
		num_label_card = 4
		try:
			#crop ra riêng khung thẻ (loại bỏ background)
			time_start_crop = timeit.default_timer()
			crop = PredictUtils.crop_image(model['card'], path, num_label_card, False)
			time_end_crop = timeit.default_timer()
			print("Detect and crop with fasterRCNN model take: ", time_start_crop = time_end_crop)

			listInfoImgCropped = PredictUtils.getImageDetectInfoPosition(model['info'], crop)
			print(listInfoImgCropped)
			listBoxesInfoImgCropped = listInfoImgCropped['boxes'].numpy()
			
			for (label, box) in zip(listInfoImgCropped['labels'], listBoxesInfoImgCropped):
				startX, startY, endX, endY = box
				img = crop[int(startY):int(endY), int(startX):int(endX)]
				strPredit = ''
				if label == 'country' or label == 'home':
					innerImg = model['east'].detectWordsBoxes(img)
					for iImg in innerImg:
						lstImg = Image.fromarray(iImg)
						strPredit += model['ocr'].predict(lstImg) + ' '

					eResult[label] = strPredit
				else:
					img = Image.fromarray(img)
					strPredit = model['ocr'].predict(img)
					eResult[label] = strPredit

			result.append(eResult)
		except:
			print(sys.exc_info())
			pass

	return result

def classify_process():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
	print("* Loading model...")
	model = {
		# 'card': core.Model.load(modelLink['card'], originalLabel['card']),
		# 'info': core.Model.load(modelLink['info'], originalLabel['info']),
		'ocr': VietOcrUtils(),
		# 'east': EASTDetection(),
		'yolo_card': YoloPredict(modelYolo['card']),
		'yolo_info': YoloPredict(modelYolo['info']),
	}

	print("* Model loaded")
	print("Ready to detect!")

	# continually pool for new images to classify
	while True:
		# attempt to grab a batch of images from the database, then
		# initialize the image IDs and batch of images themselves
		queue = db.lrange(settings.IMAGE_QUEUE, 0,
			settings.BATCH_SIZE - 1)
		imageIDs = []
		batch = None

		# loop over the queue
		for q in queue:
			# deserialize the object and obtain the input image
			q = json.loads(q.decode("utf-8"))
			# update the list of image IDs
			imageIDs.append(q["id"])
		print('get {} from queue'.format(len(imageIDs)))
		# check to see if we need to process the batch
		# imageIDs.append('IMG_5043')
		if len(imageIDs) > 0:
			# classify the batch
			print("* Batch size: {}".format(len(imageIDs)))
			# preds = model.predict(batch)
			# results = processPrediction(model, imageIDs)
			results = processPredictionWithYolo(model, imageIDs)
			# loop over the image IDs and their corresponding set of
			# results from our model
			for idx in range(len(imageIDs)):
				# initialize the list of output predictions
				print('initialize the list of output predictions: ', results)
				if len(results) > idx:
					db.set(imageIDs[idx], json.dumps(results[idx], ensure_ascii=False).encode('utf8'))
				else:
					db.set(imageIDs[idx], json.dumps({}, ensure_ascii=False).encode('utf8'))
			# remove the set of images from our queue
			db.ltrim(settings.IMAGE_QUEUE, len(imageIDs), -1)

		# sleep for a small amount
		time.sleep(settings.SERVER_SLEEP)

# if this is the main thread of execution start the model server
# process
if __name__ == "__main__":
	classify_process()