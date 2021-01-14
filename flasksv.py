from flask import Flask, session
from flask import request, jsonify
import base64
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import torch
import glob
from imutils import paths
import numpy as np
# import module sys to get the type of exception
import sys
from flask_session import Session
import uuid
import settings
import time
import redis
import json
import pandas as pd
# from flask.ext.session import Session
db = redis.StrictRedis(host=settings.REDIS_HOST,
	port=settings.REDIS_PORT, db=settings.REDIS_DB)
    
app = Flask(__name__)
SESSION_TYPE = 'filesystem'
app.config.from_object(__name__)
Session(app)


@app.route("/postimg", methods = ['POST', 'GET'])
def home():
    print("Start new request!")
    returnData = {"success": False}
    db.flushdb()
    if request.method == 'POST':
        try:
            data = request.json
            imageBase64 = data['image']
            if imageBase64 is not None:
                imageBase64 = bytes(imageBase64, encoding="utf-8")
                image_64_decode = base64.decodebytes(imageBase64)
                k = str(uuid.uuid4())
                image_result = open(k + '.jpg', 'wb')
                image_result.write(image_64_decode)
                print('Image saved!')
                k = 'IMG_5043'
                d = {"id": k}
                db.rpush(settings.IMAGE_QUEUE, json.dumps(d))
                while True:
                    output = db.get(k)
                    if output is not None:
                        returnData["predictions"] = base64.b64encode(output).decode("UTF-8")
                        db.delete(k)
                        break
                    time.sleep(settings.CLIENT_SLEEP)
                returnData["success"] = True
                return returnData
        except:
            
            print('Exception!', sys.exc_info())
            pass
        return jsonify(returnData)
    if request.method == 'GET':
        print('start request!')
        try:
            if session['num_user'] is not None:
                session['num_user'] += 1  
            else:
                session['num_user'] = 1
            print(session['num_user'])
        except:
            print(sys.exc_info())
            session['num_user'] = 1
            pass
        k = 'IMG_5043'
        d = {'id': k}
        db.rpush(settings.IMAGE_QUEUE, json.dumps(d))
        # try:
        print('here')
        while True:
            output = db.get(k)
            if output is not None:
                # output = output
                returnData["predictions"] = output #json.loads(output)
                db.delete(k)
                break
            time.sleep(settings.CLIENT_SLEEP)
        returnData["success"] = True
        f = open("myfile.txt", "w")
        f.write(str(returnData))
        f.close()
        f = open("myfile.txt", "r")
        json_string = json.dumps(f.read(), ensure_ascii=False)
        return returnData
        # except:
        #     print('exception!')
        #     pass
        # return returnData

@app.route("/test", methods = ['GET'])
def test():
    print('here')
    listTestFile = os.listdir('testing')
    # listTestFile = listTestFile[:3]
    for f in listTestFile:
        # k = 'IMG_5043'
        d = {'id': 'testing/' + f}
        db.rpush(settings.IMAGE_QUEUE, json.dumps(d))
        # break
    print(listTestFile)
    returnData = []
    numresult = 0
    while True:
        for f in listTestFile:
            k = 'testing/' + f
            output = db.get(k)
            if output is not None:
                output = json.loads(output)
                # print(output['id'])
                output['filename'] = f
                # returnData[f.split('.')[0]] = json.loads(output)
                returnData.append(output)
                db.delete(k)
                numresult += 1
            time.sleep(settings.CLIENT_SLEEP)
        if numresult == len(listTestFile):
            break
        time.sleep(settings.CLIENT_SLEEP)
    # returnData["success"] = True
    df = pd.json_normalize(returnData)
    writer = pd.ExcelWriter('/home/quan/machine_learning/testing/similar/predict_result1.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet2', index=False)
    writer.save()
    # f = open("test_result.txt", "w")
    # f.write(returnData)
    # f.close()
    return {"success":True}

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
    
