from flask import Flask, render_template, request
from cv2 import imwrite, imread, resize
import numpy as np
import tensorflow.keras.models
import re
import base64
import tensorflow as tf
import sys 
import os
sys.path.append(os.path.abspath("./model"))


d = {0: 'a',
 1: 'aa',
 2: 'ae',
 3: 'aee',
 4: 'ana',
 5: 'ba',
 6: 'bhha',
 7: 'cha',
 8: 'chha',
 9: 'd_saa',
 10: 'da',
 11: 'de',
 12: 'dh_ru',
 13: 'dha',
 14: 'dha (1)',
 15: 'dhha',
 16: 'du',
 17: 'ga',
 18: 'gha',
 19: 'h_ru',
 20: 'haa',
 21: 'he',
 22: 'hu',
 23: 'ja',
 24: 'ja (1)',
 25: 'jchcha',
 26: 'jha',
 27: 'ka',
 28: 'kha',
 29: 'khya',
 30: 'la',
 31: 'laa',
 32: 'm_saa',
 33: 'ma',
 34: 'na',
 35: 'o',
 36: 'ou',
 37: 'p_saa',
 38: 'pa',
 39: 'pha',
 40: 'ra',
 41: 'ta',
 42: 'taa',
 43: 'tha',
 44: 'thha',
 45: 'wuan',
 46: 'yaa'}

app = Flask(__name__)
    
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict_num/', methods=['GET','POST'])
def predict_num():
    # get data from drawing canvas and save as image
    parseImage(request.get_data())
    #print(type(request.get_data()))

    # read parsed image back in 8-bit, black and white mode (L)
    x = imread('output.png') 
    x = np.invert(x)
    x = resize(x,(32, 32))

    # reshape image data for use in neural network
    x = x.reshape(1, 32, 32, 3)
    model = tf.keras.models.load_model('adam.h5')
    out = model.predict(x)
    #print(out)
    print(np.argmax(out))
    response = np.array_str(np.argmax(out, axis = 1))
    return response

@app.route('/predict_char/', methods=['GET','POST'])
def predict_char():
    # get data from drawing canvas and save as image
    parseImage(request.get_data())
    #print(type(request.get_data()))

    # read parsed image back in 8-bit, black and white mode (L)
    x = imread('output.png') 
    x = np.invert(x)
    x = resize(x,(81, 81))

    # reshape image data for use in neural network
    x = x.reshape(1, 81, 81, 3)
    model = tf.keras.models.load_model('model.h5')
    out = model.predict(x)
    #print(out)
    print(np.argmax(out))
    res = np.argmax(out, axis = 1)
    response = d[res[0]]
    return response
    
def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    #print(imgstr)
    #print(re.search(b'base64,(.*)', imgData).groups())
    #print(base64.decodebytes(imgstr))
    with open('output.png','wb') as output:
        output.write(base64.decodebytes(imgstr))


if __name__ == '__main__':
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port)
