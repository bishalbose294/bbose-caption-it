import os
import pickle
import numpy as np
from PIL import Image
import re
import gc
import urllib
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input
from tensorflow.keras.layers import add, BatchNormalization, LSTM, Dense, Embedding, Dropout, Input
from tensorflow.keras import regularizers, optimizers, initializers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template,  url_for, request

app = Flask(__name__)

embedding_dim = 300
count=0
max_caption_length = 80

word_index_Mapping = pickle.load(open('word_index_Mapping.pkl','rb'))
index_word_Mapping = pickle.load(open('index_word_Mapping.pkl','rb'))

vocab_size = len(word_index_Mapping) + 1

incpmodel = InceptionV3(weights='imagenet')
inceptionModel = Model(incpmodel.input, incpmodel.layers[-2].output)

model_weights_save_path = 'model.h5'
predictionModel = load_model(model_weights_save_path)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predictCaption():
    count = 0
    url = request.form['imageSource']
    imageName = "./imageData/"+str(count)+".jpg"
    urllib.request.urlretrieve(url, imageName)
    img = Image.open(imageName)
    img = img.resize((299,299), Image.ANTIALIAS)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    vectorImg = inceptionModel.predict(img)
    in_text = 'startSeq'
    for i in range(1, max_caption_length):
        seq = [word_index_Mapping[w] for w in in_text.split() if w in word_index_Mapping]
        in_seq = pad_sequences([seq], maxlen=max_caption_length)
        inputs = [vectorImg,in_seq]
        yhat = predictionModel.predict(x=inputs, verbose=0)
        yhat = np.argmax(yhat)
        word = index_word_Mapping[yhat]
        in_text += ' ' + word
        if word == 'endSeq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    predict = re.sub(r'\b(\w+)( \1\b)+', r'\1', final)
    os.remove(imageName)
    
    del img
    del imageName
    del vectorImg
    del final
    del count
    del in_text
    del seq
    del inputs
    gc.collect()
    
    return render_template('./result.html',prediction = predict, urlImg = url)

if __name__ == '__main__':
    app.run()
