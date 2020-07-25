import os
import pickle
import numpy as np
import cv2
import re
import urllib
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input
from tensorflow.keras.layers import add, BatchNormalization, LSTM, Dense, Embedding, Dropout, Input
from tensorflow.keras import regularizers, optimizers, initializers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template,  url_for, request

app = Flask(__name__)

def loadWordEmbeddings():
    global word_index_Mapping
    global index_word_Mapping
    word_index_Mapping = pickle.load(open('word_index_Mapping.pkl','rb'))
    index_word_Mapping = pickle.load(open('index_word_Mapping.pkl','rb'))
    print('Words Loaded')

def loadModel():
    global inceptionModel
    global predictionModel
    global max_caption_length
    incpmodel = InceptionV3(weights='imagenet')
    inceptionModel = Model(incpmodel.input, incpmodel.layers[-2].output)


    embedding_dim = 300
    vocab_size = len(word_index_Mapping) + 1

    image_input = Input(shape=(2048,))
    x = Dropout(0.5)(image_input)
    image_encode = Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.000001))(x)

    text_input = Input(shape=(max_caption_length,))
    x = Embedding(vocab_size, embedding_dim, mask_zero=True)(text_input)
    x = Dropout(0.5)(x)
    text_encode = LSTM(256, activation='tanh', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.000001))(x)

    decoder_input = add([image_encode, text_encode])
    x = Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.000001))(decoder_input)
    output = Dense(vocab_size, activation='softmax')(x)
    predictionModel = Model(inputs=[image_input, text_input], outputs=output)

    model_weights_save_path = 'model.h5'
    predictionModel.load_weights(model_weights_save_path)
    
    print('Model Loaded')

def vectorize_image(imageFileName):
    global inceptionModel
    img = cv2.imread(imageFileName)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(299,299))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    vector = inceptionModel.predict(img)
    return vector

def get_caption_for_photo(photo_feature):
    global word_index_Mapping
    global index_word_Mapping
    global predictionModel
    global max_caption_length
    in_text = 'startSeq'
    for i in range(1, max_caption_length):
        seq = [word_index_Mapping[w] for w in in_text.split() if w in word_index_Mapping]
        in_seq = pad_sequences([seq], maxlen=max_caption_length)
        inputs = [photo_feature,in_seq]
        yhat = predictionModel.predict(x=inputs, verbose=0)
        yhat = np.argmax(yhat)
        word = index_word_Mapping[yhat]
        in_text += ' ' + word
        if word == 'endSeq':
            break

    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return re.sub(r'\b(\w+)( \1\b)+', r'\1', final)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predictCaption():
    global count
    url = request.form['imageSource']
    imageName = "./imageData/"+str(count)+".jpg"
    count+=1
    #print('\n\n\n\n\n\nURL : ',url)
    urllib.request.urlretrieve(url, imageName)
    img = vectorize_image(imageName)
    predict = get_caption_for_photo(img)
    os.remove(imageName)
    #print('\n\n\n\n\n\nPrediction : ',predict)
    return render_template('./result.html',prediction = predict, urlImg = url, count=count)

if __name__ == '__main__':
    word_index_Mapping = None
    index_word_Mapping = None
    inceptionModel = None
    predictionModel = None
    count=0
    max_caption_length = 80
    
    loadWordEmbeddings()
    loadModel()
    app.run()
