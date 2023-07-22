import pickle, os, io, re, gc
from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel
from keras.models import Model, load_model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from PIL import Image


cwd = os.getcwd()

############################### Custom Model #############################
embedding_dim = 300
count = 0
max_caption_length = 80
models = os.path.join(cwd, "model")
hf_model = os.path.join(cwd, "hf_model")

word_index_Mapping = pickle.load(
    open(os.path.join(models, "word_index_Mapping.pkl"), "rb")
)
index_word_Mapping = pickle.load(
    open(os.path.join(models, "index_word_Mapping.pkl"), "rb")
)
vocab_size = len(word_index_Mapping) + 1
incpmodel = InceptionV3(weights="imagenet")
inceptionModel = Model(incpmodel.input, incpmodel.layers[-2].output)
model_weights_save_path = os.path.join(models, "model.h5")
predictionModel = load_model(model_weights_save_path)


############################### Hugging Face Model #################################

# os.environ['TRANSFORMERS_CACHE'] = hf_model
# os.environ['HF_HOME'] = hf_model
# os.environ['HF_DATASETS_CACHE'] = hf_model
# os.environ['XDG_CACHE_HOME'] = hf_model

device='cpu'
encoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
decoder_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
model_checkpoint = "nlpconnect/vit-gpt2-image-captioning"
feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_checkpoint, cache_dir=hf_model)
tokenizer = AutoTokenizer.from_pretrained(decoder_checkpoint, cache_dir=hf_model)
model = VisionEncoderDecoderModel.from_pretrained(model_checkpoint, cache_dir=hf_model)


def predictCustomModel(img):
    img = img.resize((299, 299), Image.LANCZOS)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    vectorImg = inceptionModel.predict(img)
    in_text = "startSeq"
    for i in range(1, max_caption_length):
        seq = [
            word_index_Mapping[w] for w in in_text.split() if w in word_index_Mapping
        ]
        in_seq = pad_sequences([seq], maxlen=max_caption_length)
        inputs = [vectorImg, in_seq]
        yhat = predictionModel.predict(x=inputs, verbose=0)
        yhat = np.argmax(yhat)
        word = index_word_Mapping[yhat]
        in_text += " " + word
        if word == "endSeq":
            break
    final = in_text.split()
    final = final[1:-1]
    final = " ".join(final)
    predict = re.sub(r"\b(\w+)( \1\b)+", r"\1", final)
    del img
    del vectorImg
    del final
    del in_text
    del seq
    del inputs
    gc.collect()
    return predict



def predictHFModel(image, max_length=224, num_beams=4):  
   image = image.convert('RGB')
   image = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
   clean_text = lambda x: x.replace('<|endoftext|>','').split('\n')[0]
   caption_ids = model.generate(image, max_length = max_length)[0]
   caption_text = clean_text(tokenizer.decode(caption_ids))
   prediction = re.sub(r"\b(\w+)( \1\b)+", r"\1", caption_text.strip())
   del image
   del clean_text
   del caption_ids
   del caption_text
   gc.collect()
   return prediction