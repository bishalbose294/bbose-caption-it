import os, io, base64
from imageCaptionGPT import predictCustomModel, predictHFModel
from flask import Flask, render_template, request
from flask_cors import CORS
from PIL import Image
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
CORS(app)

cwd = os.getcwd()

app.config["ALLOWED_EXTENSIONS"] = [".jpg", ".png"]
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
app.config["UPLOAD_FOLDER"] = os.path.join(cwd, "uploads")


@app.route("/")
def home():
    return render_template("home.html", predictionCustomModel="", predictionHFModel="", img_data="")


@app.route("/predict", methods=["POST"])
def predictCaption():
    file = request.files["file"]
    imagePath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(imagePath)
    img = Image.open(imagePath)
    with io.BytesIO() as buf:
        img.save(buf, "jpeg")
        image_bytes = buf.getvalue()
    encoded_string = base64.b64encode(image_bytes).decode()
    predictionCustomModel = predictCustomModel(img)
    predictionHFModel = predictHFModel(img)
    predictionCustomModel = "Custom Model:- \'"+str(predictionCustomModel)+"\'"
    predictionHFModel = "HuggingFace Model:- \'"+str(predictionHFModel)+"\'"
    os.remove(imagePath)
    return render_template("home.html", predictionCustomModel=predictionCustomModel, predictionHFModel=predictionHFModel, img_data=encoded_string)


if __name__ == '__main__':
    host = '0.0.0.0'
    port = 7860
    print("#"*50,"--Application Serving Now--","#"*50)
    # app.run(host=host,port=port)
    app_serve = WSGIServer((host,port),app)
    app_serve.serve_forever()