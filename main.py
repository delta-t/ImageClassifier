import os
import base64
import io
import json
import cv2
import traceback
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, Response, abort
from markupsafe import escape
from flask_wtf import FlaskForm
from wtforms import StringField, FileField
from flask_wtf.file import FileRequired
from wtforms.validators import DataRequired
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config.update(dict(
    SECRET_KEY="gfopyvh64gfh",
    WTF_CSRF_SECRET_KEY="m2l4n4fd2rf"
))
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
imagenet_class_index = json.load(open('./imagenet_class_index.json'))
model = models.densenet121(pretrained=True)
model.eval()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def greetings_page():
    return render_template('index.html')


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.fromarray(image_bytes)
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    output = model.forward(tensor)
    _, y_hat = output.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]


class MyForm(FlaskForm):
    name = 'predict'
    photo = FileField('Photo', validators=[FileRequired('No file!')])


@app.route('/submit', methods=('GET', 'POST'))
def submit():
    try:
        form = MyForm()
        if form.validate_on_submit():
            f = form.photo.data
            if allowed_file(f.filename):
                npimg = np.fromstring(f.read(), np.uint8)
                # convert numpy array to image
                img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
                h, w, _ = img.shape
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                text = get_prediction(img)
                cv2.putText(img, text[1], (w // 4, 15*h // 16), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                _, jpg = cv2.imencode('.jpg', img)
                return Response(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n\r\n', mimetype='multipart/x-mixed-replace; boundary=frame')
            else:
                return redirect(url_for('bad_request'))

        return render_template('submit.html', form=form)
    except:
        traceback.print_exc()
        return redirect(url_for('bad_request'))


@app.route('/badrequest400')
def bad_request():
    return abort(400)


if __name__ == '__main__':
    app.run()

