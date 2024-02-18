from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

app= Flask(__name__)
app.secret_key = 'vivekmtr'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}


# Specify the path to your .hdf5 file
model_path = "model/version1.hdf5"

# Load the model
MODEL = load_model(model_path)

CLASS_NAMES =["Early Blight","Late Blight", "Healthy"]
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    images = os.listdir(app.config['UPLOAD_FOLDER'])
    if images!=[]:
        predict_image=images[0]
        response_pre=predict(predict_image)
        return render_template('index.html',images=images, response_pre=response_pre)
    else:
        return render_template('index.html',images=images)
        
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('index'))

    flash('Invalid file format')
    return redirect(request.url)

@app.route('/delete/<filename>', methods=['POST'])
def delete_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash('File deleted successfully')
    return redirect(url_for('index'))


def read_file_image(data)-> np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image


def predict(file):
    
    try: 
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        with open(file_path, 'rb') as file:
            data = file.read()
        img_response= read_file_image(data)
        img_batch = np.expand_dims(img_response,0)
        prediction=MODEL.predict(img_batch)
        predicted_class_ot = CLASS_NAMES[np.argmax(prediction[0])]
        confidence_ot =round((np.max(prediction[0]))*100,2)
        result=[predicted_class_ot,confidence_ot]
    except Exception as e:
        result=['Invalid Image','Please Upload Again !']
    finally:
         return result



if __name__ == "__main__":
    app.secret_key = 'vivekmtr'
    app.run(debug=False)