import os
import json
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load model and class mapping
model = load_model('plant_model.h5')
with open('class_indices.json') as f:
    class_indices = json.load(f)
    class_names = {v: k for k, v in class_indices.items()}

def predict_image(img_path):
    img = load_img(img_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_id = np.argmax(prediction[0])
    return class_names[class_id]

def get_plant_uses(name):
    data = {
        'Aloe_Vera': 'Used for burns, skin healing.',
        'Neem': 'Anti-bacterial, used in dental care.',
        'Tulsi': 'Boosts immunity, treats cold.'
    }
    return data.get(name, "No data available")

def get_soil_type(name):
    data = {
        'Aloe_Vera': 'Sandy, well-drained soil.',
        'Neem': 'Deep, well-drained, sandy loam.',
        'Tulsi': 'Rich loamy soil.'
    }
    return data.get(name, "Soil type unknown")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img_path = os.path.join('static', 'uploaded_image.jpg')
            file.save(img_path)
            predicted_plant = predict_image(img_path)
            prediction = {
                'plant': predicted_plant,
                'uses': get_plant_uses(predicted_plant),
                'soil': get_soil_type(predicted_plant),
                'image_path': img_path
            }
    return render_template('index.html', prediction=prediction, has_prediction=bool(prediction))

if __name__ == '__main__':
    app.run(debug=True)
