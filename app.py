import os
import pickle
from flask import Flask, request, render_template
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load .pkl model
MODEL_PATH = 'model/waste_model.pkl'
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)

model = model_from_json(model_data["architecture"])
model.set_weights(model_data["weights"])
LABELS = model_data["labels"]

# Compile model before use
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    uploaded_image = None

    if request.method == 'POST':
        img_file = request.files['image']
        if img_file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
            img_file.save(filepath)
            uploaded_image = filepath

            # Preprocess image
            img = image.load_img(filepath, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            pred = model.predict(img_array)[0]
            prediction = [(LABELS[i], round(float(p), 2)) for i, p in enumerate(pred) if p > 0.5]

    return render_template('index.html', prediction=prediction, image=uploaded_image)


if __name__ == '__main__':
    app.run(debug=True)
