# # backend/app.py
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# from werkzeug.utils import secure_filename

# # Initialize Flask app
# from flask_cors import CORS
# CORS(app, resources={r"/": {"origins": ""}})

# # Path to saved model
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "diabetic_retinopathy_model.h5")

# # Load the trained model once at startup
# model = load_model(MODEL_PATH)
# print("✅ Model loaded successfully!")

# # Define class names
# CLASS_NAMES = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']

# # Folder to temporarily save uploaded images
# UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/')
# def home():
#     return "Diabetic Retinopathy Detection Backend Running ✅"

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400
    
#     # Save uploaded image
#     filename = secure_filename(file.filename)
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(file_path)
    
#     try:
#         # Load and preprocess image
#         img = image.load_img(file_path, target_size=(224, 224))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0) / 255.0
        
#         # Make prediction
#         preds = model.predict(img_array)
#         predicted_class = CLASS_NAMES[np.argmax(preds[0])]
#         confidence = float(np.max(preds[0]))
        
#         # Cleanup (optional)
#         os.remove(file_path)
        
#         # Return JSON response
#         return jsonify({
#             'predicted_class': predicted_class,
#             'confidence': confidence,
#             'all_probabilities': {
#                 CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))
#             }
#         })
    
#     except Exception as e:
#         print("❌ Error during prediction:", e)
#         return jsonify({'error': str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins":["http://127.0.0.1:5500","http://localhost:5500"]}},supports_credentials=True)

# Folder for uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'diabetic_retinopathy_model.h5')
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Class names (modify if different)
class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']

@app.route('/')
def home():
    return jsonify({"message": "Flask backend is running successfully!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        preds = model.predict(img_array)
        predicted_class = class_names[np.argmax(preds)]
        confidence = float(np.max(preds))

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'severity': predicted_class,
            'recommendation': "Consult your eye specialist for confirmation."
        })

    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)