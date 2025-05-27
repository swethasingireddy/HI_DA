from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import librosa
import numpy as np
import os
import pandas as pd
import traceback

app = Flask(__name__)
CORS(app)

# Paths
model_dir = os.path.join(os.getcwd(), 'models')
TFLITE_MODEL_PATH = os.path.join(model_dir, 'yamnet.tflite')
CLASS_MAP_PATH = os.path.join(model_dir, 'yamnet_class_map.csv')

# Load class names from CSV
class_names = pd.read_csv(CLASS_MAP_PATH)['display_name'].tolist()

# Hazardous sound class indices
hazardous_classes = {
    11, 102, 181, 280, 281, 307, 316, 317, 318,
    319, 390, 393, 394, 420, 421, 422, 423, 424, 428, 429
}
semi_immediate_classes = {302, 312}

# Audio settings (0.975 seconds = 15,600 samples at 16kHz)
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.975
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def load_audio(file_path):
    # Load audio file at 16kHz mono
    waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return waveform

def classify_single_chunk(waveform):
    # Pad or trim to 15,600 samples (0.975s)
    if len(waveform) < CHUNK_SAMPLES:
        waveform = np.pad(waveform, (0, CHUNK_SAMPLES - len(waveform)), mode='constant')
    else:
        waveform = waveform[:CHUNK_SAMPLES]

    input_data = waveform.astype(np.float32)

    # Adjust input shape to match model expectations
    expected_shape = input_details[0]['shape']
    if len(expected_shape) == 2:
        input_data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get scores
    scores = interpreter.get_tensor(output_details[0]['index'])  # shape: (1, 521)
    mean_scores = scores[0]

    top_class = int(np.argmax(mean_scores))

    prediction = {
        'class_name': class_names[top_class],
        'score': round(float(mean_scores[top_class]), 4),
        'hazardous': top_class in hazardous_classes,
        'semi_immediate': top_class in semi_immediate_classes
    }
    return prediction

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Received request")
        if 'audio' not in request.files:
            print("No audio file in request")
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']
        print("File received:", file.filename)

        temp_path = 'temp_chunk.wav'
        file.save(temp_path)

        waveform = load_audio(temp_path)
        prediction = classify_single_chunk(waveform)

        os.remove(temp_path)
        return jsonify({'predictions': [prediction]})

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
