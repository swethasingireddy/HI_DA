import numpy as np
import soundfile as sf
from flask import Flask, request, jsonify
from scipy.signal import resample
import io
import json
import os

from tflite_runtime.interpreter import Interpreter

app = Flask(__name__)

# Load TFLite model and allocate tensors
interpreter = Interpreter(model_path="yamnet.tflite")
interpreter.allocate_tensors()

# Load class labels
with open("class_map.json", "r") as f:
    class_map = json.load(f)

# Helper function: resample to 16kHz mono
def preprocess_audio(audio_bytes):
    waveform, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    
    # Convert stereo to mono if needed
    if len(waveform.shape) == 2:
        waveform = np.mean(waveform, axis=1)

    # Resample if sample rate isn't 16000
    if sr != 16000:
        num_samples = int(len(waveform) * 16000 / sr)
        waveform = resample(waveform, num_samples)

    return waveform.astype(np.float32)

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    file = request.files['audio']
    audio_bytes = file.read()

    try:
        waveform = preprocess_audio(audio_bytes)

        # Prepare model input
        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']

        if len(input_shape) == 2:
            # [1, 16000] — add batch dimension
            input_tensor = np.expand_dims(waveform, axis=0).astype(np.float32)
        else:
            # [16000] — 1D expected
            input_tensor = waveform.astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()

        # Get predictions
        output_details = interpreter.get_output_details()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_index = int(np.argmax(output_data))
        predicted_label = class_map.get(str(predicted_index), "Unknown")

        return jsonify({
            'class_id': predicted_index,
            'class_label': predicted_label
        })

    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
