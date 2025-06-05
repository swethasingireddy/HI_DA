from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import os
import pandas as pd
import traceback
import tflite_runtime.interpreter as tflite

app = Flask(__name__)
CORS(app)

base_dir = os.path.dirname(os.path.abspath(__file__))

# Loading class names
class_map_path = os.path.join(base_dir, 'yamnet_class_map.csv')
class_names = pd.read_csv(class_map_path)['display_name'].tolist()

# Loading TFLite model
model_path = os.path.join(base_dir, 'yamnet.tflite')
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Hazardous classification indices (from YAMNet)
hazardous_classes = {
    11, 102, 181, 280, 281, 307, 316, 317, 318,
    319, 390, 393, 394, 420, 421, 422, 423, 424, 428, 429
}
# Semi Hazardous classification indices (from YAMNet)
semi_hazardous_classes = {302, 312}

sampleRate = 16000
chunkDuration = 2.0
chunkSamples = int(sampleRate * chunkDuration)

def load_audio(file_path):
    try:
        waveform, sr = librosa.load(file_path, sr=sampleRate, mono=True) # Loading and resampling audio
        return waveform
    except Exception as e:
        raise ValueError(f"librosa failed to load audio: {e}")

def classify_audio(waveform):
    if len(waveform) < chunkSamples:
        waveform = np.pad(waveform, (0, chunkSamples - len(waveform)), mode='constant')
    else:
        waveform = waveform[:chunkSamples]

    input_tensor = np.array(waveform, dtype=np.float32)

    input_shape = input_details[0]['shape']
    if len(input_shape) == 1:
        input_tensor = input_tensor[:input_shape[0]]
    else:
        raise ValueError(f"Unsupported input shape: {input_shape}")

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    top_class = int(np.argmax(output_data))
    prediction = {
        'class_name': class_names[top_class],
        'score': round(float(output_data[top_class]), 4),
        'hazardous': top_class in hazardous_classes,
        'semi_hazardous': top_class in semi_hazardous_classes
    }
    return prediction
#API endpoint to classify audio
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        file = request.files['audio']
        temp_path = os.path.join(base_dir, 'temp.wav')
        file.save(temp_path)

        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            return jsonify({'error': 'Uploaded file is empty'}), 400

        print(f"Received file: {file.filename} Saved at: {temp_path}")

        waveform = load_audio(temp_path)
        prediction = classify_audio(waveform)

        os.remove(temp_path)

        return jsonify({'predictions': [prediction]})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
