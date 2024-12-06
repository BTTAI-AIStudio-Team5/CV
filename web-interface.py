from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import torch
import joblib
from CV.cvmodel import load_model, process_img

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        print(f"Receiving file {file.filename}...")
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"File saved to {filepath}")

        # Load model and process image
        print("Loading model...")
        try:
            model = load_model()
            label_encoder = joblib.load('label_encoder.pkl')
            print("Processing image...")
            result_img, info = process_img(filepath, model)
            
            # Format detection results
            results = []
            # Save and encode the result image
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + secure_filename(file.filename))
            cv2.imwrite(output_path, result_img)
            # Add the image path to results
            results.append({
                'message': f'<img src="../uploads/result_{secure_filename(file.filename)}" class="img-fluid" alt="Processed Image">'
            })
            for i, detection in enumerate(info):
                results.append({
                    'message': f"Object {i+1}: {detection['category']} at coordinates {detection['bbox']}"
                })


            # TODO: Add STRIPS planner here
            # TODO: Add LLM here
            # TODO: Add output here

            print("Returning results...")
            return jsonify({'messages': results})
        except Exception as e:
            print(f"Error: {str(e)}")
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)