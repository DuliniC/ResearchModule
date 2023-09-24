from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import base64

from eyebrow_suggestion import EyebrowShapingWaySuggestion
from makeup_suggestion import MakeoverSuggestion

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

obj = EyebrowShapingWaySuggestion()
msobj = MakeoverSuggestion()

@app.route('/process-image', methods=['POST'])
def upload_file():
    # Check if an image file is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Get the uploaded file
    file = request.files['image']

    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img, msg = obj.main(filepath)

    processed_image = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    output_filename = f"processed_{filename}"
    output_filepath = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    cv2.imwrite(output_filepath, img)

    with open(output_filepath, 'rb') as f:
        encoded_image = base64.b64encode(f.read()).decode('utf-8')

    skin_tone, suggested_makeup = msobj.main(filepath)


    arr_string = ', '.join(str(x) for x in suggested_makeup)
    
    makeovers = f"Suggested Makeup for {skin_tone} skin tone: {arr_string}"
    response = {'message': msg, 
                'processed_image': encoded_image,
                'makeover': makeovers}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
