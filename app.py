from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
import pytesseract  # For OCR
from PIL import Image  # For opening images
import google.generativeai as genai
from werkzeug.utils import secure_filename
import re

# Load environment variables from .env file
load_dotenv()
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Access the API key
api_key = os.getenv("GEMINI_API_KEY")
if api_key is None:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Configure the API with the key
genai.configure(api_key=api_key)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No selected files'}), 400
    
    prompt = request.form.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    combined_text = ""
    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        text = extract_text_from_image(filepath)
        combined_text += text + "\n"

    try:
        score = evaluate_text(prompt, combined_text)
        return jsonify({'score': score})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def extract_text_from_image(image_path):
    """Extracts text from an image file using OCR."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def evaluate_text(prompt, text):
    """Evaluates the extracted text using Gemini API and returns a score."""
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )

    full_prompt = f"{prompt}\n{text}"
    response = model.generate_content(full_prompt)

    if response is None or not hasattr(response, 'text'):
        raise ValueError("No valid response received from the model")

    print("Response received from Gemini API:", response.text)

    score = parse_score_from_response(response.text)
    return score

def parse_score_from_response(response_text):
    """Parses and returns only the numerical score from the response text."""
    # Log the response to understand its structure
    print(f"Response Text: {response_text}")

    # Regular expression to find the score (assuming it's a float or integer preceded by "Score:")
    match = re.search(r'(?i)\bscore:\s*(\d+(\.\d+)?)\b', response_text)
    
    if match:
        # Extract the score part from the match
        score = match.group(1)
        return score
    else:
        return "Score not found"

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    app.run(debug=True)
