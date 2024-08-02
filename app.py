import logging
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
    try:
        if 'files' not in request.files:
            raise ValueError('No file part')
        
        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            raise ValueError('No selected files')
        
        prompt = request.form.get('prompt', '')
        if not prompt:
            raise ValueError('No prompt provided')

        combined_text = ""
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            text = extract_text_from_image(filepath)
            combined_text += text + "\n"

        score = evaluate_text(prompt, combined_text)
        return jsonify({'score': score})
    
    except Exception as e:
        logging.error(f"Error in upload_file: {str(e)}")  # Log the error
        return jsonify({'error': str(e)}), 500

def extract_text_from_image(image_path):
    """Extracts text from an image file using OCR."""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logging.error(f"Error extracting text from image {image_path}: {str(e)}")
        raise

def evaluate_text(prompt, text):
    """Evaluates the extracted text using Gemini API and returns a score."""
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )

        full_prompt = f"{prompt}\n{text}"
        response = model.generate_content(full_prompt)

        if response is None or not hasattr(response, 'text'):
            raise ValueError("No valid response received from the model")

        logging.info("Response received from Gemini API: %s", response.text)

        score = parse_score_from_response(response.text)
        return score
    except Exception as e:
        logging.error(f"Error in evaluate_text: {str(e)}")
        raise

def parse_score_from_response(response_text):
    """Parses and returns only the numerical score from the response text."""
    logging.info(f"Response Text: {response_text}")

    match = re.search(r'(?i)\bscore:\s*(\d+(\.\d+)?)\b', response_text)
    
    if match:
        score = match.group(1)
        return score
    else:
        return "Score not found"

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True)
