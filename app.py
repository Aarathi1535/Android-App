from flask import Flask, request, redirect, url_for, render_template, session
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
app.secret_key = os.getenv('SECRET_KEY', 'mysecret')  # For session management
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return redirect(url_for('error', message='No file part'))

    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return redirect(url_for('error', message='No selected files'))

    prompt = request.form.get('prompt', '')
    if not prompt:
        return redirect(url_for('error', message='No prompt provided'))

    combined_text = ""
    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        text = extract_text_from_image(filepath)
        combined_text += text + "\n"

    try:
        full_response = evaluate_text(prompt, combined_text)
        # Store the full response in the session
        session['full_response'] = full_response
        return redirect(url_for('result'))
    except Exception as e:
        return redirect(url_for('error', message=str(e)))

@app.route('/result')
def result():
    full_response = session.get('full_response', 'No response available')
    # Clear the response from the session
    session.pop('full_response', None)
    return render_template('result.html', response=full_response)

@app.route('/error')
def error():
    message = request.args.get('message', 'An unknown error occurred')
    return render_template('error.html', message=message)

def extract_text_from_image(image_path):
    """Extracts text from an image file using OCR."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def evaluate_text(prompt, text):
    """Evaluates the extracted text using Gemini API and returns the full response."""
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

    return response.text

def parse_score_from_response(response_text):
    """Parses and returns only the numerical score from the response text."""
    print(f"Response Text: {response_text}")

    match = re.search(r'(?i)\bscore:\s*(\d+(\.\d+)?)\b', response_text)
    
    if match:
        score = match.group(1)
        return score
    else:
        return "Score not found"

if __name__ == "__main__":
    app.run(debug=True)
