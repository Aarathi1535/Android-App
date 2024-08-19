import streamlit as st
import os
from dotenv import load_dotenv
import pytesseract
from PIL import Image
import google.generativeai as genai
import re

# Load environment variables from .env file
load_dotenv()
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Access the API key
api_key = os.getenv("GEMINI_API_KEY")
if api_key is None:
    st.error("GEMINI_API_KEY environment variable is not set")
    st.stop()

# Configure the API with the key
genai.configure(api_key=api_key)

# Functions
def extract_text_from_image(image):
    """Extracts text from an image file using OCR."""
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
    if 'Score' in response.text:
        return response.text[response.text.index('Score'):].replace('*', ' ')

# Streamlit UI
st.title("Automated Answer Sheet Evaluation")

prompt = st.text_input("Enter the prompt for evaluation")

uploaded_files = st.file_uploader("Upload your answer sheets", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if st.button("Evaluate"):
    if not uploaded_files or not prompt:
        st.error("Please upload files and provide a prompt.")
    else:
        combined_text = ""
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            text = extract_text_from_image(image)
            combined_text += text + "\n"
        
        try:
            evaluation_result = evaluate_text(prompt, combined_text)
            st.success("Evaluation completed!")
            st.text_area("Evaluation Result", evaluation_result)
        except Exception as e:
            st.error(f"Error: {str(e)}")
