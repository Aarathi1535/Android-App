import streamlit as st
import os
from dotenv import load_dotenv
from PIL import Image
import io
import google.generativeai as genai
from pdf2image import convert_from_bytes
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("GEMINI_API_KEY")
if api_key is None:
    st.error("GEMINI_API_KEY environment variable is not set")
    st.stop()

# Configure the API with the key
genai.configure(api_key=api_key)

# Functions
def convert_pdf_to_images(pdf_file):
    """Converts a PDF file into a list of images."""
    if pdf_file.type == "application/pdf":
        images = convert_from_bytes(pdf_file.read())
    return images

def evaluate_image(image, user_score):
    """Evaluates the image using Gemini API and returns the score."""
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    # Convert image to bytes
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        image_bytes = output.getvalue()

    # Prepare the prompt
    prompt = f"Extract the text from the image and evaluate it to a score of {user_score}. Give a final score as output."

    # Generate content using the model with Blob
    response = model.generate_content([prompt, image])

    if response is None or not hasattr(response, 'text'):
        raise ValueError("No valid response received from the model")

    # Print the raw response text for debugging
    #st.write("Raw Response Text:")
    st.write(response.text)

    # Extract and return the score from the response
    response_text = response.text
    # Adjust extraction logic as needed
    lines = response_text.split('\n')
    for line in lines:
        if 'Score'.lower() in line:
            return line.strip()

    return "Score not found"

# Load custom CSS
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Path to your CSS file in the 'static' directory
css_file = Path("static/styles.css")

# Apply custom CSS
load_css(css_file)

# Streamlit UI
st.title("Automated Answer Sheet Evaluation")
user_score = st.text_input("Enter the score you would want to evaluate the paper for:")
uploaded_pdf = st.file_uploader("Upload your answer sheet PDF", type=["pdf"])

if st.button("Evaluate"):
    if not uploaded_pdf:
        st.error("Please upload a PDF file.")
    else:
        combined_score = ""
        try:
            images = convert_pdf_to_images(uploaded_pdf)
            for image in images:
                score = evaluate_image(image, user_score)
                combined_score += score + "\n"
            
            st.success("Evaluation completed!")
            st.text_area("Evaluation Result", combined_score)
        except Exception as e:
            st.error(f"Error: {str(e)}")
