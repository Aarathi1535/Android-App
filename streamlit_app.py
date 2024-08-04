import os
import streamlit as st
import logging
from PIL import Image
import easyocr
import google.generativeai as genai
from datetime import date, timedelta

# Configure standard logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Access the API key
api_key = os.getenv("GEMINI_API_KEY")
if api_key is None:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Configure the API with the key
genai.configure(api_key=api_key)

# Initialize easyocr reader
reader = easyocr.Reader(['en'])

@st.cache_resource
def load_models():
    return genai.GenerativeModel("gemini-pro")

def extract_text_from_image(image):
    """Extracts text from an image file using easyocr."""
    image_array = np.array(image)
    results = reader.readtext(image_array)
    text = ' '.join([result[1] for result in results])
    return text

def get_gemini_pro_text_response(
    model,
    contents: str,
    generation_config: dict,
    stream: bool = True,
):
    responses = model.generate_content(
        contents,
        generation_config=generation_config,
        stream=stream,
    )

    final_response = []
    for response in responses:
        try:
            final_response.append(response.text)
        except IndexError:
            final_response.append("")
            continue
    return " ".join(final_response)

st.header("Automated Answer Sheet Evaluator", divider="gray")
text_model_pro = load_models()
st.subheader("AI Evaluator")
uploaded_files = st.file_uploader("Upload your answer sheets", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
marks = st.selectbox(
    "Select the score you would want to assign the paper?",
    (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
    index=None,
    placeholder="Select the marks."
)
max_output_tokens = 2048

if uploaded_files and marks:
    combined_text = ""
    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file)
            text = extract_text_from_image(image)
            combined_text += text + "\n"
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")

    prompt = f"""Extract the text from the provided images using OCR and evaluate the text to a score of {marks}.\nPlease provide some feedback."""

    config = {
        "temperature": 0.8,
        "max_output_tokens": max_output_tokens
    }

    if st.button("Evaluate"):
        with st.spinner("Evaluating your paper using Gemini..."):
            first_tab1, first_tab2 = st.tabs(["Marks", "Prompt"])
            with first_tab1:
                try:
                    response = get_gemini_pro_text_response(
                        text_model_pro,
                        prompt + combined_text,
                        generation_config=config,
                    )
                    if response:
                        st.write("Your results:")
                        st.write(response)
                        logging.info(response)
                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")
                    logging.error(f"Error during evaluation: {str(e)}")
            with first_tab2:
                st.text(prompt)
