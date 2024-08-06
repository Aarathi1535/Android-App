import os
import streamlit as st
import logging
from PIL import Image
import io
from google.cloud import vision_v1 as vision
import google.generativeai as genai
from dotenv import load_dotenv

# Configure standard logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("GEMINI_API_KEY")
if api_key is None:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Configure the API with the key
genai.configure(api_key=api_key)

# Initialize Google Cloud Vision client
vision_client = vision.ImageAnnotatorClient()

@st.cache_resource
def load_models():
    return genai.GenerativeModel("gemini-pro")

def get_gemini_pro_text_response(model, contents: str, generation_config: dict, stream: bool = True):
    try:
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
    except Exception as e:
        st.error(f"Error generating response from Gemini: {str(e)}")
        return ""

def extract_text_from_image(image: Image.Image) -> str:
    """Extract text from image using Google Cloud Vision API."""
    try:
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            content = buffer.getvalue()

        image = vision.Image(content=content)
        response = vision_client.text_detection(image=image)
        texts = response.text_annotations
        if texts:
            return texts[0].description
        return ""
    except Exception as e:
        st.error(f"Error extracting text from image: {str(e)}")
        return ""

st.header("Automated Answer Sheet Evaluator", divider="gray")
text_model_pro = load_models()
st.subheader("AI Evaluator")

uploaded_files = st.file_uploader("Upload your answer sheets", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
marks = st.selectbox(
    "Select the score you would want to assign the paper?",
    list(range(1, 21)),
    index=None,
    placeholder="Select the marks."
)
max_output_tokens = 2048

if uploaded_files and marks:
    combined_text = ""
    all_evaluations = []
    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file)

            # Extract text from image
            extracted_text = extract_text_from_image(image)

            # Create prompt and get response
            prompt = f"""Use OCR to extract the text from the provided images and evaluate the text to a score of {marks}.\nText:\n{extracted_text}\nPlease provide some feedback."""

            config = {
                "temperature": 0.8,
                "max_output_tokens": max_output_tokens
            }

            response = get_gemini_pro_text_response(
                text_model_pro,
                prompt,
                generation_config=config,
            )

            if response:
                st.write(f"Results for {uploaded_file.name}:")
                st.write(response)
                all_evaluations.append(response)
                logging.info(response)
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")

    overall_feedback = "\n\n".join(all_evaluations) if all_evaluations else "No evaluations to display."

    if st.button("Evaluate"):
        with st.spinner("Evaluating your paper using Gemini..."):
            first_tab1, first_tab2 = st.tabs(["Marks", "Prompt"])
            with first_tab1:
                st.write("Overall Feedback:")
                st.write(overall_feedback)
            with first_tab2:
                st.text(prompt)
