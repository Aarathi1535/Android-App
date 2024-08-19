import streamlit as st
import os
from dotenv import load_dotenv
from PIL import Image
import io
import google.generativeai as genai
from pdf2image import convert_from_bytes
from pathlib import Path
import base64

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if api_key is None:
    st.error("GEMINI_API_KEY environment variable is not set")
    st.stop()

genai.configure(api_key=api_key)

def convert_pdf_to_images(pdf_file):
    """Converts a PDF file into a list of images."""
    if pdf_file.type == "application/pdf":
        images = convert_from_bytes(pdf_file.read())
    return images

def evaluate_image(image, user_score):
    """Evaluates the image using Gemini API and returns the score."""
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    with io.BytesIO() as output:
        image.save(output, format="PNG")
        image_bytes = output.getvalue()

    prompt = f"Extract the text from the image and evaluate it to a score of {user_score}. Give a final score as output."

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

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    """Read a binary file and encode it in base64."""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_img_as_page_bg(img_file):
    """Set an image file (PNG or JPEG) as the background."""
    bin_str = get_base64_of_bin_file(img_file)
    # Determine the file type from the file extension
    file_extension = img_file.split('.')[-1]
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/%s;base64,%s");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    }
    </style>
    ''' % (file_extension, bin_str)
    
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call the function with your image file path
set_img_as_page_bg('answer_sheet_bg.jpg')
    
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
