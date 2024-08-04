import os
import streamlit as st
import logging
from google.cloud import logging as cloud_logging
import vertexai
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
from datetime import (
    date,
    timedelta,
)
# configure logging
logging.basicConfig(level=logging.INFO)
# attach a Cloud Logging handler to the root logger
log_client = cloud_logging.Client()
log_client.setup_logging()

@st.cache_resource
def load_models():
    text_model_pro = GenerativeModel("gemini-pro")
    return text_model_pro

def get_gemini_pro_text_response(
    model: GenerativeModel,
    contents: str,
    generation_config: GenerationConfig,
    stream: bool = True,
):
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    responses = model.generate_content(
        prompt,
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=stream,
    )

    final_response = []
    for response in responses:
        try:
            # st.write(response.text)
            final_response.append(response.text)
        except IndexError:
            # st.write(response)
            final_response.append("")
            continue
    return " ".join(final_response)

st.header("Automated Answer Sheet Evaluator", divider="gray")
text_model_pro = load_models()
st.subheader("AI Evaluator")
uploaded_files = st.file_uploader("Upload your answer sheets", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
for uploaded_file in uploaded_files:
    image = Image.open(uploaded_file)
marks = st.selectbox(
    "Select the score you would want to assign the paper?",
    (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20),
    index=None,
    placeholder="Select the marks."
)
max_output_tokens = 2048

prompt = f"""I am an Evaluator.  Extract the text from the {image} provided and evaluate the text to a  score of {marks}\n
Please provide some feedback.
"""

config = {
    "temperature": 0.8,
    "max_output_tokens": 2048,
}

generate_t2t = st.button("Evaluate.", key="generate_t2t")
if generate_t2t and prompt:
    # st.write(prompt)
    with st.spinner("Evaluating your paper using Gemini..."):
        first_tab1, first_tab2 = st.tabs(["Marks", "Prompt"])
        with first_tab1:
            response = get_gemini_pro_text_response(
                text_model_pro,
                prompt,
                generation_config=config,
            )
            if response:
                st.write("Your results:")
                st.write(response)
                logging.info(response)
        with first_tab2:
            st.text(prompt)
