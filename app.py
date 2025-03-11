"""
Visual Question Answering Streamlit Application
"""

import logging
import os
import sys
import time
from datetime import datetime

import streamlit as st
from PIL import Image

# Configure path to include parent directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(
    log_dir, f"vqa_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger("vqa_app")

# Import modules
from models import VQAInference
from utils.image_utils import resize_image

# Global variables
MODEL_OPTIONS = {"BLIP": "blip", "ViLT": "vilt"}

# Setup directories
uploads_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "static", "uploads"
)
os.makedirs(uploads_dir, exist_ok=True)

# Configure page
st.set_page_config(
    page_title="Visual Question Answering",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model(model_name):
    """Load the VQA model with caching for better performance"""
    try:
        logger.info(f"Loading model: {model_name}")
        return VQAInference(model_name=model_name)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
        return None


def process_image_and_question(image_file, question, model_name):
    """Process the uploaded image and question to generate an answer"""
    start_time = time.time()

    try:
        # Load image
        image = Image.open(image_file).convert("RGB")
        logger.info(f"Image loaded, size: {image.size}")

        # Resize image
        image = resize_image(image)
        logger.info(f"Image resized to: {image.size}")

        # Load model
        model = load_model(model_name)
        if model is None:
            return None

        # Generate answer
        logger.info(f"Generating answer for question: '{question}'")
        answer = model.predict(image, question)
        logger.info(f"Answer generated: '{answer}'")

        # Calculate processing time
        processing_time = time.time() - start_time

        return {"answer": answer, "processing_time": f"{processing_time:.2f} seconds"}
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return None


def main():
    """Main function for Streamlit app"""
    # Header
    st.title("Visual Question Answering")
    st.markdown("Upload an image, ask a question, and get AI-powered answers")

    # Sidebar for model selection
    st.sidebar.title("Model Options")
    selected_model_name = st.sidebar.radio(
        "Choose a model:", options=list(MODEL_OPTIONS.keys()), index=0
    )
    model_name = MODEL_OPTIONS[selected_model_name]

    st.sidebar.markdown("---")
    st.sidebar.markdown("## About the Models")
    st.sidebar.markdown("**BLIP**: General purpose VQA with free-form answers")
    st.sidebar.markdown("**ViLT**: Better for yes/no questions and specific categories")

    # Main content - two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Upload & Ask")
        uploaded_file = st.file_uploader(
            "Upload an image:", type=["jpg", "jpeg", "png", "bmp", "gif"]
        )

        question = st.text_input(
            "Your question about the image:", placeholder="E.g., What is in this image?"
        )

        submit_button = st.button(
            "Get Answer", type="primary", use_container_width=True
        )

        # Preview uploaded image
        if uploaded_file is not None:
            st.markdown("### Image Preview")
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown("### AI Answer")

        # Process when submit button is clicked
        if submit_button and uploaded_file is not None and question:
            with st.spinner("Generating answer..."):
                result = process_image_and_question(uploaded_file, question, model_name)

                if result:
                    st.success("Answer generated successfully!")

                    # Display results
                    st.markdown("#### Question:")
                    st.write(question)

                    st.markdown("#### Answer:")
                    st.markdown(
                        f"<div style='background-color: #f0f2f6; padding: 20px; border-radius: 5px;'>{result['answer']}</div>",
                        unsafe_allow_html=True,
                    )

                    st.markdown("#### Processing Time:")
                    st.text(result["processing_time"])
                else:
                    st.error(
                        "Failed to generate an answer. Please check the image and question, and try again."
                    )

        elif not uploaded_file and submit_button:
            st.warning("Please upload an image first.")
        elif not question and submit_button:
            st.warning("Please enter a question about the image.")
        else:
            st.info("AI answers will appear here after you submit your question")

    # Information about the application
    st.markdown("---")
    st.markdown("### About Visual Question Answering")
    st.markdown("""
    This application uses multi-modal AI, combining computer vision and natural language processing 
    to answer questions about images. Here are some examples of questions you can ask:
    
    - **Objects**: "What objects are in this image?"
    - **Counting**: "How many people are in this image?"
    - **Colors**: "What color is the car?"
    - **Actions**: "What is the person doing?"
    - **Spatial relations**: "What is to the left of the chair?"
    - **Attributes**: "Is the cat sleeping?"
    """)


if __name__ == "__main__":
    main()
