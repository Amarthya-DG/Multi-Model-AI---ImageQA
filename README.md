# Visual Question Answering (VQA) System

A multi-modal AI application that allows users to upload images and ask questions about them. This project uses pre-trained models from Hugging Face to analyze images and answer natural language questions.

## Features

- Upload images in common formats (jpg, png, etc.)
- Ask questions about image content in natural language
- Get AI-generated answers based on image content
- User-friendly Streamlit interface
- Support for various types of questions (objects, attributes, counting, etc.)

## Technical Stack

- **Python**: Main programming language
- **PyTorch & Transformers**: Deep learning frameworks for running the models
- **Streamlit**: Interactive web application framework
- **HuggingFace Models**: Pre-trained visual question answering models
- **PIL**: Image processing

## Setup Instructions

1. Clone this repository:
   ```
   git clone https://github.com/your-username/visual-question-answering.git
   cd visual-question-answering
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python run.py
   ```
   
   Or directly with Streamlit:
   ```
   streamlit run app.py
   ```

5. Open a web browser and go to `http://localhost:8501`

## Usage

1. Upload an image using the file upload area
2. Type your question about the image in the text field
3. Select a model from the sidebar (BLIP or ViLT)
4. Click "Get Answer" to get an AI-generated response
5. View the answer displayed on the right side of the screen

## Models Used

This application uses the following pre-trained models from Hugging Face:
- **BLIP**: For general visual question answering with free-form answers
- **ViLT**: For detailed understanding of image content and yes/no questions

## Project Structure

- `app.py`: Main Streamlit application
- `models/`: Contains model handling code
- `utils/`: Utility functions for image processing and more
- `static/`: Static files including uploaded images
- `run.py`: Script to run the application

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for their excellent pre-trained models
- The open-source community for various libraries used in this project 