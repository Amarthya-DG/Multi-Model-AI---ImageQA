import io
import os
import traceback
import torch
from PIL import Image, UnidentifiedImageError
from .model_loader import ModelManager


class VQAInference:
    """
    Class to perform inference with Visual Question Answering models
    """

    def __init__(self, model_name="blip", cache_dir=None):
        """
        Initialize the VQA inference

        Args:
            model_name (str, optional): Name of model to use. Defaults to "blip".
            cache_dir (str, optional): Directory to cache models. Defaults to None.
        """
        self.model_name = model_name
        self.model_manager = ModelManager(cache_dir=cache_dir)
        self.processor, self.model = self.model_manager.get_model(model_name)
        self.device = self.model_manager.device

    def predict(self, image, question):
        """
        Perform VQA prediction on an image with a question

        Args:
            image (PIL.Image.Image or str): Image to analyze or path to image
            question (str): Question to ask about the image

        Returns:
            str: Answer to the question
        """
        # Handle image input - could be a file path or PIL Image
        if isinstance(image, str):
            try:
                # Check if file exists
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image file not found: {image}")

                # Try multiple approaches to load the image
                try:
                    # Try the standard approach first
                    image = Image.open(image).convert("RGB")
                    print(
                        f"Successfully opened image: {image.size}, mode: {image.mode}"
                    )
                except Exception as img_err:
                    print(
                        f"Standard image loading failed: {img_err}, trying alternative method..."
                    )

                    # Try alternative approach with binary mode explicitly
                    with open(image, "rb") as img_file:
                        img_data = img_file.read()
                        image = Image.open(io.BytesIO(img_data)).convert("RGB")
                        print(
                            f"Alternative image loading succeeded: {image.size}, mode: {image.mode}"
                        )

            except UnidentifiedImageError as e:
                # Specific error when image format cannot be identified
                raise ValueError(f"Cannot identify image format: {str(e)}")
            except Exception as e:
                # Provide detailed error information
                error_details = traceback.format_exc()
                print(f"Error details: {error_details}")
                raise ValueError(f"Could not open image file: {str(e)}")

        # Make sure image is a PIL Image
        if not isinstance(image, Image.Image):
            raise ValueError("Image must be a PIL Image or a file path")

        # Process based on model type
        if self.model_name.lower() == "blip":
            return self._predict_with_blip(image, question)
        elif self.model_name.lower() == "vilt":
            return self._predict_with_vilt(image, question)
        else:
            raise ValueError(f"Prediction not implemented for model: {self.model_name}")

    def _predict_with_blip(self, image, question):
        """
        Perform prediction with BLIP model

        Args:
            image (PIL.Image.Image): Image to analyze
            question (str): Question to ask about the image

        Returns:
            str: Answer to the question
        """
        try:
            # Process image and text inputs
            inputs = self.processor(
                images=image, text=question, return_tensors="pt"
            ).to(self.device)

            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(**inputs)

            # Decode the output to text
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)

            return answer
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"Error in BLIP prediction: {str(e)}")
            print(f"Error details: {error_details}")
            raise RuntimeError(f"BLIP model prediction failed: {str(e)}")

    def _predict_with_vilt(self, image, question):
        """
        Perform prediction with ViLT model

        Args:
            image (PIL.Image.Image): Image to analyze
            question (str): Question to ask about the image

        Returns:
            str: Answer to the question
        """
        try:
            # Process image and text inputs
            encoding = self.processor(images=image, text=question, return_tensors="pt")

            # Move inputs to device
            for k, v in encoding.items():
                encoding[k] = v.to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**encoding)
                logits = outputs.logits

            # Get the predicted answer idx
            idx = logits.argmax(-1).item()

            # Convert to answer text
            answer = self.model.config.id2label[idx]

            return answer
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"Error in ViLT prediction: {str(e)}")
            print(f"Error details: {error_details}")
            raise RuntimeError(f"ViLT model prediction failed: {str(e)}")
