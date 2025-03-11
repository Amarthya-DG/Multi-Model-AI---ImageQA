import torch
from transformers import (
    BlipForQuestionAnswering,
    BlipProcessor,
    ViltForQuestionAnswering,
    ViltProcessor,
)


class ModelManager:
    """
    Class to manage loading and caching of various VQA models from Hugging Face
    """

    def __init__(self, cache_dir=None):
        """
        Initialize the model manager

        Args:
            cache_dir (str, optional): Directory to cache models. Defaults to None.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir
        self.models = {}
        self.processors = {}

        # Print device being used
        print(f"Using device: {self.device}")

    def load_blip(self):
        """
        Load BLIP model for VQA

        Returns:
            tuple: (processor, model)
        """
        if "blip" not in self.models:
            print("Loading BLIP model for visual question answering...")

            # Load processor and model
            processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

            # Move model to appropriate device
            model.to(self.device)

            # Store model and processor
            self.models["blip"] = model
            self.processors["blip"] = processor

            print("BLIP model loaded successfully!")

        return self.processors["blip"], self.models["blip"]

    def load_vilt(self):
        """
        Load ViLT model for VQA

        Returns:
            tuple: (processor, model)
        """
        if "vilt" not in self.models:
            print("Loading ViLT model for visual question answering...")

            # Load processor and model
            processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-vqa")
            model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-vqa")

            # Move model to appropriate device
            model.to(self.device)

            # Store model and processor
            self.models["vilt"] = model
            self.processors["vilt"] = processor

            print("ViLT model loaded successfully!")

        return self.processors["vilt"], self.models["vilt"]

    def get_model(self, model_name="blip"):
        """
        Get a model by name

        Args:
            model_name (str, optional): Name of model to load. Defaults to "blip".
                                       Options: "blip", "vilt"

        Returns:
            tuple: (processor, model)
        """
        if model_name.lower() == "blip":
            return self.load_blip()
        elif model_name.lower() == "vilt":
            return self.load_vilt()
        else:
            raise ValueError(
                f"Unknown model: {model_name}. Available models: blip, vilt"
            )
