import io
import os
import uuid
from PIL import Image


def validate_image(file_object, allowed_extensions=None):
    """
    Validate an uploaded image file

    Args:
        file_object: The uploaded file object (Streamlit's UploadedFile)
        allowed_extensions (list, optional): List of allowed file extensions.
                                           Defaults to ['.jpg', '.jpeg', '.png', '.bmp', '.gif'].

    Returns:
        bool: True if valid, False otherwise
    """
    if allowed_extensions is None:
        allowed_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]

    # Check if file exists
    if file_object is None:
        return False

    # Get filename
    filename = file_object.name

    # Check file extension
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in allowed_extensions:
        return False

    try:
        # Try opening the image to verify it's valid
        image = Image.open(file_object)
        image.verify()

        # Reset file position
        file_object.seek(0)
        return True
    except Exception as e:
        print(f"Image validation error: {e}")
        return False


def save_uploaded_image(file_object, upload_folder):
    """
    Save an uploaded image to disk with a unique filename

    Args:
        file_object: The uploaded file object (Streamlit's UploadedFile)
        upload_folder (str): The folder to save the image to

    Returns:
        str: The path to the saved image or None if failed
    """
    if file_object is None:
        return None

    try:
        # Reset file pointer to beginning
        file_object.seek(0)

        # Read file content
        contents = file_object.read()

        # Create a unique filename
        file_ext = os.path.splitext(file_object.name)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"

        # Make sure the upload folder exists
        os.makedirs(upload_folder, exist_ok=True)

        # Define the full file path
        file_path = os.path.join(upload_folder, unique_filename)

        # Save the file
        with open(file_path, "wb") as f:
            f.write(contents)

        # Verify the saved file can be opened
        try:
            with Image.open(file_path) as test_img:
                # Just test that it can be opened
                test_img.verify()
        except Exception as e:
            print(f"Verification of saved image failed: {e}")
            # Try an alternative approach to save the file
            img = Image.open(io.BytesIO(contents))
            img.save(file_path)

        return file_path
    except Exception as e:
        print(f"Error saving file: {e}")
        return None


def resize_image(image, max_size=(800, 800)):
    """
    Resize an image while maintaining aspect ratio

    Args:
        image (PIL.Image.Image or str): Image to resize or path to image
        max_size (tuple, optional): Maximum width and height. Defaults to (800, 800).

    Returns:
        PIL.Image.Image: Resized image
    """
    # Handle image input - could be a file path or PIL Image
    if isinstance(image, str):
        try:
            # Explicitly use 'RGB' mode to ensure proper color handling
            image = Image.open(image).convert("RGB")
        except Exception as e:
            raise ValueError(f"Could not open image file: {str(e)}")

    # Get original dimensions
    width, height = image.size

    # Check if resize is needed
    if width <= max_size[0] and height <= max_size[1]:
        return image

    # Calculate new dimensions
    if width > height:
        new_width = max_size[0]
        new_height = int(height * (max_size[0] / width))
    else:
        new_height = max_size[1]
        new_width = int(width * (max_size[1] / height))

    # Resize and return
    return image.resize((new_width, new_height), Image.LANCZOS)
