import numpy as np
from PIL import Image
from typing import List

def prepare_images(input_images: List[Image.Image]) -> List[np.ndarray]:
    """
    Convert PIL Images to numpy arrays and ensure they're in RGB format.
    
    Args:
    input_images (List[Image.Image]): List of PIL Image objects
    
    Returns:
    List[np.ndarray]: List of numpy arrays representing RGB images
    """
    numpy_images = []
    for img in input_images:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        numpy_images.append(np.array(img))
    return numpy_images

def process_images(clip_processor, numpy_images: List[np.ndarray]):
    """
    Process images using the CLIP processor.
    
    Args:
    clip_processor: The CLIP processor object
    numpy_images (List[np.ndarray]): List of numpy arrays representing RGB images
    
    Returns:
    Processed images tensor
    """
    processed_images = clip_processor(images=numpy_images, return_tensors='pt', padding=True)
    return processed_images.pixel_values.to('cuda')