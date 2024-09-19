import os
import torch
import argparse
from pathlib import Path
import logging
import gradio as gr
import numpy as np
from functools import partial

from models import load_clip, load_model, load_image_adapter
from image_processing import prepare_images, process_images
from text_processing import (tokenize_prompt, embed_prompt, embed_bos_token,
                             construct_input_tensors, decode_generated_ids,
                             set_vlm_prompt, get_vlm_prompt,
                             set_negative_vlm_prompt, get_negative_vlm_prompt)
from batch_processing import batch_process_images, interrupt_batch_processing
from utils import is_cuda_out_of_memory_error
from gradio_ui import create_ui, launch_ui

# Constants
CLIP_PATH = "google/siglip-so400m-patch14-384"
CHECKPOINT_PATH = Path("wpkklhc6")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
ORIGINAL_MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B"
QUANTIZED_MODEL_PATH = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables to store loaded models
clip_processor = None
clip_model = None
tokenizer = None
text_model = None
image_adapter = None

@torch.no_grad()
def stream_chat(input_images, vlm_prompt=None, negative_vlm_prompt=None, seed=-1, max_new_tokens=300, length_penalty=1.0, num_beams=1):
    """
    Generate captions for the given input images using the loaded models.

    Args:
    input_images (list): List of input images
    vlm_prompt (str, optional): Custom VLM prompt to use. If None, use the default.
    negative_vlm_prompt (str, optional): Custom negative VLM prompt to use. If None, no negative prompt is used.
    seed (int, optional): Random seed for generation. If -1, a random seed will be used.
    max_new_tokens (int, optional): Maximum number of new tokens to generate. Default is 300.
    length_penalty (float, optional): Encourages (>1.0) or discourages (<1.0) longer sequences. Default is 1.0.
    num_beams (int, optional): Number of beams for beam search. Default is 1.

    Returns:
    str: Generated captions or error message
    """
    global clip_processor, clip_model, tokenizer, text_model, image_adapter
    
    if not all([clip_processor, clip_model, tokenizer, text_model, image_adapter]):
        return "Error: Models not loaded. Please load a model first."

    torch.cuda.empty_cache()

    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set the VLM prompt and negative prompt if provided
    if vlm_prompt is not None:
        set_vlm_prompt(vlm_prompt)
    if negative_vlm_prompt is not None:
        set_negative_vlm_prompt(negative_vlm_prompt)

    # Tokenize and embed text
    prompt, negative_prompt = tokenize_prompt(tokenizer)
    prompt_embeds, negative_prompt_embeds = embed_prompt(text_model, prompt, negative_prompt)
    embedded_bos = embed_bos_token(text_model, tokenizer)

    # Prepare and process images
    numpy_images = prepare_images(input_images)
    pixel_values = process_images(clip_processor, numpy_images)

    # Process images with CLIP and image adapter
    with torch.amp.autocast('cuda', enabled=True):
        vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
        image_features = vision_outputs.hidden_states[-2]
        embedded_images = image_adapter(image_features)
        embedded_images = embedded_images.to()

    # Construct input tensors
    inputs_embeds, input_ids, attention_mask, negative_inputs_embeds, negative_input_ids, negative_attention_mask = construct_input_tensors(
        embedded_bos, embedded_images, prompt_embeds, negative_prompt_embeds, tokenizer, prompt, negative_prompt
    )

    # Generate captions
    generate_kwargs = {
        "input_ids": input_ids,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "top_k": 10,
        "temperature": 0.5,
        "suppress_tokens": None,
        "length_penalty": length_penalty,
        "num_beams": num_beams
    }

    if negative_inputs_embeds is not None:
        generate_kwargs["negative_prompt_ids"] = negative_input_ids
        generate_kwargs["negative_prompt_attention_mask"] = negative_attention_mask

    generate_ids = text_model.generate(**generate_kwargs)

    # Decode generated IDs into captions
    captions = decode_generated_ids(tokenizer, generate_ids, input_ids)

    return captions

def load_selected_model(model_type, custom_model_url=""):
    """
    Load the selected model type and initialize global model variables.
    Ensures previous models are unloaded before loading new ones.

    Args:
    model_type (str): Type of model to load ("Original", "4-bit Quantized", or "Custom")
    custom_model_url (str): URL of the custom model (used when model_type is "Custom")

    Returns:
    str: Status message indicating success or failure of model loading
    """
    global clip_processor, clip_model, tokenizer, text_model, image_adapter
    
    # Unload previous models
    if clip_model is not None:
        del clip_model
    if text_model is not None:
        del text_model
    if image_adapter is not None:
        del image_adapter
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    try:
        # Load CLIP model
        logger.info("Starting CLIP model loading")
        clip_processor, clip_model = load_clip()
        
        # Load selected LLM and tokenizer
        logger.info(f"Starting {model_type} model loading")
        if model_type == "Custom":
            if not custom_model_url:
                return "Error: Custom model URL is required for Custom model type"
            tokenizer, text_model = load_model(custom_model_url, use_4bit=False)
        else:
            model_path = QUANTIZED_MODEL_PATH if model_type == "4-bit Quantized" else ORIGINAL_MODEL_PATH
            tokenizer, text_model = load_model(model_path, use_4bit=(model_type == "4-bit Quantized"))
        
        # Load image adapter
        logger.info("Starting image adapter loading")
        image_adapter = load_image_adapter(clip_model, text_model, CHECKPOINT_PATH)
        
        logger.info("All models loaded successfully")
        return f"Successfully loaded {model_type} model"
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return f"Error loading model: {str(e)}"
    finally:
        # Clear CUDA cache again after loading
        torch.cuda.empty_cache()

def main():
    """
    Main function to set up and run the JoyCaption application.
    Parses command-line arguments, creates the UI, and launches the application.
    """
    parser = argparse.ArgumentParser(description="Run the JoyCaption demo")
    parser.add_argument("--listen", action="store_true", help="Listen on all interfaces")
    args = parser.parse_args()

    # Create Gradio UI
    demo = create_ui(
        stream_chat_func=stream_chat,
        batch_process_images_func=lambda input_folder, batch_size, seed, max_new_tokens, length_penalty, num_beams: batch_process_images(
            input_folder,
            batch_size,
            partial(stream_chat, seed=seed, max_new_tokens=max_new_tokens, length_penalty=length_penalty, num_beams=num_beams)
        ),
        interrupt_batch_processing_func=interrupt_batch_processing,
        load_model_func=load_selected_model
    )

    # Launch UI
    launch_ui(demo, args)

if __name__ == "__main__":
    main()