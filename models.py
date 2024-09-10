# models.py

import torch
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
import time

"""
This module contains functions and classes for loading and initializing various models
used in the JoyCaption application, including CLIP, LLM, tokenizer, and image adapter.
"""

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for model paths
CLIP_PATH = "google/siglip-so400m-patch14-384"

class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
    
    def forward(self, vision_outputs: torch.Tensor):
        x = self.linear1(vision_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        return x

    @classmethod
    def from_pretrained(cls, checkpoint_path, input_features, output_features):
        model = cls(input_features, output_features)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        
        # Resize weights if necessary
        for name, param in model.state_dict().items():
            if name in state_dict:
                if state_dict[name].shape != param.shape:
                    logger.warning(f"Resizing {name} from {state_dict[name].shape} to {param.shape}")
                    if len(param.shape) == 2:  # For weight matrices
                        state_dict[name] = cls._resize_2d(state_dict[name], param.shape)
                    else:  # For bias vectors
                        state_dict[name] = cls._resize_1d(state_dict[name], param.shape[0])
        
        model.load_state_dict(state_dict, strict=False)
        return model

    @staticmethod
    def _resize_2d(tensor, new_shape):
        return nn.functional.interpolate(
            tensor.unsqueeze(0).unsqueeze(0),
            size=new_shape,
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)

    @staticmethod
    def _resize_1d(tensor, new_size):
        return nn.functional.interpolate(
            tensor.unsqueeze(0).unsqueeze(0),
            size=(new_size,),
            mode='linear',
            align_corners=False
        ).squeeze(0).squeeze(0)

def load_clip():
    """Load and initialize the CLIP model"""
    logger.info("Starting CLIP model loading")
    start_time = time.time()
    
    logger.info("Loading CLIP processor")
    clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
    
    logger.info("Loading CLIP model")
    clip_model = AutoModel.from_pretrained(CLIP_PATH)
    clip_model = clip_model.vision_model
    clip_model.eval()
    clip_model.requires_grad_(False)
    
    logger.info("Moving CLIP model to CUDA")
    clip_model.to('cuda')
    
    end_time = time.time()
    logger.info(f"CLIP model loading completed in {end_time - start_time:.2f} seconds")
    return clip_processor, clip_model

def load_tokenizer(model_path):
    """
    Load and initialize the tokenizer for the specified model.

    Args:
    model_path (str): Path to the model

    Returns:
    AutoTokenizer: Initialized tokenizer
    """
    logger.info(f"Starting tokenizer loading for {model_path}")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    
    end_time = time.time()
    logger.info(f"Tokenizer loading completed in {end_time - start_time:.2f} seconds")
    return tokenizer

def load_llm(model_path, use_4bit=False):
    """Load and initialize the LLM"""
    logger.info(f"Starting LLM loading: {model_path}")
    start_time = time.time()
    
    if use_4bit:
        logger.info("Configuring 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        logger.info("Loading 4-bit quantized model")
        text_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )
    else:
        logger.info("Loading full-precision model")
        text_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    
    text_model.eval()
    
    end_time = time.time()
    logger.info(f"LLM loading completed in {end_time - start_time:.2f} seconds")
    return text_model

def load_image_adapter(clip_model, text_model, checkpoint_path):
    """Load and initialize the image adapter"""
    logger.info("Starting image adapter loading")
    start_time = time.time()
    
    input_features = clip_model.config.hidden_size
    output_features = text_model.config.hidden_size
    
    logger.info(f"Loading image adapter weights from {checkpoint_path / 'image_adapter.pt'}")
    image_adapter = ImageAdapter.from_pretrained(
        checkpoint_path / "image_adapter.pt",
        input_features,
        output_features
    )
    image_adapter.eval()
    
    logger.info("Moving image adapter to CUDA")
    image_adapter.to("cuda")
    
    end_time = time.time()
    logger.info(f"Image adapter loading completed in {end_time - start_time:.2f} seconds")
    return image_adapter

def load_model(model_path, use_4bit=False):
    """
    Load the selected model and tokenizer

    Args:
    model_path (str): Path or identifier of the model to load
    use_4bit (bool): Whether to use 4-bit quantization

    Returns:
    tuple: (tokenizer, llm)
    """
    logger.info(f"Starting model loading process for {model_path}")
    start_time = time.time()
    
    tokenizer = load_tokenizer(model_path)
    llm = load_llm(model_path, use_4bit)
    
    end_time = time.time()
    logger.info(f"Total model loading time: {end_time - start_time:.2f} seconds")
    
    return tokenizer, llm