import spaces
import gradio as gr
from huggingface_hub import InferenceClient
from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
from pathlib import Path
import torch
import torch.amp.autocast_mode
from PIL import Image
import os
import torchvision.transforms.functional as TVF
import time
import warnings
import tempfile
import uuid

# Suppress the flash attention warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*Torch was not compiled with flash attention.*")

# Define constants for model paths and UI title
CLIP_PATH = "google/siglip-so400m-patch14-384"
CHECKPOINT_PATH = Path("cgrkzexw-599808")
TITLE = "<h1><center>JoyCaption Alpha Two (2024-09-26a)</center></h1>"

# Define different caption types and their corresponding prompts
CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a descriptive caption for this image in a formal tone.",
        "Write a descriptive caption for this image in a formal tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a formal tone.",
    ],
    "Descriptive (Informal)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Training Prompt": [
        "Write a stable diffusion prompt for this image.",
        "Write a stable diffusion prompt for this image within {word_count} words.",
        "Write a {length} stable diffusion prompt for this image.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Booru tag list": [
        "Write a list of Booru tags for this image.",
        "Write a list of Booru tags for this image within {word_count} words.",
        "Write a {length} list of Booru tags for this image.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}

# Get HuggingFace token from environment variable
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Define the ImageAdapter class for processing image features
class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int, ln1: bool, pos_emb: bool, num_image_tokens: int, deep_extract: bool):
        super().__init__()
        self.deep_extract = deep_extract

        # Adjust input features if using deep extraction
        if self.deep_extract:
            input_features = input_features * 5

        # Define neural network layers
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
        self.pos_emb = None if not pos_emb else nn.Parameter(torch.zeros(num_image_tokens, input_features))

        # Other tokens (<|image_start|>, <|image_end|>, <|eot_id|>)
        self.other_tokens = nn.Embedding(3, output_features)
        self.other_tokens.weight.data.normal_(mean=0.0, std=0.02)   # Matches HF's implementation of llama3

    def forward(self, vision_outputs: torch.Tensor):
        # Process the vision outputs
        if self.deep_extract:
            x = torch.concat((
                vision_outputs[-2],
                vision_outputs[3],
                vision_outputs[7],
                vision_outputs[13],
                vision_outputs[20],
            ), dim=-1)
            assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"  # batch, tokens, features
            assert x.shape[-1] == vision_outputs[-2].shape[-1] * 5, f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
        else:
            x = vision_outputs[-2]

        # Apply layer normalization and positional embedding if enabled
        x = self.ln1(x)
        if self.pos_emb is not None:
            assert x.shape[-2:] == self.pos_emb.shape, f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
            x = x + self.pos_emb

        # Apply linear layers and activation
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        # Add <|image_start|> and <|image_end|> tokens
        other_tokens = self.other_tokens(torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1))
        assert other_tokens.shape == (x.shape[0], 2, x.shape[2]), f"Expected {(x.shape[0], 2, x.shape[2])}, got {other_tokens.shape}"
        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

        return x

    def get_eot_embedding(self):
        # Get the embedding for the end-of-text token
        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)

# Load CLIP model
print("Loading CLIP")
clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
clip_model = AutoModel.from_pretrained(CLIP_PATH)
clip_model = clip_model.vision_model

# Load custom vision model checkpoint
assert (CHECKPOINT_PATH / "clip_model.pt").exists()
print("Loading VLM's custom vision model")
checkpoint = torch.load(CHECKPOINT_PATH / "clip_model.pt", map_location='cpu', weights_only=True)
checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
clip_model.load_state_dict(checkpoint)
del checkpoint

# Set CLIP model to evaluation mode and move to GPU
clip_model.eval()
clip_model.requires_grad_(False)
clip_model.to("cuda")

# Load tokenizer
print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH / "text_model", use_fast=True)
assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast), f"Tokenizer is of type {type(tokenizer)}"

# Load language model
print("Loading LLM")
print("Loading VLM's custom text model")
text_model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_PATH / "text_model", torch_dtype=torch.bfloat16).to("cuda").eval()
# Load image adapter
print("Loading image adapter")
image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size, False, False, 38, False)
image_adapter.load_state_dict(torch.load(CHECKPOINT_PATH / "image_adapter.pt", map_location="cpu", weights_only=True))
image_adapter.eval()
image_adapter.to("cuda")

def save_caption(image_path, caption: str, input_folder: str = None):
    """
    Save the caption to a text file with the same name as the image file.
    """
    try:
        if isinstance(image_path, str):
            # For batch processing or single file input, use the original image path
            if os.path.isabs(image_path):
                # If it's an absolute path, use it directly
                base_name = os.path.splitext(image_path)[0]
            else:
                # If it's a relative path, join it with the input_folder
                full_path = os.path.join(input_folder, image_path)
                base_name = os.path.splitext(full_path)[0]
            caption_file = f"{base_name}.txt"
        else:
            # For Gradio interface, use a temporary file name in the input folder or current directory
            if input_folder and os.path.isdir(input_folder):
                temp_dir = input_folder
            else:
                temp_dir = os.getcwd()  # Use current working directory as fallback
            temp_filename = f"temp_caption_{uuid.uuid4()}.txt"
            caption_file = os.path.join(temp_dir, temp_filename)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(caption_file), exist_ok=True)
        
        with open(caption_file, "w", encoding="utf-8") as f:
            f.write(caption)
        print(f"Caption saved to: {caption_file}")
    except Exception as e:
        print(f"Error saving caption: {str(e)}")
        print(f"Attempted to save to: {caption_file}")

@spaces.GPU()
@torch.no_grad()
def stream_chat(input_images, input_folder: str, caption_type: str, caption_length: str | int, extra_option: str, name_input: str, custom_prompt: str, batch_size: int = 5) -> tuple[str, list[tuple[str, str]], str]:
    torch.cuda.empty_cache()
    start_time = time.time()

    # Determine caption length
    length = None if caption_length == "any" else caption_length

    if isinstance(length, str):
        try:
            length = int(length)
        except ValueError:
            pass
    
    # Build prompt based on caption type and length
    if length is None:
        map_idx = 0
    elif isinstance(length, int):
        map_idx = 1
    elif isinstance(length, str):
        map_idx = 2
    else:
        raise ValueError(f"Invalid caption length: {length}")
    
    prompt_str = CAPTION_TYPE_MAP[caption_type][map_idx]

    # Add extra option to prompt
    if extra_option != "None":
        prompt_str += " " + extra_option
    
    # Add name, length, word_count to prompt
    prompt_str = prompt_str.format(name=name_input, length=caption_length, word_count=caption_length)

    # Use custom prompt if provided
    if custom_prompt.strip() != "":
        prompt_str = custom_prompt.strip()
    
    # For debugging
    print(f"Prompt: {prompt_str}")

    # Determine if processing a single image or a batch
    if input_images is not None:
        # Single image processing
        if isinstance(input_images, str):
            image_files = [input_images]
            batch_images = [Image.open(input_images).convert('RGB')]
        else:
            # Gradio Image component
            image_files = [input_images]
            batch_images = [input_images]
        total_images = 1
    else:
        # Batch processing
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        total_images = len(image_files)
        batch_images = None  # We'll load these in batches

    all_captions = []
    progress_text = ""

    # Process images in batches
    for i in range(0, total_images, batch_size):
        if batch_images is None:
            batch_files = image_files[i:i+batch_size]
            batch_images = []
            for f in batch_files:
                try:
                    img = Image.open(os.path.join(input_folder, f)).convert('RGB')
                    batch_images.append(img)
                except Exception as e:
                    print(f"Error opening image {f}: {e}")
                    all_captions.append((f, f"Error: {str(e)}"))
            
            # Adjust batch_size if the last batch is smaller
            current_batch_size = len(batch_images)
        else:
            batch_files = image_files  # For single image case
            current_batch_size = 1
        
        if not batch_images:
            continue  # Skip this iteration if no valid images were loaded

        # Preprocess image batch
        pixel_values = []
        for image in batch_images:
            if isinstance(image, gr.components.Image):
                image = Image.fromarray(image.image)
            image = image.resize((384, 384), Image.LANCZOS)
            pv = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
            pv = TVF.normalize(pv, [0.5], [0.5])
            pixel_values.append(pv)
        
        pixel_values = torch.cat(pixel_values, dim=0).to('cuda')

        # Embed image batch
        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            vision_outputs = clip_model(pixel_values=pixel_values, output_hidden_states=True)
            embedded_images = image_adapter(vision_outputs.hidden_states)
            embedded_images = embedded_images.to('cuda')
        
        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt_str,
            },
        ]

        # Format the conversation
        convo_string = tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = True)
        assert isinstance(convo_string, str)

        # Tokenize the conversation
        convo_tokens = tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False, truncation=False)
        prompt_tokens = tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False, truncation=False)
        assert isinstance(convo_tokens, torch.Tensor) and isinstance(prompt_tokens, torch.Tensor)
        convo_tokens = convo_tokens.squeeze(0)
        prompt_tokens = prompt_tokens.squeeze(0)

        # Calculate where to inject the image
        eot_id_indices = (convo_tokens == tokenizer.convert_tokens_to_ids("<|eot_id|>")).nonzero(as_tuple=True)[0].tolist()
        assert len(eot_id_indices) == 2, f"Expected 2 <|eot_id|> tokens, got {len(eot_id_indices)}"

        preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]

        # Embed the tokens
        convo_embeds = text_model.model.embed_tokens(convo_tokens.unsqueeze(0).to('cuda'))

        # Construct the input for each image in the batch
        batch_input_embeds = []
        batch_input_ids = []
        batch_attention_mask = []

        for img_embed in embedded_images:
            # Combine conversation embeddings with image embeddings
            input_embeds = torch.cat([
                convo_embeds[:, :preamble_len],
                img_embed.unsqueeze(0).to(dtype=convo_embeds.dtype),
                convo_embeds[:, preamble_len:],
            ], dim=1).to('cuda')
            
            # Create input IDs with placeholder zeros for image tokens
            input_ids = torch.cat([
                convo_tokens[:preamble_len].unsqueeze(0),
                torch.zeros((1, img_embed.shape[0]), dtype=torch.long),
                convo_tokens[preamble_len:].unsqueeze(0),
            ], dim=1).to('cuda')
            
            # Create attention mask
            attention_mask = torch.ones_like(input_ids)
            
            batch_input_embeds.append(input_embeds)
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)

        # Stack batch inputs
        batch_input_embeds = torch.cat(batch_input_embeds, dim=0)
        batch_input_ids = torch.cat(batch_input_ids, dim=0)
        batch_attention_mask = torch.cat(batch_attention_mask, dim=0)

        # Generate captions for the batch
        try:
            generate_ids = text_model.generate(
                batch_input_ids,
                inputs_embeds=batch_input_embeds,
                attention_mask=batch_attention_mask,
                max_new_tokens=300,
                do_sample=True,
                suppress_tokens=None,
                num_return_sequences=1,
            )

            # Trim off the prompt for each generated caption
            generate_ids = generate_ids[:, batch_input_ids.shape[1]:]
            
            # Process each generated caption
            batch_captions = []
            for j, file_name in enumerate(batch_files):
                try:
                    caption_ids = generate_ids[j]
                    eos_token_id = tokenizer.eos_token_id
                    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    
                    # Find the first occurrence of either eos_token_id or eot_token_id
                    end_indices = torch.where((caption_ids == eos_token_id) | (caption_ids == eot_token_id))[0]
                    
                    if len(end_indices) > 0:
                        end_index = end_indices[0].item()
                        caption_ids = caption_ids[:end_index]
                    
                    # Decode the caption
                    caption = tokenizer.decode(caption_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    caption = caption.strip()
                    batch_captions.append(caption)
                    
                    # Save caption to file
                    save_caption(file_name, caption, input_folder)
                except Exception as e:
                    print(f"Error processing caption for image {file_name}: {e}")
                    batch_captions.append(f"Error: {str(e)}")

            all_captions.extend(list(zip(batch_files, batch_captions)))

        except Exception as e:
            print(f"Error processing batch: {e}")
            batch_captions = [f"Error: {str(e)}"] * len(batch_files)
            all_captions.extend(list(zip(batch_files, batch_captions)))

        # Update progress
        progress = (i + current_batch_size) / total_images
        elapsed_time = time.time() - start_time
        eta = (elapsed_time / progress) * (1 - progress) if progress > 0 else 0
        progress_text = f"Processing: {progress:.2%} complete | ETA: {eta:.2f} seconds"
        yield prompt_str, all_captions, progress_text

        if input_images is not None:
            break  # Exit after processing single image

        # Clear batch_images for the next iteration
        batch_images = None

    progress_text = "Processing: 100% complete"
    yield prompt_str, all_captions, progress_text


# Define the Gradio interface
with gr.Blocks() as demo:
    gr.HTML(TITLE)

    with gr.Row():
        with gr.Column(scale=1):
            # Input components
            input_image = gr.Image(label="Input Image", type="pil")
            input_folder = gr.Textbox(label="Input Folder Path (for batch processing)")

            caption_type = gr.Dropdown(
                choices=["Descriptive", "Descriptive (Informal)", "Training Prompt", "MidJourney", "Booru tag list", "Booru-like tag list", "Art Critic", "Product Listing", "Social Media Post"],
                label="Caption Type",
                value="Descriptive",
            )

            caption_length = gr.Dropdown(
                choices=["any", "very short", "short", "medium-length", "long", "very long"] +
                        [str(i) for i in range(20, 261, 10)],
                label="Caption Length",
                value="long",
            )

            extra_option = gr.Dropdown(
                choices=[
                    "None",
                    "If there is a person/character in the image you must refer to them as {name}.",
                    "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
                    "Include information about lighting.",
                    "Include information about camera angle.",
                    "Include information about whether there is a watermark or not.",
                    "Include information about whether there are JPEG artifacts or not.",
                    "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
                    "Do NOT include anything sexual; keep it PG.",
                    "Do NOT mention the image's resolution.",
                    "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
                    "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
                    "Do NOT mention any text that is in the image.",
                    "Specify the depth of field and whether the background is in focus or blurred.",
                    "If applicable, mention the likely use of artificial or natural lighting sources.",
                    "Do NOT use any ambiguous language.",
                    "Include whether the image is sfw, suggestive, or nsfw.",
                    "ONLY describe the most important elements of the image."
                ],
                label="Extra Option",
                value="None",
            )

            name_input = gr.Textbox(label="Person/Character Name (if applicable)")
            gr.Markdown("**Note:** Name input is only used if the Extra Option requires it.")

            custom_prompt = gr.Textbox(label="Custom Prompt (optional, will override all other settings)")

            batch_size = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Batch Size")

            single_caption_button = gr.Button("Caption Single Image")
            batch_caption_button = gr.Button("Batch Caption")
        
        with gr.Column(scale=2):
            # Output components
            output_prompt = gr.Textbox(label="Prompt that was used")
            output_captions = gr.Dataframe(headers=["Image", "Caption"], label="Captions")
            progress_bar = gr.Textbox(label="Progress")
    
    def process_output(prompt, captions, progress):
        return prompt, captions, progress

    # Connect the buttons to the stream_chat function
    single_caption_button.click(fn=stream_chat, inputs=[input_image, gr.Textbox(value=""), caption_type, caption_length, extra_option, name_input, custom_prompt, batch_size], outputs=[output_prompt, output_captions, progress_bar], postprocess=process_output)
    batch_caption_button.click(fn=stream_chat, inputs=[gr.Image(value=None), input_folder, caption_type, caption_length, extra_option, name_input, custom_prompt, batch_size], outputs=[output_prompt, output_captions, progress_bar], postprocess=process_output)


if __name__ == "__main__":
    demo.launch()