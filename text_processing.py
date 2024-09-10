import torch

vlm_prompt = "A descriptive caption for this image:\n"
negative_vlm_prompt = ""

def set_vlm_prompt(new_prompt):
    """
    Set a new VLM prompt.
    
    Args:
    new_prompt (str): The new prompt to use
    """
    global vlm_prompt
    vlm_prompt = new_prompt

def get_vlm_prompt():
    """
    Get the current VLM prompt.
    
    Returns:
    str: The current VLM prompt
    """
    return vlm_prompt

def set_negative_vlm_prompt(new_negative_prompt):
    """
    Set a new negative VLM prompt.
    
    Args:
    new_negative_prompt (str): The new negative prompt to use
    """
    global negative_vlm_prompt
    negative_vlm_prompt = new_negative_prompt

def get_negative_vlm_prompt():
    """
    Get the current negative VLM prompt.
    
    Returns:
    str: The current negative VLM prompt
    """
    return negative_vlm_prompt

def tokenize_prompt(tokenizer):
    """
    Tokenize the VLM prompt and negative prompt.
    
    Args:
    tokenizer: The tokenizer object
    
    Returns:
    tuple: (torch.Tensor, torch.Tensor) Tokenized prompt and negative prompt
    """
    prompt_tokens = tokenizer.encode(vlm_prompt, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False)
    negative_prompt_tokens = tokenizer.encode(negative_vlm_prompt, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False) if negative_vlm_prompt else None
    return prompt_tokens, negative_prompt_tokens

def embed_prompt(text_model, tokenized_prompt, tokenized_negative_prompt=None):
    """
    Embed the tokenized prompt and negative prompt.
    
    Args:
    text_model: The text model object
    tokenized_prompt (torch.Tensor): Tokenized prompt
    tokenized_negative_prompt (torch.Tensor, optional): Tokenized negative prompt
    
    Returns:
    tuple: (torch.Tensor, torch.Tensor) Embedded prompt and negative prompt
    """
    prompt_embeds = text_model.model.embed_tokens(tokenized_prompt.to('cuda'))
    negative_prompt_embeds = text_model.model.embed_tokens(tokenized_negative_prompt.to('cuda')) if tokenized_negative_prompt is not None else None
    return prompt_embeds, negative_prompt_embeds

def embed_bos_token(text_model, tokenizer):
    """
    Embed the beginning-of-sequence token.
    
    Args:
    text_model: The text model object
    tokenizer: The tokenizer object
    
    Returns:
    torch.Tensor: Embedded BOS token
    """
    return text_model.model.embed_tokens(torch.tensor([[tokenizer.bos_token_id]], device=text_model.device, dtype=torch.int64))

def construct_input_tensors(embedded_bos, embedded_images, prompt_embeds, negative_prompt_embeds, tokenizer, prompt, negative_prompt):
    """
    Construct input tensors for the model.
    
    Args:
    embedded_bos (torch.Tensor): Embedded BOS token
    embedded_images (torch.Tensor): Embedded images
    prompt_embeds (torch.Tensor): Embedded prompt
    negative_prompt_embeds (torch.Tensor): Embedded negative prompt
    tokenizer: The tokenizer object
    prompt (torch.Tensor): Tokenized prompt
    negative_prompt (torch.Tensor): Tokenized negative prompt
    
    Returns:
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: inputs_embeds, input_ids, attention_mask
    """
    inputs_embeds = torch.cat([
        embedded_bos.expand(embedded_images.shape[0], -1, -1),
        embedded_images.to(dtype=embedded_bos.dtype),
        prompt_embeds.expand(embedded_images.shape[0], -1, -1),
    ], dim=1)

    input_ids = torch.cat([
        torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).expand(embedded_images.shape[0], -1),
        torch.zeros((embedded_images.shape[0], embedded_images.shape[1]), dtype=torch.long),
        prompt.expand(embedded_images.shape[0], -1),
    ], dim=1).to('cuda')

    attention_mask = torch.ones_like(input_ids)

    if negative_prompt_embeds is not None:
        negative_inputs_embeds = torch.cat([
            embedded_bos.expand(embedded_images.shape[0], -1, -1),
            embedded_images.to(dtype=embedded_bos.dtype),
            negative_prompt_embeds.expand(embedded_images.shape[0], -1, -1),
        ], dim=1)

        negative_input_ids = torch.cat([
            torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).expand(embedded_images.shape[0], -1),
            torch.zeros((embedded_images.shape[0], embedded_images.shape[1]), dtype=torch.long),
            negative_prompt.expand(embedded_images.shape[0], -1),
        ], dim=1).to('cuda')

        negative_attention_mask = torch.ones_like(negative_input_ids)

        return inputs_embeds, input_ids, attention_mask, negative_inputs_embeds, negative_input_ids, negative_attention_mask
    else:
        return inputs_embeds, input_ids, attention_mask, None, None, None

def decode_generated_ids(tokenizer, generate_ids, input_ids):
    """
    Decode the generated IDs into captions.
    
    Args:
    tokenizer: The tokenizer object
    generate_ids (torch.Tensor): Generated IDs
    input_ids (torch.Tensor): Input IDs
    
    Returns:
    List[str]: List of decoded captions
    """
    generate_ids = generate_ids[:, input_ids.shape[1]:]
    generate_ids = generate_ids[:, :(generate_ids != tokenizer.eos_token_id).cumsum(dim=-1).argmax(dim=-1).max() + 1]

    captions = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return [caption.strip() for caption in captions]