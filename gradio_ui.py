import gradio as gr
from text_processing import get_vlm_prompt, set_vlm_prompt, get_negative_vlm_prompt, set_negative_vlm_prompt

TITLE = "<h1><center>JoyCaption Pre-Alpha Un-Official Extended Repo</center></h1>"

def create_ui(stream_chat_func, batch_process_images_func, interrupt_batch_processing_func, load_model_func):
    """
    Create the Gradio UI for the JoyCaption application.
    
    Args:
    stream_chat_func: Function to generate a caption for a single image
    batch_process_images_func: Function to process images in batches
    interrupt_batch_processing_func: Function to interrupt batch processing
    load_model_func: Function to load the selected model

    Returns:
    gr.Blocks: Gradio Blocks interface
    """
    with gr.Blocks() as demo:
        gr.HTML(TITLE)
        
        # Model selection and loading
        with gr.Row():
            model_type = gr.Radio(["Original", "4-bit Quantized", "Custom(8B models only)"], label="LLM Model Type", value="Original")
            custom_model_url = gr.Textbox(label="Custom Model URL (Hugging Face)", visible=False)
        
        load_model_button = gr.Button("Load Model")
        model_status = gr.Textbox(label="Model Loading Status")
        
        # VLM Prompt inputs (hidden from UI)
        vlm_prompt = gr.Textbox(label="VLM Prompt", value=get_vlm_prompt(), visible=False)
        negative_vlm_prompt = gr.Textbox(label="Negative VLM Prompt", value=get_negative_vlm_prompt(), visible=False)
        
        # Generation parameters
        with gr.Row():
            with gr.Column():
                seed = gr.Number(label="Seed", value=-1, precision=0, info="Set to -1 for random seed")
                max_new_tokens = gr.Number(label="Max New Tokens", value=300, precision=0, info="Maximum number of new tokens to generate")
            with gr.Column():    
                length_penalty = gr.Number(label="Length Penalty", value=1.0, precision=2, info="Encourages (>1.0) or discourages (<1.0) longer sequences")
                num_beams = gr.Number(label="Number of Beams", value=1, precision=0, info="Number of beams for beam search")

        # Image captioning interface
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
                caption_button = gr.Button("Caption")
            
            with gr.Column():
                output_caption = gr.Textbox(label="Caption")
        
        def caption_with_prompt(img, prompt, negative_prompt, seed, max_new_tokens, length_penalty, num_beams):
            set_vlm_prompt(prompt)
            set_negative_vlm_prompt(negative_prompt)
            return stream_chat_func([img], prompt, negative_prompt, seed, max_new_tokens, length_penalty, num_beams)[0]
        
        caption_button.click(fn=caption_with_prompt, inputs=[input_image, vlm_prompt, negative_vlm_prompt, seed, max_new_tokens, length_penalty, num_beams], outputs=[output_caption])
        
        # Batch processing interface
        gr.Markdown("## Batch Processing")
        with gr.Row():
            batch_input_folder = gr.Textbox(label="Batch Input Folder Path")
            batch_size = gr.Number(label="Batch Size", value=10, minimum=1, step=1)
            batch_process_button = gr.Button("Start Batch Processing")
            interrupt_button = gr.Button("Interrupt Processing")
        
        batch_output = gr.Textbox(label="Batch Processing Result")
        
        # Function to toggle visibility of custom model URL input
        def toggle_custom_model_url(choice):
            return gr.update(visible=choice == "Custom")
        
        # Event handlers
        model_type.change(fn=toggle_custom_model_url, inputs=[model_type], outputs=[custom_model_url])
        
        def batch_process_with_prompt(input_folder, batch_size, prompt, negative_prompt, seed, max_new_tokens, length_penalty, num_beams):
            set_vlm_prompt(prompt)
            set_negative_vlm_prompt(negative_prompt)
            return batch_process_images_func(input_folder, batch_size, seed, max_new_tokens, length_penalty, num_beams)
        
        batch_process_button.click(
            fn=batch_process_with_prompt,
            inputs=[batch_input_folder, batch_size, vlm_prompt, negative_vlm_prompt, seed, max_new_tokens, length_penalty, num_beams],
            outputs=[batch_output]
        )
        
        interrupt_button.click(
            fn=interrupt_batch_processing_func,
            inputs=[],
            outputs=[batch_output]
        )
        
        load_model_button.click(
            fn=load_model_func,
            inputs=[model_type, custom_model_url],
            outputs=[model_status]
        )

    return demo

def launch_ui(demo, args):
    """
    Launch the Gradio UI.
    
    Args:
    demo: Gradio Blocks interface
    args: Command-line arguments
    """
    if args.listen:
        demo.launch(server_name="0.0.0.0")
    else:
        demo.launch()