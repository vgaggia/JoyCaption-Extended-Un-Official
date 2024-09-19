import os
import time
import threading
from typing import List, Tuple
from PIL import Image
import gradio as gr
import multiprocessing
from functools import partial

# Global variable to control the interrupt
interrupt_processing = threading.Event()

def get_image_text_pairs(input_folder: str) -> List[Tuple[str, str]]:
    """
    Create a list of tuples containing image file paths and corresponding text file paths (if they exist).
    
    Args:
    input_folder (str): Path to the folder containing images and text files
    
    Returns:
    List[Tuple[str, str]]: List of tuples (image_path, text_path)
    """
    image_text_pairs = []
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(input_folder, txt_filename)
            if os.path.exists(txt_path):
                image_text_pairs.append((image_path, txt_path))
            else:
                image_text_pairs.append((image_path, None))
    return image_text_pairs

def process_image(image_path):
    """
    Load and preprocess a single image.
    
    Args:
    image_path (str): Path to the image file
    
    Returns:
    PIL.Image or None: Processed image or None if there was an error
    """
    try:
        with Image.open(image_path) as img:
            return img.copy()
    except Exception as e:
        print(f"Error opening {image_path}: {str(e)}")
        return None

def batch_process_images(input_folder: str, batch_size: int, stream_chat_func, progress=gr.Progress()):
    global interrupt_processing
    interrupt_processing.clear()
    
    image_text_pairs = get_image_text_pairs(input_folder)
    total_images = len(image_text_pairs)
    
    progress(0, desc="Starting batch processing")
    
    start_time = time.time()
    processed_images = 0
    skipped_images = 0
    
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        for i in range(0, total_images, batch_size):
            if interrupt_processing.is_set():
                pool.terminate()
                return f"Batch processing interrupted. Processed {processed_images}/{total_images} images (including {skipped_images} skipped)."
            
            batch_pairs = image_text_pairs[i:i+batch_size]
            batch_image_paths = []
            for image_path, txt_path in batch_pairs:
                if txt_path is None:
                    batch_image_paths.append(image_path)
                else:
                    skipped_images += 1
                    print(f"Skipping {image_path} as text pair already exists")
            
            batch_images = list(filter(None, pool.map(process_image, batch_image_paths)))
            
            if batch_images:
                try:
                    captions = stream_chat_func(batch_images)
                    
                    for idx, (caption, image_path) in enumerate(zip(captions, batch_image_paths)):
                        if interrupt_processing.is_set():
                            return f"Processing interrupted. Processed {processed_images}/{total_images} images (including {skipped_images} skipped)."
                        
                        txt_filename = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
                        txt_path = os.path.join(input_folder, txt_filename)
                        
                        with open(txt_path, 'w', encoding='utf-8') as txt_file:
                            txt_file.write(caption)
                        
                        print(f"Processed {image_path} and generated {txt_filename}")
                        processed_images += 1
                        
                        progress((processed_images + skipped_images) / total_images, 
                                 desc=f"Processed {processed_images}/{total_images} images ({skipped_images} skipped)")
                    
                except Exception as e:
                    error_message = str(e)
                    print(f"Error processing batch: {error_message}")
                    if "CUDA out of memory" in error_message:
                        pool.terminate()
                        return "Error: CUDA out of memory. Please reduce the batch size and try again."
            
            elapsed_time = time.time() - start_time
            total_processed = processed_images + skipped_images
            
            if processed_images > 0:
                avg_time_per_image = elapsed_time / processed_images
                remaining_images = total_images - total_processed
                estimated_remaining_time = avg_time_per_image * remaining_images
                eta = time.strftime("%H:%M:%S", time.gmtime(estimated_remaining_time))
            else:
                eta = "Calculating..."
            
            progress((processed_images + skipped_images) / total_images, 
                     desc=f"Processed {processed_images}/{total_images} images ({skipped_images} skipped). ETA: {eta}")
    
    total_time = time.time() - start_time
    return f"Batch processing completed. Processed {processed_images} images, skipped {skipped_images} images, total {total_images} images in {total_time:.2f} seconds. Check the input folder for generated .txt files."

def interrupt_batch_processing():
    """
    Set the interrupt flag to stop batch processing.
    
    Returns:
    str: Status message
    """
    global interrupt_processing
    interrupt_processing.set()
    return "Interrupting batch processing. Please wait for the current batch to finish."