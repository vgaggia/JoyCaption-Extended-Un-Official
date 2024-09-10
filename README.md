# JoyCaption Pre-Alpha Extended support (Un-Official repo)

## Description
JoyCaption is an image captioning application that uses advanced AI models to generate descriptive captions for images. This project is a heavily modified version of an original repository, adapted to use more recent models and technologies.

## Features
- Image captioning using state-of-the-art AI models
- Batch processing capabilities for multiple images
- User-friendly Gradio interface
- Support for different CUDA versions and CPU-only setups
- Custom model loading from Hugging Face

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA toolkit

### Setup
1. Clone this repository:
   ```
   git clone https://github.com/vgaggia/JoyCaption-Extended-Un-Official
   cd JoyCaption-Extended-Un-Official
   ```

2. Run the installation script:
   ```
   install.bat
   ```
   This script will create a virtual environment and install the necessary dependencies.

4. Install PyTorch:
   Open CMD and type nvcc --version
   Visit https://pytorch.org/get-started/locally/ and follow the instructions to install PyTorch with the appropriate CUDA version for your system.

5. Download JoyCaption image_adapter.pt from https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/tree/main/wpkklhc6
   Place it in the folder named "wpkklhc6"

## Usage

To start the application, run:
```
start.bat
```

This will launch the Gradio interface, where you can:
- Upload individual images for captioning
- Process batches of images
- Switch between different model types (Original, 4-bit Quantized, or Custom)
- Load custom models from Hugging Face

### Loading Custom Models
1. Select "Custom" from the Model Type radio buttons.
2. Enter the Hugging Face model identifier (e.g., "meta-llama/Meta-Llama-3.1-70B") in the Custom Model URL field.
3. Click "Load Model" to initialize the custom model.

## Project Structure
- `app.py`: Main application file
- `models.py`: Model loading and initialization
- `image_processing.py`: Image preparation and processing
- `text_processing.py`: Text tokenization and embedding
- `batch_processing.py`: Batch image processing functionality
- `gradio_ui.py`: Gradio interface setup
- `utils.py`: Utility functions
- `install.bat`: Installation script
- `start.bat`: Application startup script

## License
This project is licensed under the MIT License.

## Acknowledgements
This project is based on the original work by [Fancy Feast](https://huggingface.co/fancyfeast), with significant modifications and improvements. The original project was also released under the MIT License.

## Disclaimer
This is a un-official pre-alpha version and may contain bugs or unfinished features. Use at your own risk.

## Troubleshooting
If you encounter any issues with model loading or CUDA compatibility, ensure that you have installed the correct version of PyTorch for your CUDA setup. You can check your CUDA version by running `nvidia-smi` in the command prompt as well as `nvcc --version` to make sure you have cuda toolkit installed.
