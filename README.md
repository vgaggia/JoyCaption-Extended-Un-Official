# JoyCaption Alpha two Extended support (Un-Official repo)

## Description
JoyCaption is an image captioning application that uses advanced AI models to generate descriptive captions for images. This project is a heavily modified version of an original repository, adapted to use more recent models and technologies.
If you would like to use the old version of this repo please switch your branch to alpha-1

## Features
- Batch processing capabilities for multiple images
- User-friendly Gradio interface
- Support for different CUDA versions and CPU-only setups
- Custom model loading from Hugging Face

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA toolkit (for GPU support)

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
   This script will create a virtual environment, install the necessary dependencies, and open a seperate command prompt window with instructions to install PyTorch with the appropriate CUDA version for your system.

3. The required model files are already included in the `cgrkzexw-599808` folder. This folder contains:
   - `clip_model.pt`
   - `config.yaml`
   - `image_adapter.pt`
   - `text_model` folder with necessary files

## Usage

To start the application, run:
```
start.bat
```

This will launch the application, where you can:
- Upload individual images for captioning
- Process batches of images
- Adjust settings as needed

## License
This project is licensed under the MIT License.

## Acknowledgements
This project is based on the original work by [Fancy Feast](https://huggingface.co/fancyfeast), with modifications. The original project was also released under the MIT License.

## Disclaimer
This is an unofficial pre-alpha version and may contain bugs or unfinished features. Use at your own risk.

## Troubleshooting
- If you encounter any issues with model loading or CUDA compatibility, ensure that you have run the install.bat script successfully.
- You can check your CUDA version by running `nvidia-smi` in the command prompt.
- Verify that the CUDA toolkit is installed by running `nvcc --version`.
- If you're having issues with specific modules or dependencies, make sure your virtual environment is activated and all requirements are installed correctly.
- For any persistent issues, please check the project's issue tracker or create a new issue with a detailed description of the problem.
