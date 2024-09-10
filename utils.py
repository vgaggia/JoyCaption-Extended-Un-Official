def is_cuda_out_of_memory_error(error_message: str) -> bool:
    """
    Check if the error message indicates a CUDA out of memory error.
    
    Args:
    error_message (str): The error message to check
    
    Returns:
    bool: True if the error is a CUDA out of memory error, False otherwise
    """
    return "CUDA out of memory" in error_message

# Add any other utility functions here as needed