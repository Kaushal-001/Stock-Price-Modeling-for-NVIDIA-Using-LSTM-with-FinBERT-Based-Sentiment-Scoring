import os 
import pandas as pd

def load_data(file_path: str)-> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is empty or cannot be parsed.
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error loading file {file_path}: {e}")
    
    if data.empty:
        raise ValueError(f"The file {file_path} is empty.")
    
    return data