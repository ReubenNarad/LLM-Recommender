import json
from random import sample

def extract_rows(num_rows):
    '''
    Extracts a specified number of rows randomly from data

    Args:
    - num_rows (int): The number of rows to be randomly extracted.

    Returns:
    - list: A list of randomly selected rows from the JSON file.
    '''
    
    # TODO: correct file path to be swapped out
    
    # Path to the JSON file
    filepath = '../data/video/formatted_video.json'

    # Open the file and read JSON lines into a list
    with open(filepath, 'r') as file:
        # Parse each line in the file as JSON and store in a list
        lines = [json.loads(line) for line in file]

    # Retrieve 'num_rows' entries randomly from the list of lines
    extracted_rows = sample(lines, num_rows)

    return extracted_rows