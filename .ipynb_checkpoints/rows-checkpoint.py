import json
from random import sample

def extract_rows(num_rows):
    # Load the data
    filepath = 'data/video/formatted_video.json'

    with open(filepath, 'r') as file:
        lines = [json.loads(line) for line in file]

    # Retrieve 'num_rows' entries randomly
    extracted_rows = sample(lines, num_rows)

    return extracted_rows