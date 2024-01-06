import json
from random import sample
from random import randint

def format_preferences(user, threshold: int = 4):
    """
    Categorizes a user's reviews into likes, dislikes, and selects a target review.

    Args:
    - user (dict): A dictionary with of a user's reviews containing a list of review dicts.
    - threshold (int, optional): Minimum rating defining a 'like'. Default is 4.

    Returns:
    - dict: Contains lists of 'likes' and 'dislikes', the 'target' review title, and 'truth' as a boolean.
    """
    
    likes, dislikes = [], []
    reviews = list(user.values())[0]
    num_reviews = len(reviews)

    # Randomly select a target review to use as ground truth
    target = reviews.pop(randint(0, num_reviews - 1))
    truth = target['overall'] >= threshold  # Boolean indicating if target review is a 'like'
    target_title = target['title']
    
    # Categorize remaining reviews based on threshold
    for review in reviews:
        if review['overall'] >= threshold:
            likes.append(review['title'])
        else:
            dislikes.append(review['title'])
    
    return {
        'likes': likes,
        'dislikes': dislikes,
        'target': target_title,
        'truth': truth
    }

def extract_rows(num_rows: int, data_path: str):
    '''
    Extracts a specified number of rows randomly from data

    Args:
    - num_rows (int): The number of rows to be randomly extracted.

    Returns:
    - list: A list of randomly selected rows from the JSON file.
    '''

    # Open the file and read JSON lines into a list
    with open(data_path, 'r') as file:
        # Parse each line in the file as JSON and store in a list
        lines = [json.loads(line) for line in file]

    # Retrieve 'num_rows' entries randomly from the list of lines
    extracted_rows = sample(lines, num_rows)

    return extracted_rows