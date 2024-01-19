import json
from random import sample
from random import randint
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

# NOTE: format_preferences() truncates reviews to 1000 characters and history to 6 reviews.

def format_preferences(user, threshold: int = 4):
    """
    Categorizes a user's reviews into likes, dislikes, and selects a target review.

    Args:
    - user (dict): A dictionary with of a user's reviews containing a list of review dicts.
    - threshold (int, optional): Minimum rating defining a 'like'. Default is 4.

    Returns:
    - dict: Contains lists of 'likes' and 'dislikes', the 'target' review title, and 'truth' as a boolean.
    """
    
    likes, dislikes, history = [], [], []
    reviews = list(user.values())[0]
    num_reviews = len(reviews)

    # Randomly select a target review to use as ground truth
    target = reviews.pop(randint(0, num_reviews - 1))
    target_description = target['description']
    truth = target['overall'] >= threshold  # Boolean indicating if target review is a 'like'
    target_title = target['title']
    
    # Categorize remaining reviews based on threshold
    for i, review in enumerate(reviews):
        if review['overall'] >= threshold:
            likes.append(review['title'])
        else:
            dislikes.append(review['title'])
        
        if i > num_reviews - 8:
            history.append(f"Title: {review['title']}, Stars: {review['overall']}/5, Review Text: {review['reviewText'][:1000]}")
    
    return {
        'likes': likes,
        'dislikes': dislikes,
        'target': target_title,
        'target_description': target_description,
        'truth': truth,
        'history': history
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
    extracted_rows = lines[:num_rows]

    return extracted_rows

def evaluate(pred, truth):

    # Calculate F1 score
    f1 = f1_score(truth, pred)

    # Calculate recall
    recall = recall_score(truth, pred)

    # Calculate precision
    precision = precision_score(truth, pred)

    # Calculate ROC AUC
    try:
        roc_auc = roc_auc_score(truth, pred)
    except Exception as e:
        print(e)
        roc_auc = None

    # Print or use the calculated metrics
    print(f"F1 Score: {f1}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"ROC AUC: {roc_auc}")
    return f1, recall, precision, roc_auc

