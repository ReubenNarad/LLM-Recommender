from rows import extract_rows
from random import randint

rows = extract_rows(2)


def format_preferences(user):
    likes = []
    dislikes = []
    reviews = list(user.values())[0]
    num_reviews = len(reviews)

    target = reviews.pop(randint(0, num_reviews - 1))
    truth = True if target['overall'] >= 4 else False
    target_title = target['title']
    
    for review in reviews:
        title = review['title']
        if review['overall'] >= 4:
            likes.append(title)
        else:
            dislikes.append(title)
    
    return {
        'likes': likes,
        'dislikes': dislikes,
        'target': target_title,
        'truth': truth
    }