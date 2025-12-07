"""
Label Dictionary - 30 Keywords
Matches your training label dictionary exactly
"""

LABEL_DICT = {
    'bed': 0, 'bird': 1, 'cat': 2, 'dog': 3, 'down': 4, 'eight': 5, 'five': 6, 'four': 7,
    'go': 8, 'happy': 9, 'house': 10, 'left': 11, 'marvin': 12, 'nine': 13, 'no': 14,
    'off': 15, 'on': 16, 'one': 17, 'right': 18, 'seven': 19, 'sheila': 20, 'six': 21,
    'stop': 22, 'three': 23, 'tree': 24, 'two': 25, 'up': 26, 'wow': 27, 'yes': 28, 'zero': 29
}

# Reverse mapping: index to label
IDX_TO_LABEL = {v: k for k, v in LABEL_DICT.items()}

# List of keywords in order
KEYWORDS_LIST = [LABEL_DICT[k] for k in sorted(LABEL_DICT, key=LABEL_DICT.get)]
