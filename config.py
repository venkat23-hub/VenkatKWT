import os
import torch

# === Base dataset directory ===
BASE_DIR = "/content/drive/MyDrive/Keyword_spotting_transformer/data"

# === Training folders ===
TRAIN_DIRS = [
    os.path.join(BASE_DIR, "processed_train_augmented_exercise_bike_0dB"),
    os.path.join(BASE_DIR, "processed_train_augmented_exercise_bike_10dB"),
    os.path.join(BASE_DIR, "processed_train_augmented_pinknoise_0dB"),
    os.path.join(BASE_DIR, "processed_train_augmented_pinknoise_10dB"),
    os.path.join(BASE_DIR, "processed_train_augmented_whitenoise_0dB"),
    os.path.join(BASE_DIR, "processed_train_augmented_whitenoise_10dB"),
]

# === Validation folders ===
VALID_DIRS = [
    os.path.join(BASE_DIR, "processed_valid_augmented_exercise_bike_0dB"),
    os.path.join(BASE_DIR, "processed_valid_augmented_exercise_bike_10dB"),
    os.path.join(BASE_DIR, "processed_valid_augmented_pinknoise_0dB"),
    os.path.join(BASE_DIR, "processed_valid_augmented_pinknoise_10dB"),
    os.path.join(BASE_DIR, "processed_valid_augmented_whitenoise_0dB"),
    os.path.join(BASE_DIR, "processed_valid_augmented_whitenoise_10dB"),
]

# === Test folders ===
# TEST_DIRS = [
#     os.path.join(BASE_DIR, "processed_test_augmented_exercise_bike_0dB"),
#     os.path.join(BASE_DIR, "processed_test_augmented_exercise_bike_10dB"),
#     os.path.join(BASE_DIR, "processed_test_augmented_pinknoise_0dB"),
#     os.path.join(BASE_DIR, "processed_test_augmented_pinknoise_10dB"),
#     os.path.join(BASE_DIR, "processed_test_augmented_whitenoise_0dB"),
#     os.path.join(BASE_DIR, "processed_test_augmented_whitenoise_10dB"),
# ]

# === Model Hyperparameters ===
N_MELS = 40  # Mel Spectrogram frequency bins
FIXED_TIME_DIM = 101  # Fixed time dimension for spectrograms
spectrogram_size = (N_MELS, FIXED_TIME_DIM)  # Define spectrogram size as a tuple
PATCH_SIZE = (40, 16)  # Patch size for transformer
DIM = 128  # Transformer model embedding dimension
DEPTH = 6  # Number of transformer blocks
HEADS = 8  # Attention heads
MLP_DIM = 256  # MLP hidden layer size
DROPOUT = 0.4  # Dropout probability
NUM_CLASSES = 30  # Number of keyword classes

# === Training Hyperparameters ===
EPOCHS = 20
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Label Dictionary ===
label_dict = {
    'bed': 0, 'bird': 1, 'cat': 2, 'dog': 3, 'down': 4, 'eight': 5, 'five': 6, 'four': 7,
    'go': 8, 'happy': 9, 'house': 10, 'left': 11, 'marvin': 12, 'nine': 13, 'no': 14,
    'off': 15, 'on': 16, 'one': 17, 'right': 18, 'seven': 19, 'sheila': 20, 'six': 21,
    'stop': 22, 'three': 23, 'tree': 24, 'two': 25, 'up': 26, 'wow': 27, 'yes': 28, 'zero': 29
}





# import os
# import torch
# # === Dataset Paths ===
# BASE_DIR = "/content/drive/MyDrive/Keyword_spotting_transformer/data"
# TRAIN_DIR = os.path.join(BASE_DIR, "processed_train_augmented_whitenoise")
# VALID_DIR = os.path.join(BASE_DIR, "processed_valid_augmented_whitenoise")
# TEST_DIR = os.path.join(BASE_DIR, "processed_test_augmented_whitenoise")


# # TRAIN_DIR = os.path.join(BASE_DIR, "train_processed")
# # VALID_DIR = os.path.join(BASE_DIR, "train_processed")
TEST_DIR = os.path.join(BASE_DIR, "test_processed")

# # === Model Hyperparameters ===
# N_MELS = 40  # Mel Spectrogram frequency bins
# FIXED_TIME_DIM = 101  # Fixed time dimension for spectrograms
# spectrogram_size = (N_MELS, FIXED_TIME_DIM)  # Define spectrogram size as a tuple
# PATCH_SIZE = (40, 16)  # Patch size for transformer
# DIM = 128  # Transformer model embedding dimension
# DEPTH = 6  # Number of transformer blocks
# HEADS = 8  # Attention heads
# MLP_DIM = 256  # MLP hidden layer size
# DROPOUT = 0.3  # Dropout probability
# NUM_CLASSES = 30  # Number of keyword classes

# # === Training Hyperparameters ===
# EPOCHS = 20
# LEARNING_RATE = 1e-5
# BATCH_SIZE = 16
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # === Label Dictionary ===
# label_dict = {
#     'bed': 0, 'bird': 1, 'cat': 2, 'dog': 3, 'down': 4, 'eight': 5, 'five': 6, 'four': 7,
#     'go': 8, 'happy': 9, 'house': 10, 'left': 11, 'marvin': 12, 'nine': 13, 'no': 14,
#     'off': 15, 'on': 16, 'one': 17, 'right': 18, 'seven': 19, 'sheila': 20, 'six': 21,
#     'stop': 22, 'three': 23, 'tree': 24, 'two': 25, 'up': 26, 'wow': 27, 'yes': 28, 'zero': 29
# }
