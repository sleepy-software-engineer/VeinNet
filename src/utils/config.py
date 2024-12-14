# Set the path to the dataset where the images are stored based on your local setup.
DATASET_PATH = "/home/lucian/University/MSc-Courses/BiometricSystems/data/"
# Refers to the hand - left or right. [l, r]
HAND = "l"
# Refers to the spectrum of the image. [940, 850, 700, 630, 460, WHT]
SPECTRUM = "940"
# Seed for reproducibility.
SEED = 42
# NOTE: do not change the following values.
PATIENTS = [f"{i:03}" for i in range(1, 101)]
